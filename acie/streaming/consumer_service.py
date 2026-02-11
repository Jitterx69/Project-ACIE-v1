"""
Kafka Streaming Consumer Service
Consumes inference requests from Kafka, processes them, and publishes results
"""

import logging
import json
import time
import torch
import os
from typing import Dict, Any

from acie.streaming.kafka_client import ACIEEventConsumer, ACIEEventProducer, get_producer
from acie.core.acie_core import ACIECore
from acie.inference.counterfactual import CounterfactualEngine
from acie.logging import logger

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(',')
INPUT_TOPIC = os.getenv('KAFKA_INPUT_TOPIC', 'acie.inference.requests')
OUTPUT_TOPIC = os.getenv('KAFKA_OUTPUT_TOPIC', 'acie.inference.results')
GROUP_ID = os.getenv('KAFKA_GROUP_ID', 'acie-inference-group')
MODEL_PATH = os.getenv('MODEL_PATH', 'outputs/acie_final.ckpt')

class StreamingInferenceService:
    """Service to process streaming inference requests"""
    
    def __init__(self):
        self.model = None
        self.cf_engine = None
        self._load_model()
        
        self.producer = ACIEEventProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
        self.consumer = ACIEEventConsumer(
            topics=[INPUT_TOPIC],
            group_id=GROUP_ID,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS
        )
        
    def _load_model(self):
        """Load the model"""
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            if os.path.exists(MODEL_PATH):
                self.model = ACIECore.load_from_checkpoint(MODEL_PATH)
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                self.model.eval()
                self.cf_engine = CounterfactualEngine(self.model)
                logger.info("Model loaded successfully")
            else:
                logger.error(f"Model not found at {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def process_event(self, event: Dict[str, Any]):
        """Process a single event"""
        try:
            event_type = event.get('type')
            data = event.get('data', {})
            
            if event_type == 'inference_request':
                self._handle_inference(data)
            else:
                logger.warning(f"Unknown event type: {event_type}")
                
        except Exception as e:
            logger.error(f"Error processing event: {e}")

    def _handle_inference(self, data: Dict[str, Any]):
        """Handle inference request"""
        if not self.model or not self.cf_engine:
            logger.error("Model not loaded, skipping inference")
            return
            
        start_time = time.time()
        request_id = data.get('request_id')
        
        try:
            observation = data.get('observation')
            intervention = data.get('intervention')
            model_version = data.get('model_version', 'latest')
            
            # Prepare tensor
            obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            if torch.cuda.is_available():
                obs_tensor = obs_tensor.cuda()
                
            # Run inference
            with torch.no_grad():
                result = self.cf_engine.generate_counterfactual(obs_tensor, intervention)
                
            # Prepare response
            response = {
                "request_id": request_id,
                "counterfactual": result['counterfactual'].cpu().squeeze(0).tolist(),
                "latent_state": result['latent'].cpu().squeeze(0).tolist(),
                "confidence": float(result.get('confidence', 0.95)),
                "model_version": model_version,
                "latency_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time()
            }
            
            # Publish result
            logger.info(f"Processed request {request_id}, sending result")
            self.producer.send_event(OUTPUT_TOPIC, "inference_result", response)
            
        except Exception as e:
            logger.error(f"Inference failed for request {request_id}: {e}")
            # Optionally send error event

    def run(self):
        """Run the service"""
        logger.info(f"Starting streaming inference service on {INPUT_TOPIC}")
        self.consumer.start_consuming(self.process_event)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping service...")
            self.consumer.stop()
            self.producer.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    service = StreamingInferenceService()
    service.run()
