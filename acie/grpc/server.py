"""
gRPC Server implementation for ACIE
Provides high-performance inter-service communication
"""

import grpc
from concurrent import futures
import time
import logging
import torch
import numpy as np
from typing import Dict, Any

# Import generated gRPC code
try:
    from acie.grpc import acie_pb2
    from acie.grpc import acie_pb2_grpc
except ImportError:
    # Handle case where code isn't generated yet (during initial setup)
    logging.warning("gRPC code not found. Please run protoc generation.")
    acie_pb2 = None
    acie_pb2_grpc = None

from acie.core.acie_core import ACIECore
from acie.inference.counterfactual import CounterfactualEngine
from acie.cache import get_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("acie.grpc")

class InferenceServicer(acie_pb2_grpc.InferenceServiceServicer if acie_pb2_grpc else object):
    """
    gRPC Servicer for ACIE Inference
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.cf_engine = None
        self.model_path = model_path
        self._load_model()
        
        # Initialize cache
        self.cache = get_cache()
        if self.cache.is_available():
            logger.info(f"Connected to Redis cache at {self.cache.host}:{self.cache.port}")
            
    def _load_model(self):
        """Load the ACIE model"""
        try:
            import os
            if not self.model_path:
                self.model_path = os.getenv("MODEL_PATH", "outputs/acie_final.ckpt")
            
            logger.info(f"Loading model from {self.model_path}")
            if os.path.exists(self.model_path):
                self.model = ACIECore.load_from_checkpoint(self.model_path)
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                    logger.info("Model moved to GPU")
                
                self.model.eval()
                self.cf_engine = CounterfactualEngine(self.model)
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model file not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def CounterfactualInference(self, request, context):
        """
        Handle single counterfactual inference request
        """
        start_time = time.time()
        
        try:
            # Check cache
            if self.cache.is_available():
                # specific cache key generation for gRPC request
                # (converting request to dict for cache key)
                intervention_dict = dict(request.intervention)
                cached = self.cache.get_cached_inference(
                    list(request.observation),
                    intervention_dict,
                    request.model_version
                )
                if cached:
                    return acie_pb2.InferenceResponse(
                        counterfactual=cached['counterfactual'],
                        latent_state=cached['latent_state'],
                        confidence=cached['confidence'],
                        model_version=request.model_version,
                        latency_ms=(time.time() - start_time) * 1000,
                        request_id=request.request_id
                    )

            if not self.model or not self.cf_engine:
                context.abort(grpc.StatusCode.UNAVAILABLE, "Model not loaded")
                
            # Prepare input
            obs_tensor = torch.tensor(request.observation, dtype=torch.float32).unsqueeze(0)
            if torch.cuda.is_available():
                obs_tensor = obs_tensor.cuda()
                
            # Intervention map
            intervention = dict(request.intervention)
            
            # Inference
            with torch.no_grad():
                result = self.cf_engine.generate_counterfactual(obs_tensor, intervention)
                
            # Format response
            response = acie_pb2.InferenceResponse(
                counterfactual=result['counterfactual'].cpu().squeeze(0).tolist(),
                latent_state=result['latent'].cpu().squeeze(0).tolist(),
                confidence=float(result.get('confidence', 0.95)),
                model_version=request.model_version,
                latency_ms=(time.time() - start_time) * 1000,
                request_id=request.request_id
            )
            
            # Cache result
            if self.cache.is_available():
                response_data = {
                    "counterfactual": list(response.counterfactual),
                    "latent_state": list(response.latent_state),
                    "confidence": response.confidence
                }
                self.cache.cache_inference_result(
                    list(request.observation),
                    intervention,
                    request.model_version,
                    response_data
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def StreamInference(self, request_iterator, context):
        """
        Handle streaming inference requests
        """
        for request in request_iterator:
            yield self.CounterfactualInference(request, context)

    def HealthCheck(self, request, context):
        """
        Health check endpoint
        """
        return acie_pb2.HealthResponse(
            status="serving" if self.model else "not_serving",
            gpu_available=torch.cuda.is_available(),
            gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            version="2.1.0"
        )


def serve(port: int = 50051, max_workers: int = 10):
    """Start gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    acie_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    
    server.start()
    logger.info(f"gRPC server started on port {port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
