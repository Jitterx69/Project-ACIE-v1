"""
Kafka Client Wrapper for ACIE
Provides EventProducer and EventConsumer for streaming inference
"""

import json
import logging
from typing import Dict, Any, Callable, Optional, List
import threading
import time

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
except ImportError:
    KafkaProducer = None
    KafkaConsumer = None
    KafkaError = Exception

logger = logging.getLogger(__name__)

class ACIEEventProducer:
    """Kafka Producer for ACIE events"""
    
    def __init__(self, bootstrap_servers: List[str] = ['localhost:9092']):
        self.producer = None
        self.bootstrap_servers = bootstrap_servers
        if KafkaProducer:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    acks='all',
                    retries=3
                )
                logger.info(f"Connected to Kafka at {bootstrap_servers}")
            except Exception as e:
                logger.error(f"Failed to connect to Kafka producer: {e}")
        else:
            logger.warning("kafka-python not installed")

    def send_event(self, topic: str, event_type: str, data: Dict[str, Any]):
        """Send event to Kafka topic"""
        if not self.producer:
            logger.warning("Producer not initialized, skipping event")
            return

        message = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        
        try:
            future = self.producer.send(topic, message)
            # future.get(timeout=10) # Don't block for high throughput
        except Exception as e:
            logger.error(f"Failed to send event: {e}")

    def close(self):
        if self.producer:
            self.producer.close()


class ACIEEventConsumer:
    """Kafka Consumer for ACIE events"""
    
    def __init__(
        self, 
        topics: List[str], 
        group_id: str,
        bootstrap_servers: List[str] = ['localhost:9092'],
        auto_offset_reset: str = 'latest'
    ):
        self.consumer = None
        self.running = False
        self.thread = None
        
        if KafkaConsumer:
            try:
                self.consumer = KafkaConsumer(
                    *topics,
                    bootstrap_servers=bootstrap_servers,
                    group_id=group_id,
                    auto_offset_reset=auto_offset_reset,
                    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
                )
                logger.info(f"Connected to Kafka consumer (group: {group_id})")
            except Exception as e:
                logger.error(f"Failed to connect to Kafka consumer: {e}")
        else:
            logger.warning("kafka-python not installed")

    def start_consuming(self, callback: Callable[[Dict[str, Any]], None]):
        """Start consuming messages in a background thread"""
        if not self.consumer:
            return

        self.running = True
        self.thread = threading.Thread(target=self._consume_loop, args=(callback,))
        self.thread.daemon = True
        self.thread.start()

    def _consume_loop(self, callback: Callable[[Dict[str, Any]], None]):
        """Internal consumption loop"""
        logger.info("Started consumption loop")
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                try:
                    event = message.value
                    callback(event)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except Exception as e:
            logger.error(f"Consumer loop error: {e}")
        finally:
            logger.info("Stopped consumption loop")

    def stop(self):
        """Stop consuming"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.consumer:
            self.consumer.close()


# Global Instances (Lazy loaded)
_producer = None

def get_producer(bootstrap_servers: List[str] = None):
    """Get global producer instance"""
    global _producer
    if _producer is None:
        servers = bootstrap_servers or ['localhost:9092']
        _producer = ACIEEventProducer(bootstrap_servers=servers)
    return _producer
