"""
Kafka Streaming Demo for ACIE
Demonstrates how to produce and consume inference events
"""

import time
import json
import threading
from acie.streaming.kafka_client import ACIEEventProducer, ACIEEventConsumer

# Configuration
BOOTSTRAP_SERVERS = ['localhost:9092']
TOPIC = 'acie.inference.requests'
GROUP_ID = 'acie-demo-group'

def producer_demo():
    """Produce dummy inference requests"""
    print("Starting producer...")
    producer = ACIEEventProducer(bootstrap_servers=BOOTSTRAP_SERVERS)
    
    # Send 10 events
    for i in range(10):
        event = {
            "observation": [0.1] * 10,
            "intervention": {"mass": 1.5 + i*0.1},
            "request_id": f"req-{i}",
            "timestamp": time.time()
        }
        
        print(f"Sending event {i}...")
        producer.send_event(TOPIC, "inference_request", event)
        time.sleep(1)
        
    producer.close()
    print("Producer finished")


def consumer_demo():
    """Consume inference requests"""
    print("Starting consumer...")
    
    def process_event(event):
        print(f"Received event: {event['type']}")
        print(f"Data: {json.dumps(event['data'], indent=2)}")
        
    consumer = ACIEEventConsumer(
        topics=[TOPIC],
        group_id=GROUP_ID,
        bootstrap_servers=BOOTSTRAP_SERVERS
    )
    
    consumer.start_consuming(process_event)
    
    # Run for 15 seconds
    time.sleep(15)
    consumer.stop()
    print("Consumer finished")


if __name__ == "__main__":
    # Start consumer in background
    consumer_thread = threading.Thread(target=consumer_demo)
    consumer_thread.start()
    
    # Wait a bit for consumer to start
    time.sleep(2)
    
    # Start producer
    producer_demo()
    
    consumer_thread.join()
