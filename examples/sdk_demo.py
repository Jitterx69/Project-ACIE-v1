"""
Demo script for ACIE Python SDK
"""

import sys
import os
import time
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from acie_sdk.client import ACIEClient
from acie_sdk.exceptions import ACIEConnectionError

def main():
    print("🚀 ACIE SDK Demo")
    print("----------------")
    
    # Initialize client
    client = ACIEClient("http://localhost:8080", api_key="test-key")
    
    # 1. Check Health
    print("\n1. Checking Server Health...")
    try:
        health = client.health_check()
        print(f"✅ Server is online: {health}")
    except ACIEConnectionError:
        print("❌ Server is offline. Please start the server with 'acie serve'")
        # Continue for demo purposes if server is down, but normally we'd exit
        print("   (Skipping remaining specific calls due to connection failure)")
        return

    # 2. List Models
    print("\n2. Listing Available Models...")
    try:
        models = client.list_models()
        print(f"✅ Found {len(models)} models")
        for m in models[:3]:
            print(f"   - {m.get('name')} (v{m.get('version')})")
    except Exception as e:
        print(f"❌ Failed to list models: {e}")

    # 3. Running Inference
    print("\n3. Running Counterfactual Inference...")
    try:
        # Simulate observation (6000 dim)
        obs = [random.gauss(0, 1) for _ in range(100)] # Reduced for demo
        intervention = {"mass": 1.5, "metallicity": 0.02}
        
        print(f"   Input: Observation (len={len(obs)}), Intervention={intervention}")
        
        start = time.time()
        result = client.infer(obs, intervention)
        latency = (time.time() - start) * 1000
        
        print(f"✅ Inference successful ({latency:.1f}ms)")
        print(f"   Confidence: {result.get('confidence', 0.0):.4f}")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")

if __name__ == "__main__":
    main()
