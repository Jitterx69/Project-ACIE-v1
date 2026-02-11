"""
ACIE SDK Quick Start Example

Demonstrates basic usage of the ACIE SDK for counterfactual inference.
"""

from acie_sdk import ACIEClient
import numpy as np

def main():
    print("ACIE SDK Quick Start\n" + "=" * 50)
    
    # Initialize client
    print("\n1. Connecting to ACIE API...")
    client = ACIEClient(
        api_url="http://localhost:8080",
        api_key="your-api-key-here",  #  Optional
        timeout=30
    )
    
    # Health check
    print("\n2. Checking server health...")
    try:
        health = client.health_check()
        print(f"✓ Server status: {health.get('status', 'unknown')}")
        print(f"  Models loaded: {health.get('models_loaded', [])}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return
    
    # Single inference
    print("\n3. Performing counterfactual inference...")
    observation = np.random.randn(6000).tolist()  # Mock observation
    intervention = {"mass": 1.5, "metallicity": 0.02}
    
    try:
        result = client.infer(
            observation=observation,
            intervention=intervention,
            model_version="latest"
        )
        
        print("✓ Inference successful!")
        print(f"  Counterfactual shape: {len(result['counterfactual'])} dimensions")
        print(f"  Latent state shape: {len(result['latent_state'])} dimensions")
        print(f"  Confidence: {result['confidence']:.4f}")
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
    
    # Get model info
    print("\n4. Fetching model information...")
    try:
        model_info = client.get_model_info("latest")
        print(f"✓ Model version: {model_info.get('version', 'unknown')}")
        print(f"  Architecture: {model_info.get('architecture', 'N/A')}")
        print(f"  Training date: {model_info.get('trained_at', 'N/A')}")
    except Exception as e:
        print(f"✗ Failed to get model info: {e}")
    
    # List available models
    print("\n5. Listing available models...")
    try:
        models = client.list_models()
        print(f"✓ Found {len(models)} model(s):")
        for model in models:
            print(f"  - {model.get('version', 'unknown')}: {model.get('description', 'No description')}")
    except Exception as e:
        print(f"✗ Failed to list models: {e}")
    
    # Get metrics
    print("\n6. Fetching system metrics...")
    try:
        metrics = client.get_metrics()
        print(f"✓ Metrics retrieved:")
        print(f"  Total requests: {metrics.get('total_requests', 'N/A')}")
        print(f"  Avg latency: {metrics.get('avg_latency_ms', 'N/A')}ms")
        print(f"  GPU utilization: {metrics.get('gpu_utilization', 'N/A')}%")
    except Exception as e:
        print(f"ℹ️  Metrics not available: {e}")
    
    # Close client
    client.close()
    print("\n" + "=" * 50)
    print("✓ Quick start complete!")


if __name__ == "__main__":
    main()
