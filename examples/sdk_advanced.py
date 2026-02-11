"""
ACIE SDK Advanced Example

Demonstrates advanced SDK features including:
- Batch inference
- Context manager usage
- Error handling
- Custom configurations
"""

from acie_sdk import ACIEClient, ACIEError
import numpy as np
import time


def batch_inference_example():
    """Demonstrate batch inference with multiple observations."""
    print("\n📦 Batch Inference Example")
    print("-" * 50)
    
    with ACIEClient("http://localhost:8080") as client:
        # Generate multiple observations
        n_samples = 10
        observations = [np.random.randn(6000).tolist() for _ in range(n_samples)]
        interventions = [{"mass": 1.0 + i * 0.1} for i in range(n_samples)]
        
        print(f"Processing {n_samples} observations...")
        start_time = time.time()
        
        try:
            results = client.batch_infer(observations, interventions)
            elapsed = time.time() - start_time
            
            print(f"✓ Batch inference complete!")
            print(f"  Processed: {len(results)} samples")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Throughput: {len(results)/elapsed:.1f} samples/sec")
            
            # Analyze results
            confidences = [r['confidence'] for r in results]
            print(f"  Avg confidence: {np.mean(confidences):.4f}")
            print(f"  Min confidence: {np.min(confidences):.4f}")
            print(f"  Max confidence: {np.max(confidences):.4f}")
            
        except ACIEError as e:
            print(f"✗ Batch inference failed: {e}")


def error_handling_example():
    """Demonstrate robust error handling."""
    print("\n⚠️  Error Handling Example")
    print("-" * 50)
    
    client = ACIEClient(
        api_url="http://localhost:8080",
        max_retries=2,
        timeout=10
    )
    
    # Test with invalid data
    print("Testing error handling with invalid data...")
    
    try:
        result = client.infer(
            observation=[],  # Empty observation (invalid)
            intervention={"invalid_var": 999}
        )
    except ACIEError as e:
        print(f"✓ Caught expected error: {type(e).__name__}")
        print(f"  Message: {e}")
    
    # Test with non-existent model
    print("\nTesting with non-existent model...")
    try:
        result = client.get_model_info("non-existent-model")
    except ACIEError as e:
        print(f"✓ Caught expected error: {type(e).__name__}")
        print(f"  Message: {e}")
    
    client.close()


def performance_monitoring():
    """Monitor inference performance over multiple calls."""
    print("\n📊 Performance Monitoring Example")
    print("-" * 50)
    
    with ACIEClient("http://localhost:8080") as client:
        latencies = []
        n_runs = 20
        
        print(f"Running {n_runs} inference calls...")
        observation = np.random.randn(6000).tolist()
        intervention = {"mass": 1.5}
        
        for i in range(n_runs):
            start = time.time()
            try:
                result = client.infer(observation, intervention)
                latency = (time.time() - start) * 1000  # Convert to ms
                latencies.append(latency)
                
                if (i + 1) % 5 == 0:
                    print(f"  Progress: {i+1}/{n_runs} ({(i+1)/n_runs*100:.0f}%)")
                    
            except ACIEError as e:
                print(f"  Request {i+1} failed: {e}")
        
        if latencies:
            print(f"\n✓ Performance results:")
            print(f"  Mean latency: {np.mean(latencies):.2f}ms")
            print(f"  Median latency: {np.median(latencies):.2f}ms")
            print(f"  P95 latency: {np.percentile(latencies, 95):.2f}ms")
            print(f"  P99 latency: {np.percentile(latencies, 99):.2f}ms")
            print(f"  Min latency: {np.min(latencies):.2f}ms")
            print(f"  Max latency: {np.max(latencies):.2f}ms")


def custom_configuration():
    """Demonstrate custom client configurations."""
    print("\n⚙️  Custom Configuration Example")
    print("-" * 50)
    
    # Client with custom settings
    client = ACIEClient(
        api_url="http://localhost:8080",
        api_key="custom-api-key",
        timeout=60,  # Longer timeout
        max_retries=5,  # More retries
        verify_ssl=True  # SSL verification
    )
    
    print("✓ Client configured with custom settings:")
    print(f"  API URL: {client.api_url}")
    print(f"  Timeout: {client.timeout}s")
    print(f"  Max retries: {client.max_retries}")
    print(f"  SSL verification: {client.verify_ssl}")
    
    client.close()


def main():
    """Run all advanced examples."""
    print("\n" + "=" * 60)
    print("ACIE SDK Advanced Examples")
    print("=" * 60)
    
    try:
        batch_inference_example()
    except Exception as e:
        print(f"Batch inference example error: {e}")
    
    try:
        error_handling_example()
    except Exception as e:
        print(f"Error handling example error: {e}")
    
    try:
        performance_monitoring()
    except Exception as e:
        print(f"Performance monitoring example error: {e}")
    
    try:
        custom_configuration()
    except Exception as e:
        print(f"Custom configuration example error: {e}")
    
    print("\n" + "=" * 60)
    print("✓ All examples complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
