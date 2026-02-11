"""
Benchmark models: Eager vs TorchScript vs Quantized
Reports latency, throughput, and memory usage.
"""

import torch
import time
import psutil
import os
import argparse
import numpy as np
from typing import Dict, Any, List

# Import ACIE components (mocking model for now if not available)
try:
    from acie.core.acie_core import ACIECore
    from acie.optimization import quantize_dynamic, prune_model, export_to_torchscript
except ImportError:
    # Minimal mock for testing benchmark structure
    import torch.nn as nn
    class ACIECore(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
        def forward(self, x):
            return self.net(x)
            
        @classmethod
        def load_from_checkpoint(cls, path):
            return cls()
            
    from acie.optimization import quantize_dynamic, prune_model, export_to_torchscript


def measure_performance(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 10,
    num_runs: int = 100
) -> Dict[str, float]:
    """Measure latency and throughput"""
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
            
    # Benchmark
    latencies = []
    
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_runs):
            iter_start = time.time()
            _ = model(input_tensor)
            latencies.append((time.time() - iter_start) * 1000)  # ms
        
        total_time = time.time() - start_time
        
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    throughput = num_runs / total_time
    
    return {
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "throughput_req_s": throughput
    }


def get_model_size(model: torch.nn.Module, temp_path: str = "temp_model.pt") -> float:
    """Get model size in MB"""
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size_mb


def main():
    parser = argparse.ArgumentParser(description="ACIE Model Benchmark")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--runs", type=int, default=1000, help="Number of runs")
    args = parser.parse_args()
    
    print(f"🚀 Benchmarking ACIE (Batch Size: {args.batch_size})...")
    print("-" * 60)
    
    # Create dummy input
    input_shape = (args.batch_size, 10)  # Adjust based on real model
    input_tensor = torch.randn(input_shape)
    
    # 1. Base Model
    print("Loading base model...")
    base_model = ACIECore() #.load_from_checkpoint("path/to/model")
    base_model.eval()
    
    base_metrics = measure_performance(base_model, input_tensor, num_runs=args.runs)
    base_size = get_model_size(base_model)
    
    print(f"Base Model (FP32):")
    print(f"  Latency (P95): {base_metrics['p95_latency_ms']:.4f} ms")
    print(f"  Throughput:    {base_metrics['throughput_req_s']:.2f} req/s")
    print(f"  Size:          {base_size:.2f} MB")
    print()
    
    # 2. Quantized Model
    print("Quantizing model...")
    quant_model = quantize_dynamic(base_model)
    
    quant_metrics = measure_performance(quant_model, input_tensor, num_runs=args.runs)
    quant_size = get_model_size(quant_model) # Note: state_dict save might not reflect quantization size accurately without script
    
    print(f"Quantized Model (INT8):")
    print(f"  Latency (P95): {quant_metrics['p95_latency_ms']:.4f} ms")
    print(f"  Throughput:    {quant_metrics['throughput_req_s']:.2f} req/s")
    print(f"  Speedup:       {quant_metrics['throughput_req_s']/base_metrics['throughput_req_s']:.2f}x")
    print()
    
    # 3. TorchScript Model
    print("Tracing model (TorchScript)...")
    ts_model = torch.jit.trace(base_model, input_tensor)
    
    ts_metrics = measure_performance(ts_model, input_tensor, num_runs=args.runs)
    
    print(f"TorchScript Model:")
    print(f"  Latency (P95): {ts_metrics['p95_latency_ms']:.4f} ms")
    print(f"  Throughput:    {ts_metrics['throughput_req_s']:.2f} req/s")
    print(f"  Speedup:       {ts_metrics['throughput_req_s']/base_metrics['throughput_req_s']:.2f}x")
    print()
    
    print("-" * 60)
    print("Benchmarking Complete")


if __name__ == "__main__":
    main()
