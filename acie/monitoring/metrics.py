"""
Prometheus metrics instrumentation for ACIE API
Provides detailed monitoring of inference performance and system health
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi import Response
import time
import psutil
import torch
from functools import wraps

# ============================================================================
# Metrics Definitions
# ============================================================================

# Request counters
inference_requests_total = Counter(
    'acie_inference_requests_total',
    'Total number of inference requests',
    ['model_version', 'endpoint', 'status']
)

# Latency histograms
inference_latency_seconds = Histogram(
    'acie_inference_latency_seconds',
    'Inference request latency in seconds',
    ['model_version', 'endpoint'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

# Batch size tracking
batch_size_histogram = Histogram(
    'acie_batch_size',
    'Distribution of batch sizes',
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
)

# GPU metrics
gpu_utilization_percent = Gauge(
    'acie_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

gpu_memory_used_bytes = Gauge(
    'acie_gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['gpu_id']
)

gpu_memory_total_bytes = Gauge(
    'acie_gpu_memory_total_bytes',
    'GPU total memory in bytes',
    ['gpu_id']
)

gpu_temperature_celsius = Gauge(
    'acie_gpu_temperature_celsius',
    'GPU temperature in Celsius',
    ['gpu_id']
)

# System metrics
cpu_utilization_percent = Gauge(
    'acie_cpu_utilization_percent',
    'CPU utilization percentage'
)

memory_used_bytes = Gauge(
    'acie_memory_used_bytes',
    'System memory used in bytes'
)

memory_total_bytes = Gauge(
    'acie_memory_total_bytes',
    'System total memory in bytes'
)

# Model metrics
model_accuracy = Gauge(
    'acie_model_accuracy',
    'Model accuracy score',
    ['model_version', 'dataset']
)

active_models = Gauge(
    'acie_active_models',
    'Number of loaded models'
)

# Error tracking
error_counter = Counter(
    'acie_errors_total',
    'Total number of errors',
    ['error_type', 'endpoint']
)

# ============================================================================
# Metric Collection Functions
# ============================================================================

def update_gpu_metrics():
    """Update GPU metrics if CUDA is available"""
    if not torch.cuda.is_available():
        return
    
    try:
        for i in range(torch.cuda.device_count()):
            # Memory stats
            mem_allocated = torch.cuda.memory_allocated(i)
            mem_reserved = torch.cuda.memory_reserved(i)
            mem_total = torch.cuda.get_device_properties(i).total_memory
            
            gpu_memory_used_bytes.labels(gpu_id=str(i)).set(mem_allocated)
            gpu_memory_total_bytes.labels(gpu_id=str(i)).set(mem_total)
            
            # Utilization (approximation using memory)
            utilization = (mem_allocated / mem_total) * 100 if mem_total > 0 else 0
            gpu_utilization_percent.labels(gpu_id=str(i)).set(utilization)
            
    except Exception as e:
        print(f"Error updating GPU metrics: {e}")


def update_system_metrics():
    """Update system-level metrics"""
    try:
        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_utilization_percent.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_bytes.set(memory.used)
        memory_total_bytes.set(memory.total)
        
    except Exception as e:
        print(f"Error updating system metrics: {e}")


def track_inference(model_version: str, endpoint: str):
    """
    Decorator to track inference metrics
    
    Usage:
        @track_inference("latest", "counterfactual")
        def inference_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error_counter.labels(
                    error_type=type(e).__name__,
                    endpoint=endpoint
                ).inc()
                raise
            finally:
                # Record metrics
                latency = time.time() - start_time
                inference_latency_seconds.labels(
                    model_version=model_version,
                    endpoint=endpoint
                ).observe(latency)
                
                inference_requests_total.labels(
                    model_version=model_version,
                    endpoint=endpoint,
                    status=status
                ).inc()
                
                # Update system metrics
                update_gpu_metrics()
                update_system_metrics()
        
        return wrapper
    return decorator


async def metrics_endpoint():
    """Endpoint handler for Prometheus metrics"""
    # Update metrics before serving
    update_gpu_metrics()
    update_system_metrics()
    
    # Generate metrics in Prometheus format
    metrics_output = generate_latest()
    
    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================================
# Helper Functions
# ============================================================================

def set_model_count(count: int):
    """Set the number of active models"""
    active_models.set(count)


def record_batch_size(size: int):
    """Record a batch size"""
    batch_size_histogram.observe(size)


def record_model_accuracy(model_version: str, dataset: str, accuracy: float):
    """Record model accuracy"""
    model_accuracy.labels(model_version=model_version, dataset=dataset).set(accuracy)


def record_error(error_type: str, endpoint: str):
    """Record an error"""
    error_counter.labels(error_type=error_type, endpoint=endpoint).inc()
