# ACIE System Expansion - Quick Start Guide

## 🚀 What Was Added

This expansion adds production-grade infrastructure to ACIE:

### 1. GPU/CUDA Acceleration ⚡
- **config/gpu_config.yaml** - Multi-GPU training configuration
- Mixed precision (FP16), distributed training (DDP)
- 10-100x training speedup potential

### 2. Modern FastAPI Server 🌐
- **acie/api/fastapi_server.py** - Async REST API
- Auto-generated OpenAPI docs at `/docs`
- Batch inference support
- GPU-accelerated inference

### 3. Production Deployment 🐳
- **Dockerfile.production** - Optimized multi-stage build
- **docker-compose.production.yml** - Full stack (API, Redis, Postgres, Prometheus, Grafana)
- **k8s/deployment.yaml** - Kubernetes with auto-scaling

### 4. Monitoring & Observability 📊
- **acie/monitoring/metrics.py** - Prometheus instrumentation
- GPU/CPU/memory metrics
- Request latency tracking
- Grafana dashboards

## 🏃 Quick Start

### Option 1: Docker Compose (Easiest)

```bash
# 1. Build and start all services
docker-compose -f docker-compose.production.yml up --build

# 2. Access services:
# - API: http://localhost:8080
# - Docs: http://localhost:8080/docs
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

### Option 2: Run FastAPI Locally

```bash
# 1. Install new dependencies
pip install -r requirements.txt

# 2. Start FastAPI server
python -m uvicorn acie.api.fastapi_server:app --host 0.0.0.0 --port 8080 --reload

# 3. Open docs: http://localhost:8080/docs
```

### Option 3: GPU Training

```bash
# Train with GPU acceleration
python -m acie.cli train --config config/gpu_config.yaml --gpus 1
```

## 📚 API Usage Examples

### Python SDK:
```python
import requests

# Counterfactual inference
response = requests.post("http://localhost:8080/api/v2/inference/counterfactual", json={
    "observation": [0.1] * 6000,
    "intervention": {"mass": 1.5},
    "model_version": "latest"
})

result = response.json()
print(f"Counterfactual: {result['counterfactual'][:5]}...")
print(f"Latency: {result['latency_ms']:.2f}ms")
```

### cURL:
```bash
curl -X POST "http://localhost:8080/api/v2/inference/counterfactual" \
  -H "Content-Type: application/json" \
  -d '{
    "observation": [0.1, 0.2, ...],
    "intervention": {"mass": 1.5}
  }'
```

## 🎯 Next Steps

See **EXPANSION_ROADMAP.md** for complete implementation plan including:
- Kafka streaming
- gRPC service
- Model quantization
- Advanced monitoring
- Security & authentication

## 📖 Documentation
- API Docs: http://localhost:8080/docs
- Expansion Roadmap: `EXPANSION_ROADMAP.md`
- Task Tracking: See artifact `task.md`
