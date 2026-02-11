# ACIE API Reference 📚

## Overview

The ACIE API provides access to:
- **Inference**: High-throughput counterfactual generation.
- **Management**: Model registry and deployment control.
- **Monitoring**: Metrics and health checks.

**Base URL**: `http://localhost:8080`

---

## Authentication

All API requests (except health checks) require an API Key.

**Header**: `Authorization: Bearer <YOUR_API_KEY>`

---

## Endpoints

### Inference 🔮

#### POST `/api/v2/inference/counterfactual`
Run counterfactual inference on a single observation.

**Request Body**:
```json
{
  "observation": [0.1, 0.5, ...],
  "intervention": {
    "mass": 1.5
  },
  "model_version": "latest"
}
```

**Response**:
```json
{
  "counterfactual": [0.2, 0.6, ...],
  "latent_state": [0.01, -0.05, ...],
  "confidence": 0.98
}
```

#### POST `/api/v2/inference/batch`
Run inference on a batch of observations.

---

### Models 📦

#### GET `/api/v2/models`
List all registered models.

#### GET `/api/v2/models/{version}`
Get details for a specific model version.

---

### System ⚙️

#### GET `/health`
Check system status.
**Response**: `{"status": "healthy", "uptime": 1234}`

#### GET `/api/metrics`
Prometheus metrics endpoint.

---

## Python SDK 🐍

Use the provided SDK for easier integration.

```python
from acie_sdk.client import ACIEClient

client = ACIEClient("http://localhost:8080", api_key="key")
result = client.infer(obs, {"mass": 2.0})
```

See [examples/sdk_demo.py](examples/sdk_demo.py) for a complete example.
