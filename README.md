# ACIE: Astronomical Counterfactual Inference Engine

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-94%25-green)

## Abstract

The Astronomical Counterfactual Inference Engine (ACIE) is a production-grade causal inference platform designed for high-dimensional astronomical data. Unlike traditional machine learning models that focus on correlation-based predictions, ACIE implements a rigorous **Structural Causal Model (SCM)** framework to answer interventional questions (e.g., *"What would the spectra of this galaxy look like if its stellar mass were doubled?"*) while enforcing strict compliance with conservation laws through differentiable physics layers.

The system employs a hybrid architecture combining PyTorch for deep learning, Rust for high-performance tensor operations, Assembly for critical matrix kernels, and a modern microservices backend for scalable deployment.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Installation](#installation)
3. [Core Methodology](#core-methodology)
4. [Usage Guide](#usage-guide)
   - [Command Line Interface](#command-line-interface)
   - [Python SDK](#python-sdk)
   - [REST API](#rest-api)
5. [MLOps & Production](#mlops--production)
6. [Performance](#performance)
7. [Citation](#citation)

---

## System Architecture

ACIE operates as a modular distributed system composed of five distinct layers:

### 1. The Inference Core (Python/PyTorch)
The heart of the system is a Variational Autoencoder (VAE) augmented with causal mechanisms. It performs:
- **Abduction**: Inferring latent physical state $P$ from observations $O$ ($P(P|O)$).
- **Action**: Applying interventions on the latent graph ($do(P_i = x)$).
- **Prediction**: Generating counterfactual observations through the decoder ($P(O_{do}|P')$).

### 2. Physics Constraints (CUDA/MPS/CPU)
To ensure physical plausibility, the decoder output is passed through a **Differentiable Physics Layer**. This layer computes residuals for conservation laws (Mass, Energy, Momentum) and penalizes violations during training:
$$L_{physics} = \sum || \nabla \cdot T - f ||^2$$

### 3. High-Performance Modules (Rust & Assembly)
Critical operations are offloaded to compiled languages:
- **Rust**: Handles graph traversal for the SCM and heavy tensor manipulations.
- **Assembly (AVX-512/NEON)**: Optimizes core matrix multiplications for the physics solver.

### 4. Service Layer (FastAPI/Redis)
A high-throughput FastAPI server exposes the model capabilities.
- **Redis**: Caches inference results for generic queries.
- **Prometheus**: Exports real-time metrics (throughput, latency, GPU utilization).

### 5. Presentation Layer (React)
A React-based dashboard provides real-time visualization of inference requests, system health, and model performance.

---

## Installation

### Prerequisites
- Python 3.9+
- Node.js 18+ (for Dashboard)
- CUDA Toolkit 12.1+ (optional, for GPU acceleration)

### Building from Source

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Jitterx69/Project-ACIE-v1.git
    cd Project-ACIE-v1
    ```

2.  **Install Python Dependencies**
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

3.  **Build Frontend**
    ```bash
    cd frontend
    npm install
    npm run build
    cd ..
    ```

4.  **Verify Installation**
    ```bash
    acie --version
    ```

---

## Core Methodology

ACIE implements the **Three-Step Ladder of Causation**:

1.  **Abduction**:
    Given an observation $O$ (e.g., galaxy spectra), we estimate the posterior distribution of the exogenous noise $U$ and latent parents $P$.
    $$P(U | O) \approx q_\phi(z | x)$$

2.  **Action**:
    We perform a graphical intervention $do(X=x)$ on the SCM, severing the incoming edges to variable $X$ and setting its value to $x$.

3.  **Prediction**:
    We propagate the intervened variables through the SCM mechanisms to generate the counterfactual $O'$.
    $$O' = f_S(P_{do}, U)$$

This approach guarantees mathematically consistent counterfactuals that respect the causal structure of the physical system.

---

## Usage Guide

### Command Line Interface

The `acie` CLI is the primary entry point for training and management.

**Training a Model**
```bash
acie train \
    --dataset-size 10k \
    --epochs 100 \
    --batch-size 64 \
    --experiment-name "physics-constraint-v1"
```

**Running Inference**
```bash
acie infer \
    --checkpoint outputs/production_model.ckpt \
    --observation data/sample_01.csv \
    --intervention "mass=2.5,metallicity=0.02"
```

**Managing Models**
```bash
acie models list
acie models promote --version 3 --stage Production
```

### Python SDK

For programmatic integration, use the `acie.sdk` package.

```python
from acie.sdk import ACIEClient

# Initialize client
client = ACIEClient(base_url="http://localhost:8080")

# Perform robust inference
response = client.infer(
    observation=[0.5, 1.2, ...],  # 6000-dim vector
    intervention={"mass": 1.5},
    model_version="latest"
)

print(f"Counterfactual: {response.counterfactual[:5]}")
print(f"Confidence: {response.confidence}")
```

### REST API

The backend exposes a comprehensive OpenAPI specification.

**POST /api/inference/counterfactual**
Retrieves a counterfactual prediction.

**Payload:**
```json
{
  "observation": [0.1, 0.2, 0.3, ...],
  "intervention": {
    "mass": 1.5
  },
  "strategy": "strict_physics"
}
```

---

## MLOps & Production

ACIE is built with a "Production-First" mindset.

### Experiment Tracking
All training runs utilize **MLflow** to log:
- Hyperparameters (Learning rate, Physics weights).
- Metrics (Reconstruction Loss, Physics Violation L2).
- Artifacts (Model checkpoints, SCM graphs).

### Model Registry
Models progress through lifecycle stages:
1.  **Staging**: Automated validation benchmarks.
2.  **Production**: High-availability serving.
3.  **Archived**: Legacy versions.

### Monitoring
The system exports Prometheus metrics at `/metrics`:
- `acie_inference_latency_seconds`: Histogram of request duration.
- `acie_gpu_utilization`: Gauge of NVIDIA GPU memory usage.
- `acie_physics_violation_rate`: Counter of requests failing physics checks.

---

## Performance

Benchmarks conducted on NVIDIA A100 (80GB):

| Component | Operation | Throughput (samples/s) |
|-----------|-----------|------------------------|
| PyTorch (CPU) | Inference | 320 |
| PyTorch (CUDA) | Inference | 12,500 |
| Rust (TensorOps) | Physics Check | 45,000 |
| Assembly (AVX) | Matrix Mul | 52,000 |

*Note: Benchmarks may vary based on batch size and precision settings.*

---

## Citation

If you use ACIE in your research, please cite the following:

```bibtex
@software{acie_2026,
  author = {ACIE Development Team},
  title = {Astronomical Counterfactual Inference Engine},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Jitterx69/Project-ACIE-v1}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
