# ACIE - Astronomical Counterfactual Inference Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Java](https://img.shields.io/badge/Java-17%2B-red.svg)](https://www.oracle.com/java/)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🚀 **A multi-language, physics-constrained, causal inference system for astronomical observations that estimates interventional and counterfactual distributions over astrophysical data.**

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Datasets](#-datasets)
- [Quick Start](#-quick-start)
- [Multi-Language Components](#-multi-language-components)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Mathematical Foundation](#-mathematical-foundation)
- [Citation](#-citation)
- [License](#-license)

## 🌟 Overview

ACIE is an advanced deep learning system that performs **counterfactual inference** on astronomical observations. Unlike traditional ML systems that predict correlations, ACIE answers causal questions like:

> *"What would the observable properties of this galaxy be if its initial mass were 1.5× higher?"*

The system combines **Python**, **Java**, **Rust**, **Assembly**, and **R** to deliver high-performance causal inference with physics-based constraints.

## ✨ Key Features

- **🔍 Causal Reasoning**: Explicit structural causal models (SCM) with intervention operators
- **⚛️ Physics-Constrained**: Enforces conservation laws and stability constraints via differentiable physics layers
- **🔄 Counterfactual Inference**: 3-step abduction-action-prediction pipeline
- **👁️ Partial Observability**: Infers latent physical states from incomplete observations
- **📊 Identifiability Optimization**: Maximizes causal identifiability under constraints
- **🚄 High Performance**: Multi-language architecture with Assembly-level optimizations
- **🌐 RESTful API**: Java Spring Boot server for production inference
- **📈 Statistical Analysis**: R-based analytics and interactive Shiny dashboards

## 🏗️ Architecture

ACIE is built on a sophisticated multi-language architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    Python Core (PyTorch)                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ SCM Engine  │  │ VAE Inference│  │ Counterfactual │  │
│  │   (acie/)   │  │   Network    │  │    Pipeline    │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
└───────────┬─────────────────────────────────────────────┘
            │
    ┌───────┴───────┬──────────────┬──────────────┐
    │               │              │              │
┌───▼────┐    ┌────▼─────┐   ┌───▼─────┐   ┌────▼────┐
│  Rust  │    │   Java   │   │   ASM   │   │    R    │
│ Tensor │    │ REST API │   │ Matrix  │   │Analysis │
│  Ops   │    │  Server  │   │ Kernels │   │Visuals  │
└────────┘    └──────────┘   └─────────┘   └─────────┘
```

### Component Interactions

1. **Python Core**: Main training pipeline, SCM, and inference engine
2. **Rust**: High-performance tensor operations, graph algorithms, physics simulations
3. **Java**: Production-ready REST API server with Python bridge
4. **Assembly**: Ultra-fast matrix kernels for critical operations
5. **R**: Statistical analysis, visualization, and interactive dashboards

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Java 17+ (for API server)
- Rust 1.70+ (for performance modules)
- NASM (for assembly modules)
- R 4.0+ (for analytics)
- Maven 3.6+ (for Java builds)

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Jitterx69/Project-ACIE-v1.git
cd Project-ACIE-v1

# 2. Install Python dependencies
pip install -r requirements.txt
pip install -e .

# 3. Build Rust components
cd rust
cargo build --release
cd ..

# 4. Build Assembly modules
cd asm
make
cd ..

# 5. Build Java API server
cd java
mvn clean package
cd ..

# 6. (Optional) Install R packages
Rscript -e "install.packages(c('tidyverse', 'ggplot2', 'shiny', 'plotly'))"
```

### Quick Build (Using Makefile)

```bash
# Build all components
make all

# Run tests across all languages
make test
```

## 📦 Datasets

ACIE requires large-scale synthetic datasets for training. Due to GitHub file size limits, the datasets are hosted externally.

### 📥 Download Datasets

**Google Drive Link**: [ACIE Datasets](https://drive.google.com/drive/folders/19axWZDvMbTpHdN8KRrOIuzCYcYxn_9df?usp=drive_link)

### Available Datasets (~14GB total)

| File Name | Size | Description |
|-----------|------|-------------|
| `acie_observational_10k_x_10k.csv` | 758 MB | Observational data (10k samples) |
| `acie_observational_20k_x_20k.csv` | 3.0 GB | Observational data (20k samples) |
| `acie_counterfactual_10k_x_10k.csv` | 750 MB | Counterfactual pairs (10k) |
| `acie_hard_intervention_20k_x_20k.csv` | 3.0 GB | Hard intervention data (20k) |
| `acie_environment_shift_20k_x_20k.csv` | 3.0 GB | Environment distribution shift |
| `acie_instrument_shift_20k_x_20k.csv` | 3.5 GB | Instrument calibration shift |

### Dataset Setup

After downloading, place all CSV files in the `lib/` directory:

```bash
# Create lib directory if it doesn't exist
mkdir -p lib

# Move downloaded datasets
mv ~/Downloads/*.csv lib/

# Verify datasets
ls -lh lib/*.csv
```

### Data Format

Each CSV file contains:
- **Columns 0-1999** (10k) or **0-3999** (20k): Latent physical variables **P**
- **Columns 2000-7999** (10k) or **4000-14999** (20k): Observable variables **O**
- **Remaining columns**: Noise/bias variables **N**

### Generate Datasets (Alternative)

If you cannot download the datasets, regenerate them using provided scripts:

```bash
# Generate all datasets (requires ~14GB disk space and several hours)
python lib/scripts/ds_gen.py
python lib/scripts/ds_gen2.py
python lib/scripts/env_shift.py
python lib/scripts/instrument_shift.py
```

## 🚀 Quick Start

### 1. Training

#### Quick Training (10k dataset)

```bash
# Fast start with quickstart script
python scripts/train_quickstart.py
```

#### Full Training (20k dataset)

```bash
# Aggressive training configuration
python scripts/train_aggressive.py
```

#### Custom Training via CLI

```bash
python -m acie.cli train \
  --data-dir lib \
  --output-dir outputs/my_model \
  --dataset-size 10k \
  --max-epochs 50 \
  --batch-size 128 \
  --learning-rate 1e-4
```

#### Hyperparameter Tuning

```bash
# Run hyperparameter search
python scripts/hyperparam_tuning.py
```

### 2. Inference

#### Counterfactual Inference

```bash
python -m acie.cli infer \
  --checkpoint outputs/my_model/acie_final.ckpt \
  --observation-file my_observation.csv \
  --intervention "mass=1.5" \
  --output-dir results/
```

#### Programmatic Inference

```python
from acie.core.acie_core import ACIECore
from acie.inference.counterfactual import CounterfactualEngine
import torch

# Load trained model
model = ACIECore.load_from_checkpoint("outputs/my_model/acie_final.ckpt")
cf_engine = CounterfactualEngine(model)

# Prepare observation
observation = torch.randn(1, 6000)  # 10k dim

# Perform intervention
intervention = {"mass": 1.5}
counterfactual = cf_engine.generate_counterfactual(
    observation, 
    intervention
)

print(f"Counterfactual shape: {counterfactual.shape}")
```

### 3. Evaluation

```bash
python -m acie.cli evaluate \
  --checkpoint outputs/my_model/acie_final.ckpt \
  --data-dir lib \
  --dataset-size 10k
```

### 4. Java API Server

Start the production REST API:

```bash
cd java
mvn spring-boot:run
```

The server will start on `http://localhost:8080`

#### API Endpoints

**POST** `/api/inference/counterfactual`

```json
{
  "observation": [0.1, 0.2, ...],
  "intervention": {
    "mass": 1.5,
    "temperature": 5000
  },
  "modelPath": "outputs/my_model/acie_final.ckpt"
}
```

**Response**:
```json
{
  "counterfactual": [0.15, 0.25, ...],
  "latentState": [0.05, 0.08, ...],
  "timestamp": "2026-02-11T14:30:00Z"
}
```

### 5. R Analytics Dashboard

Launch the interactive Shiny dashboard:

```bash
Rscript r/shiny_dashboard.R
```

Access at `http://localhost:3838`

## 🔬 Multi-Language Components

### Python (`acie/`)

The core implementation with PyTorch:

- **`core/`**: SCM engine and ACIE core logic
- **`models/`**: Neural network architectures (VAE, physics layers)
- **`training/`**: Training pipeline with PyTorch Lightning
- **`inference/`**: Counterfactual and interventional inference
- **`data/`**: Data loading and preprocessing
- **`eval/`**: Evaluation metrics and validation

### Java (`java/`)

Spring Boot REST API server:

- **`ACIEInferenceServer.java`**: Main server application
- **`InferenceController.java`**: REST endpoints
- **`CounterfactualInferenceService.java`**: Business logic
- **`PythonModelBridge.java`**: Python integration via Jython/py4j

### Rust (`rust/`)

High-performance compute modules:

- **`tensor_ops.rs`**: Optimized tensor operations
- **`scm_graph.rs`**: SCM graph algorithms
- **`physics.rs`**: Physics simulation kernels
- **`data_loader.rs`**: Fast data loading

Build and use:
```bash
cd rust
cargo build --release
cargo test
```

### Assembly (`asm/`)

Critical matrix operations:

- **`matrix_kernels.asm`**: Hand-optimized matrix multiplication
- **`acie_asm_wrapper.c`**: C wrapper for Python FFI
- **`asm_python.py`**: Python bindings

Build:
```bash
cd asm
make
python asm_python.py  # Test
```

### R (`r/`)

Statistical analysis and visualization:

- **`acie_analysis.R`**: Statistical tests and model diagnostics
- **`shiny_dashboard.R`**: Interactive web dashboard

## ⚙️ Configuration

Configuration files in `config/`:

### `default_config.yaml`
Full training configuration with all hyperparameters:
```yaml
model:
  latent_dim: 2000
  observable_dim: 6000
  
training:
  max_epochs: 100
  batch_size: 128
  learning_rate: 1e-4
  
losses:
  reconstruction_weight: 1.0
  kl_weight: 0.1
  physics_constraint_weight: 5.0
  identifiability_weight: 2.0
```

### `dev_config.yaml`
Fast development mode (smaller models, fewer epochs):
```yaml
model:
  latent_dim: 500
  observable_dim: 1500
  
training:
  max_epochs: 10
  batch_size: 64
```

### `aggressive_config.yaml`
Maximum performance training:
```yaml
training:
  max_epochs: 200
  batch_size: 256
  precision: 16  # Mixed precision
  accumulate_grad_batches: 4
```

Specify config during training:
```bash
python -m acie.cli train --config config/aggressive_config.yaml
```

## 📁 Project Structure

```
ACIE/
├── acie/                      # Python core package
│   ├── core/                  # SCM and ACIE engine
│   │   ├── acie_core.py       # Main ACIE class
│   │   └── scm.py             # Structural causal model
│   ├── models/                # Neural architectures
│   │   ├── networks.py        # VAE, encoders, decoders
│   │   └── physics_layers.py # Physics constraint layers
│   ├── training/              # Training pipeline
│   │   ├── train.py           # Lightning trainer
│   │   └── losses.py          # Loss functions
│   ├── inference/             # Inference engines
│   │   ├── inference.py       # Latent inference
│   │   └── counterfactual.py  # Counterfactual generation
│   ├── data/                  # Data utilities
│   │   └── dataset.py         # PyTorch datasets
│   ├── eval/                  # Evaluation
│   │   └── metrics.py         # Evaluation metrics
│   ├── integration/           # Multi-language bridges
│   │   └── multi_language.py  # Rust/Java/ASM integration
│   └── cli.py                 # Command-line interface
├── java/                      # Java REST API server
│   ├── src/main/java/ai/acie/server/
│   │   ├── ACIEInferenceServer.java
│   │   ├── controller/        # REST controllers
│   │   ├── service/           # Business logic
│   │   ├── model/             # Data models
│   │   └── python/            # Python bridge
│   └── pom.xml                # Maven configuration
├── rust/                      # Rust performance modules
│   ├── src/
│   │   ├── lib.rs             # Library entry
│   │   ├── tensor_ops.rs      # Tensor operations
│   │   ├── scm_graph.rs       # Graph algorithms
│   │   ├── physics.rs         # Physics kernels
│   │   └── data_loader.rs     # Data loading
│   └── Cargo.toml             # Rust configuration
├── asm/                       # Assembly kernels
│   ├── matrix_kernels.asm     # Matrix operations
│   ├── acie_asm_wrapper.c     # C wrapper
│   ├── asm_python.py          # Python bindings
│   └── Makefile               # Build script
├── r/                         # R analytics
│   ├── acie_analysis.R        # Statistical analysis
│   └── shiny_dashboard.R      # Interactive dashboard
├── scripts/                   # Utility scripts
│   ├── train_quickstart.py    # Quick training
│   ├── train_aggressive.py    # Full training
│   ├── hyperparam_tuning.py   # Hyperparameter search
│   ├── demo_*.py              # Demo scripts
│   └── deploy.sh              # Deployment script
├── tests/                     # Test suites
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── config/                    # Configuration files
│   ├── default_config.yaml
│   ├── dev_config.yaml
│   └── aggressive_config.yaml
├── lib/                       # Data directory
│   ├── *.csv                  # Datasets (download separately)
│   └── scripts/               # Data generation scripts
├── requirements.txt           # Python dependencies
├── setup.py                   # Python package setup
├── Makefile                   # Master build script
└── README.md                  # This file
```

## 🧮 Mathematical Foundation

### Structural Causal Model (SCM)

ACIE models the causal relationships between latent physical states **P**, observables **O**, and noise **N**:

```
P = f_P(N_P)              # Latent physical variables
O = f_O(P, N_O)           # Observables from physics
```

### Counterfactual Inference

Three-step process:

1. **Abduction**: Infer latent state from observation
   ```
   q_θ(P|O) ≈ P(P|O)
   ```

2. **Action**: Apply intervention
   ```
   do(P_j = p*)
   ```

3. **Prediction**: Generate counterfactual
   ```
   P(O_{do(P)}|O) = ∫ P(O|P') q_θ(P'|O) dP'
   ```

### Physics Constraints

Enforce physical laws as differentiable constraints:

```
L_physics = ||C(P)||²
```

Where `C(P) = 0` encodes:
- Energy conservation
- Momentum conservation
- Thermodynamic stability
- Causality constraints

### Identifiability Optimization

Maximize mutual information between latents and counterfactuals:

```
L_ident = -I(P; O_{do(P)})
```

### Total Training Objective

```
L_total = L_recon + β·L_KL + λ_p·L_physics + λ_i·L_ident
```

Where:
- `L_recon`: Reconstruction loss (VAE)
- `L_KL`: KL divergence regularization
- `L_physics`: Physics constraint violation
- `L_ident`: Identifiability term

## 📚 Citation

If you use ACIE in your research, please cite:

```bibtex
@software{acie2026,
  title={ACIE: Astronomical Counterfactual Inference Engine},
  author={ACIE Development Team},
  year={2026},
  url={https://github.com/Jitterx69/Project-ACIE-v1},
  note={Multi-language causal inference system for astronomy}
}
```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact

For questions, issues, or collaborations:

- **GitHub Issues**: [Project-ACIE-v1/issues](https://github.com/Jitterx69/Project-ACIE-v1/issues)
- **Repository**: [github.com/Jitterx69/Project-ACIE-v1](https://github.com/Jitterx69/Project-ACIE-v1)

---

<div align="center">

**Built with ❤️ using Python, Java, Rust, Assembly, and R**

*Pushing the boundaries of causal inference in astronomy* 🌌

</div>
