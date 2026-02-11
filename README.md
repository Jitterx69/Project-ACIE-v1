# ACIE - Astronomical Counterfactual Inference Engine

A physics-constrained, causal inference system for astronomical observations that estimates interventional and counterfactual distributions over astrophysical data.

## Overview

ACIE is a deep learning system that performs **counterfactual inference** on astronomical observations. Unlike traditional ML systems that predict correlations, ACIE answers causal questions like:

> "What would the observable properties of this galaxy be if its initial mass were 1.5× higher?"

### Key Features

- **Causal Reasoning**: Explicit structural causal models with intervention operators
- **Physics-Constrained**: Enforces conservation laws and stability constraints
- **Counterfactual Inference**: 3-step abduction-action-prediction pipeline
- **Partial Observability**: Infers latent physical states from incomplete observations
- **Identifiability Optimization**: Maximizes causal identifiability under constraints

## Installation

```bash
# Clone repository
cd ACIE

# Install dependencies
pip install -r requirements.txt

# Install ACIE package
pip install -e .
```

## Quick Start

### Training

```bash
# Quick start training on 10k dataset
python scripts/train_quickstart.py

# Or use CLI
python -m acie.cli train \
  --data-dir lib \
  --output-dir outputs/my_model \
  --dataset-size 10k \
  --max-epochs 50 \
  --batch-size 128
```

### Inference

```bash
# Perform counterfactual inference
python -m acie.cli infer \
  --checkpoint outputs/my_model/acie_final.ckpt \
  --observation-file my_observation.csv \
  --intervention "mass=1.5" \
  --output-dir results/
```

### Evaluation

```bash
# Evaluate model performance
python -m acie.cli evaluate \
  --checkpoint outputs/my_model/acie_final.ckpt \
  --data-dir lib \
  --dataset-size 10k
```

## Architecture

ACIE consists of several key components:

1. **Structural Causal Model (SCM)**: DAG representing causal relationships
2. **Inference Engine**: VAE-based latent state inference P(P|O)
3. **Counterfactual Engine**: Twin networks for counterfactual generation
4. **Physics Layers**: Differentiable constraints for physical laws
5. **Training Pipeline**: Multi-objective optimization with PyTorch Lightning

## Data Format

ACIE expects CSV files with the following structure:
- Columns 0-1999 (or 0-3999): Latent physical variables P
- Columns 2000-7999 (or 4000-14999): Observable variables O  
- Remaining columns: Noise/bias variables N

Datasets available in `lib/`:
- `acie_observational_10k_x_10k.csv`: Observational data
- `acie_counterfactual_10k_x_10k.csv`: Counterfactual pairs
- `acie_hard_intervention_20k_x_20k.csv`: Intervention data
- `acie_environment_shift_20k_x_20k.csv`: Environment shift
- `acie_instrument_shift_20k_x_20k.csv`: Instrument shift

## Configuration

Configuration files in `config/`:
- `default_config.yaml`: Full training configuration
- `dev_config.yaml`: Fast development mode

Modify hyperparameters, loss weights, and physics constraints in these files.

## Project Structure

```
ACIE/
├── acie/
│   ├── core/              # SCM and ACIE engine
│   ├── inference/         # Latent inference and counterfactuals
│   ├── models/            # Neural network architectures
│   ├── training/          # Training pipeline and losses
│   ├── data/              # Data loading
│   ├── eval/              # Evaluation metrics
│   └── cli.py             # Command-line interface
├── config/                # Configuration files
├── scripts/               # Example scripts
├── lib/                   # Data directory
└── tests/                 # Unit and integration tests
```

## Mathematical Foundation

ACIE implements counterfactual inference via:

**Latent Inference**: q_θ(P|O) - Variational posterior over latent states

**Intervention**: do(P_j = p*) - Hard intervention on variables

**Counterfactual**: P(O_do(P)|O) - Counterfactual distribution

**Physics**: C(P) = 0 - Constraint enforcement

**Identifiability**: max I(P; O_do(P)) - Mutual information optimization

## Citation

If you use ACIE in your research, please cite:

```bibtex
@software{acie2026,
  title={ACIE: Astronomical Counterfactual Inference Engine},
  author={ACIE Team},
  year={2026},
  url={https://github.com/yourusername/ACIE}
}
```

## License

MIT License

## Contact

For questions and issues, please open an issue on GitHub.
