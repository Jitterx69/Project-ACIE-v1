# ACIE Enhanced CLI Examples

Quick reference for the new Typer-based CLI.

## Installation

```bash
# Install CLI dependencies
pip install typer[all] rich

# Or use the extras
pip install -e ".[cli]"
```

## Commands

### 1. Train Model

```bash
# Basic training
acie train --data-dir ./data --output-dir ./outputs

# Advanced training
acie train \
  --data-dir ./data \
  --output-dir ./outputs \
  --batch-size 256 \
  --max-epochs 200 \
  --gpus 2 \
  --learning-rate 0.0001
```

### 2. Inference

```bash
# Single inference
acie infer \
  --checkpoint outputs/model.ckpt \
  --observation-file observation.csv \
  --intervention "mass=1.5,metallicity=0.02"
```

### 3. Evaluation

```bash
# Evaluate on test data
acie evaluate \
  --checkpoint outputs/model.ckpt \
  --data-dir ./data \
  --output-dir ./eval_results
```

### 4. Benchmark ⚡ (NEW)

```bash
# Performance benchmark
acie benchmark \
  --checkpoint outputs/model.ckpt \
  --num-samples 1000 \
  --device mps
```

### 5. Serve API 🌐 (NEW)

```bash
# Start inference server
acie serve \
  --checkpoint outputs/model.ckpt \
  --port 8080 \
  --workers 4

# Access at http://localhost:8080
# API docs at http://localhost:8080/docs
```

### 6. Export Model 💾 (NEW)

```bash
# Export to ONNX
acie export \
  --checkpoint outputs/model.ckpt \
  --output-path model.onnx \
  --format onnx

# Export to TorchScript
acie export \
  --checkpoint outputs/model.ckpt \
  --output-path model.pt \
  --format torchscript
```

### 7. Version Info 📦 (NEW)

```bash
# Show system information
acie version
```

## Help

```bash
# Main help
acie --help

# Command-specific help
acie train --help
acie benchmark --help
```

## Features

✨ **Rich Console Output**
- Colorized output
- Progress bars
- Spinners
- Tables

⚡ **Better UX**
- Clear error messages
- Interactive prompts
- Auto-completion support

🎯 **New Commands**
- `benchmark` - Performance testing
- `serve` - Start API server
- `export` - Model export
- `version` - System info

## Migration from Old CLI

The old CLI is still available as `acie.cli_legacy`.

### Old Command → New Command

```bash
# Old
python -m acie.cli train --data-dir ./data --output-dir ./outputs

# New (shorter!)
acie train --data-dir ./data --output-dir ./outputs
```

All existing commands work the same, just with better output!
