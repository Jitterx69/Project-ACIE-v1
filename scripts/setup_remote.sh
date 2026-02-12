#!/bin/bash
set -e

# Remote Setup Script for ACIE Training
# Usage: ./setup_remote.sh

echo "=== Starting ACIE Remote Setup ==="

# 1. Check for Python 3.9+
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: $PYTHON_VERSION"

# 2. Check for CUDA (nvidia-smi)
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: nvidia-smi not found. GPU training might not work."
fi

# 3. Create Virtual Environment
VENV_DIR="venv_remote"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment $VENV_DIR exists. Skipping creation."
else
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# 4. Install Dependencies
echo "Installing dependencies..."
pip install --upgrade pip

# Install critical dependencies with specific versions (matching local success)
# Use robust installation method
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning torchmetrics
pip install "numpy<2.0" padding pandas scipy networkx
pip install "bcrypt<4.0.0" passlib python-jose[cryptography]  # Security deps

# Install project in editable mode
pip install -e .

echo "=== Setup Complete ==="
echo "Activate environment with: source $VENV_DIR/bin/activate"
echo "Run training with: python acie/training/train.py --data_dir /path/to/data"
