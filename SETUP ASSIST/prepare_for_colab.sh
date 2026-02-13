#!/bin/bash

# Script to prepare a zip file for Google Colab
# Usage: ./SETUP\ ASSIST/prepare_for_colab.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Assuming script is in "SETUP ASSIST", project root is one level up
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# If script was in "scripts", it would also be one level up.
# Let's just robustly verify we are in the project root
if [ -d "$SCRIPT_DIR/../acie" ]; then
    PROJECT_ROOT="$SCRIPT_DIR/.."
elif [ -d "$SCRIPT_DIR/acie" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
fi

cd "$PROJECT_ROOT" || { echo "Error: Could not find project root"; exit 1; }

OUTPUT_DIR="SETUP ASSIST"
mkdir -p "$OUTPUT_DIR"
OUTPUT_ZIP="$OUTPUT_DIR/ACIE_Project.zip"

echo "Creating $OUTPUT_ZIP from $(pwd)..."

# Create zip with ONLY the ESSENTIAL files for training
# Excludes frontends, other languages, docs, tests, and local envs
zip -r "$OUTPUT_ZIP" \
    acie/ \
    config/ \
    setup.py \
    requirements.txt \
    .env.example \
    README.md

echo "Done! Upload '$OUTPUT_ZIP' to your Google Drive to train on Colab."
