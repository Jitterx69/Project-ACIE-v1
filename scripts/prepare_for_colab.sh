#!/bin/bash

# Script to prepare a zip file for Google Colab
# Usage: ./scripts/prepare_for_colab.sh

OUTPUT_ZIP="ACIE_Project.zip"

echo "Creating $OUTPUT_ZIP for Google Colab..."

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
