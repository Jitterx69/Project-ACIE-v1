#!/bin/bash

# Script to prepare a zip file for Google Colab
# Usage: ./scripts/prepare_for_colab.sh

OUTPUT_ZIP="ACIE_Project.zip"

echo "Creating $OUTPUT_ZIP for Google Colab..."

# Create zip excluding virtual environments, git, and caches
zip -r "$OUTPUT_ZIP" . \
    -x "*.git*" \
    -x "*venv*" \
    -x "*__pycache__*" \
    -x "*.DS_Store" \
    -x "*outputs/*" \
    -x "*logs/*" \
    -x "*idea*" \
    -x "*vscode*"

echo "Done! Upload '$OUTPUT_ZIP' to your Google Drive to train on Colab."
