#!/bin/bash
set -e

# Sync Script for ACIE
# Usage: ./sync_to_remote.sh <user>@<host>

if [ -z "$1" ]; then
    echo "Usage: ./sync_to_remote.sh <user>@<host>"
    exit 1
fi

REMOTE_HOST=$1
REMOTE_DIR="~/acie_remote"

echo "=== Syncing Code to $REMOTE_HOST:$REMOTE_DIR ==="

rsync -avz --exclude-from='.gitignore' \
    --exclude 'venv' \
    --exclude 'venv_new' \
    --exclude '__pycache__' \
    --exclude '.git' \
    . "$REMOTE_HOST:$REMOTE_DIR"

echo "=== Code Sync Complete ==="

# Optional: Sync Data
# If data is in a local directory, uncomment valid rsync command
# rsync -avz data/ "$REMOTE_HOST:$REMOTE_DIR/data/"

echo "Run './scripts/setup_remote.sh' on the remote machine to install."
