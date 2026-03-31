#!/bin/bash
# Train FluviCLIP model (main experiment)
# Usage: ./scripts/train_fluviclip.sh

echo "======================================"
echo "Training FluviCLIP"
echo "======================================"

cd "$(dirname "$0")/.."

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"

# Train FluviCLIP
python code/Train_FluviCLIP.py

echo "Training completed!"
