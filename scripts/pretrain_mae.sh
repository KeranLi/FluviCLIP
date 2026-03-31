#!/bin/bash
# MAE Pre-training on multi-source datasets
# Usage: ./scripts/pretrain_mae.sh

echo "======================================"
echo "MAE Pre-training"
echo "======================================"

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"

# Run MAE pre-training
python code/scripts/pretrain_mae.py

echo "Pre-training completed!"
echo "Checkpoints saved to output/MAE_Pretrain/"
