#!/bin/bash
# Knowledge Distillation for lightweight deployment
# Usage: ./scripts/distill_model.sh

echo "======================================"
echo "Knowledge Distillation"
echo "======================================"
echo "Distilling FluviCLIP (287M) -> Student (15M)"
echo "Target: 19.1x parameter reduction"
echo ""

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"

# Run distillation
python code/scripts/train_distillation.py

echo "Distillation completed!"
echo "Student model saved to output/Distillation/"
