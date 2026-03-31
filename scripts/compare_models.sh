#!/bin/bash
# Comprehensive model comparison (all models in one script)
# Usage: ./scripts/compare_models.sh

echo "======================================"
echo "Running Comprehensive Model Comparison"
echo "======================================"

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"

# Run comparison
python code/scripts/compare_all_models.py

echo "Comparison completed!"
echo "Results saved to output/comparison/"
