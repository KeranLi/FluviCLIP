#!/bin/bash
# 5-Fold Cross-Validation with 6:2:2 split
# Usage: ./scripts/run_5fold_cv.sh

echo "======================================"
echo "Running 5-Fold Cross-Validation"
echo "======================================"

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"

# Run 5-fold CV
python code/scripts/cross_validation_5fold.py

echo "5-Fold CV completed!"
echo "Results saved to output/5fold_cv/"
