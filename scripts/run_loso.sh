#!/bin/bash
# Leave-One-Station-Out (LOSO) Cross-Validation
# Usage: ./scripts/run_loso.sh

echo "======================================"
echo "Running LOSO Cross-Validation"
echo "======================================"

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"

# Run LOSO validation
python code/scripts/evaluate_loso.py

echo "LOSO validation completed!"
echo "Results saved to output/LOSO_results.csv"
