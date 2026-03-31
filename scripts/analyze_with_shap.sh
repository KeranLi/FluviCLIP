#!/bin/bash
# SHAP interpretability analysis
# Usage: ./scripts/analyze_with_shap.sh

echo "======================================"
echo "SHAP Interpretability Analysis"
echo "======================================"

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"

# Run SHAP analysis
python code/scripts/analyze_shap.py

echo "SHAP analysis completed!"
echo "Visualizations saved to output/shap/"
