#!/bin/bash
# Compare long-tail handling methods
# Usage: ./scripts/compare_longtail_methods.sh

echo "======================================"
echo "Comparing Long-Tail Handling Methods"
echo "======================================"

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"

echo "Testing the following methods:"
echo "  1. Focal Loss + Focal MSE"
echo "  2. Inverse Frequency + Weighted MSE"
echo "  3. GHMC + Gradient Harmonized"
echo "  4. LDAM (adapted) + Margin-adjusted"
echo "  5. L1 Loss + MAE-based"
echo ""

# Run comparison
python code/scripts/Train_longtail_methods.py

echo "Long-tail methods comparison completed!"
echo "Results saved to output/longtail_comparison/"
