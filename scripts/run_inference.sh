#!/bin/bash
# Run inference on test images
# Usage: ./scripts/run_inference.sh [image_path]

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"

# Default input directory
INPUT_DIR="${1:-datasets/inference}"
OUTPUT_DIR="output/inference"

mkdir -p "$OUTPUT_DIR"

echo "======================================"
echo "Running Inference"
echo "======================================"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Run inference with uncertainty and Grad-CAM
python code/inference.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --uncertainty \
    --gradcam

echo "Inference completed!"
echo "Results saved to $OUTPUT_DIR"
