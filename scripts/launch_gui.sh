#!/bin/bash
# Launch DeepSuspend GUI application
# Usage: ./scripts/launch_gui.sh

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"

echo "======================================"
echo "Launching DeepSuspend GUI"
echo "======================================"

python code/deepsuspend_gui.py
