#!/bin/bash
# Train all baseline models for comparison
# Usage: ./scripts/run_all_baselines.sh

echo "======================================"
echo "Training All Baseline Models"
echo "======================================"

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"

# Create output directory
mkdir -p output/baselines

# CNN Baselines
echo "Training ResNet50..."
python code/scripts/baselines/Train_ResNet50.py > output/baselines/resnet50.log 2>&1

echo "Training Res2Net..."
python code/scripts/baselines/Train_Res2Net.py > output/baselines/res2net.log 2>&1

echo "Training ResNeXt50..."
python code/scripts/baselines/Train_ResNeXt.py > output/baselines/resnext50.log 2>&1

# Transformer Baselines
echo "Training ViT..."
python code/scripts/baselines/Train_ViT.py > output/baselines/vit.log 2>&1

echo "Training Swin-T..."
python code/scripts/baselines/Train_SwinT.py > output/baselines/swint.log 2>&1

echo "Training CoaT..."
python code/scripts/baselines/Train_CoaT.py > output/baselines/coat.log 2>&1

# U-Net Baselines
echo "Training U-Net 2D..."
python code/scripts/baselines/Train_Unet.py > output/baselines/unet2d.log 2>&1

echo "All baseline training completed!"
echo "Logs saved to output/baselines/"
