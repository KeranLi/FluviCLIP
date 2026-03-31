# Baseline Training Scripts

This directory contains training scripts for all baseline models used in the comparison experiments.

## CNN Baselines

- `Train_ResNet50.py` - ResNet50 baseline
- `Train_Res2Net.py` - Res2Net with multi-scale features
- `Train_ResNeXt.py` - ResNeXt50 with grouped convolutions

## Transformer Baselines

- `Train_ViT.py` - Vision Transformer (ViT)
- `Train_SwinT.py` - Swin Transformer
- `Train_CoaT.py` - Convolutional Vision Transformer (CoaT)

## U-Net Baselines

- `Train_Unet.py` - 2D U-Net
- `Train_Unet2D_H_L.py` - 2D U-Net with High/Low split
- `Train_Unet3D_H_L.py` - 3D U-Net with High/Low split

## High/Low Split Variants

- `Train_ViT_H_L.py` - ViT with separate models for high/low SSC
- `Train_SwinT_H_L.py` - Swin-T with separate models for high/low SSC
- `Train_DeiT_H_L.py` - DeiT with separate models for high/low SSC

## Usage

Run from project root:

```bash
cd F:\code\FluviCLIP
python code/scripts/baselines/Train_ResNet50.py
python code/scripts/baselines/Train_SwinT.py
```

## Note

These scripts automatically adjust Python path to import from `code/` directory.
