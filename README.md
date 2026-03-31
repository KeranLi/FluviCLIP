# FluviCLIP

**FluviCLIP**: Capture Head and Long-Tail Extreme Suspended Sediment Concentration Variations from Remote Sensing by Multimodal Contrastive Learning

This repository contains the implementation of FluviCLIP, a multimodal contrastive learning framework for Suspended Sediment Concentration (SSC) estimation from remote sensing imagery.

## Project Structure

```
FluviCLIP/
├── code/                       # Main source code
│   ├── models/                 # Model architectures
│   │   ├── fluviformer.py     # FluviFormer with FSS and Fluvial Attention
│   │   ├── fluviclip.py       # FluviCLIP multimodal architecture
│   │   ├── resnet.py          # ResNet, Res2Net, ResNeXt baselines
│   │   ├── sequence_models.py # LSTM, GRU baselines
│   │   ├── ml_baselines.py    # SVM, XGBoost, LightGBM baselines
│   │   └── ...
│   ├── utils/                  # Utility functions
│   ├── configs/                # Configuration files
│   ├── scripts/                # Python scripts for training/evaluation
│   │   ├── pretrain_mae.py
│   │   ├── cross_validation_5fold.py  # 5-Fold CV with 6:2:2 split
│   │   ├── evaluate_loso.py
│   │   ├── analyze_shap.py
│   │   ├── compare_all_models.py
│   │   ├── Train_longtail_methods.py
│   │   └── baselines/         # Individual baseline training scripts
│   ├── Train_FluviCLIP.py
│   ├── inference.py
│   └── deepsuspend_gui.py
├── scripts/                    # Shell/PowerShell scripts for experiments
├── datasets/                   # Dataset directory
└── README.md
```

## Quick Start (Using Shell Scripts)

### Linux/macOS (Bash)

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Train FluviCLIP
./scripts/train_fluviclip.sh

# Run 5-Fold Cross-Validation (6:2:2 split, 5-fold average)
./scripts/run_5fold_cv.sh

# LOSO Validation
./scripts/run_loso.sh

# Compare Long-Tail Methods
./scripts/compare_longtail_methods.sh

# Full Pipeline
./scripts/run_full_pipeline.sh
```

### Windows (PowerShell)

```powershell
# Train FluviCLIP
.\scripts\train_fluviclip.ps1

# Run all experiments
.\scripts\run_all_experiments.ps1
```

## Detailed Usage

### 1. Environment Setup

```bash
pip install -r code/environments/requirements.txt
```

### 2. MAE Pre-training (Optional)

```bash
./scripts/pretrain_mae.sh
```

### 3. Train FluviCLIP

```bash
# Using shell script
./scripts/train_fluviclip.sh

# Or directly
python code/Train_FluviCLIP.py
```

### 4. 5-Fold Cross-Validation (6:2:2 Split)

Implements 5-fold stratified cross-validation with 60% train / 20% validation / 20% test split:

```bash
./scripts/run_5fold_cv.sh
```

This produces:
- 5-fold average metrics (Mean ± Std)
- Head/Tail/Overall R² and MAE
- Results saved to `output/5fold_cv/`

### 5. Long-Tail Handling Methods Comparison

Compares 5 different long-tail handling approaches:

```bash
./scripts/compare_longtail_methods.sh
```

Methods compared:
1. **Focal Loss + Focal MSE**: Focuses on hard examples
2. **Inverse Frequency + Weighted MSE**: Inverse frequency weighting
3. **GHMC + Gradient Harmonized**: Balances gradient contributions
4. **LDAM (adapted) + Margin-adjusted**: Label-distribution-aware margins
5. **L1 Loss + MAE-based**: Robust L1 regression

Results saved to `output/longtail_comparison/results.csv`

### 6. LOSO Validation

Leave-One-Station-Out cross-validation for spatial generalization:

```bash
./scripts/run_loso.sh
```

### 7. Model Comparison

Comprehensive comparison of all baselines:

```bash
./scripts/compare_models.sh
```

### 8. SHAP Analysis

```bash
./scripts/analyze_with_shap.sh
```

### 9. Knowledge Distillation

```bash
./scripts/distill_model.sh
```

### 10. Inference

```bash
./scripts/run_inference.sh path/to/images
```

### 11. Launch GUI

```bash
./scripts/launch_gui.sh
```

## Key Features

- **FluviFormer**: Vision Transformer with fluvial morphological priors
- **Multimodal Contrastive Learning**: Integrates RemoteCLIP text encoder
- **Gated Dual-Branch Head**: Decoupled optimization for head/tail distributions
- **Comprehensive Baselines**: ResNet, Res2Net, ResNeXt, ViT, Swin-T, U-Net, LSTM, GRU, SVM, XGBoost, LightGBM
- **Foundation Model Comparisons**: RemoteCLIP, HyperFree, SkySense, HyperSigma, CMID, SpectralGPT
- **Long-Tail Methods**: Focal Loss, Inverse Frequency, GHMC, LDAM, L1 Loss
- **5-Fold CV**: Stratified 6:2:2 split with mean±std reporting
- **Uncertainty Quantification**: Monte Carlo Dropout
- **Interpretability**: SHAP analysis

## Results

Expected performance on Test Set (5-fold CV average):

| Model | Head R² (%) | Tail R² (%) | Overall R² (%) |
|-------|-------------|-------------|----------------|
| FluviCLIP (Ours) | 93.7 ± 1.2 | 85.7 ± 2.1 | 91.2 ± 1.5 |
| Swin-T + RemoteCLIP | 70.2 ± 3.5 | 41.8 ± 5.2 | 62.5 ± 3.8 |
| ResNet50 + CLIP | 60.5 ± 4.1 | 25.3 ± 6.2 | 48.7 ± 4.5 |

## Citation

If you use this code, please cite:

```bibtex
@article{fluviclip2024,
  title={FluviCLIP: Capture Head and Long-Tail Extreme Suspended Sediment Concentration Variations from Remote Sensing by Multimodal Contrastive Learning},
  author={Li, Keran and Chu, Feiyue and Cai, Jiarui and Li, Anzhou and Zheng, Dongyu and Si, Xu and Koeshidayatullah, Ardiansyah and Hu, Linshu and Hu, Xiumian},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024}
}
```

## License

This code is provided for research purposes.
