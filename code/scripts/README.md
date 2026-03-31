# Scripts Directory

This directory contains specialized scripts for analysis, evaluation, and advanced training.

## Analysis & Evaluation Scripts

- `analyze_shap.py` - SHAP (SHapley Additive exPlanations) analysis for model interpretability
- `evaluate_loso.py` - Leave-One-Station-Out (LOSO) cross-validation for spatial generalization
- `compare_all_models.py` - Comprehensive comparison of all baseline models

## Advanced Training Scripts

- `pretrain_mae.py` - Masked Autoencoder pre-training on multi-source datasets
- `train_distillation.py` - Knowledge distillation for lightweight model deployment
- `Train_longtail_methods.py` - Comparison of various long-tail handling methods

## Usage

Run these scripts from the project root directory:

```bash
cd F:\code\FluviCLIP
python code/scripts/analyze_shap.py
python code/scripts/evaluate_loso.py
python code/scripts/compare_all_models.py
```

## Directory Structure

```
code/
├── scripts/          # This directory - analysis & evaluation scripts
├── models/           # Model architectures
├── utils/            # Utility functions
├── configs/          # Configuration files
├── Train_*.py        # Main training scripts (in code/)
└── inference.py      # Inference script (in code/)
```

Note: These scripts automatically add the parent directory to Python path to import modules from `models/`, `utils/`, and `configs/`.
