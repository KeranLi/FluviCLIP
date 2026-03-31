# FluviCLIP Experiment Scripts

This directory contains convenient scripts for running experiments.

## Available Scripts

### Main Training

| Script | Description |
|--------|-------------|
| `train_fluviclip.sh` / `.ps1` | Train FluviCLIP (main model) |
| `pretrain_mae.sh` | MAE pre-training on multi-source datasets |
| `distill_model.sh` | Knowledge distillation for lightweight deployment |

### Evaluation & Analysis

| Script | Description |
|--------|-------------|
| `run_5fold_cv.sh` | 5-Fold Cross-Validation (6:2:2 split) |
| `run_loso.sh` | Leave-One-Station-Out validation |
| `compare_longtail_methods.sh` | Compare long-tail handling methods |
| `compare_models.sh` | Comprehensive model comparison |
| `analyze_with_shap.sh` | SHAP interpretability analysis |

### Utilities

| Script | Description |
|--------|-------------|
| `run_inference.sh` | Run inference on test images |
| `launch_gui.sh` | Launch DeepSuspend GUI |
| `run_full_pipeline.sh` | Run complete pipeline |

## Usage

### Linux/macOS (Bash)

```bash
cd F:\code\FluviCLIP

# Make scripts executable
chmod +x scripts/*.sh

# Train FluviCLIP
./scripts/train_fluviclip.sh

# Run 5-fold CV
./scripts/run_5fold_cv.sh

# Compare long-tail methods
./scripts/compare_longtail_methods.sh

# Full pipeline
./scripts/run_full_pipeline.sh
```

### Windows (PowerShell)

```powershell
cd F:\code\FluviCLIP

# Train FluviCLIP
.\scripts\train_fluviclip.ps1

# Run all experiments
.\scripts\run_all_experiments.ps1
```

## Long-Tail Handling Methods

The `compare_longtail_methods.sh` script tests the following methods:

1. **Focal Loss + Focal MSE**: Down-weights easy examples, focuses on hard ones
2. **Inverse Frequency + Weighted MSE**: Weights samples by inverse frequency
3. **GHMC + Gradient Harmonized**: Balances gradient contributions
4. **LDAM (adapted) + Margin-adjusted**: Label-distribution-aware margins
5. **L1 Loss + MAE-based**: L1 loss for robust regression

## 5-Fold Cross-Validation

The `run_5fold_cv.sh` script implements:
- Stratified sampling to maintain Head/Tail distribution
- 6:2:2 split (60% train, 20% val, 20% test) per fold
- 5 folds with different random seeds
- Average and standard deviation across folds

## Output Structure

```
output/
├── 5fold_cv/              # 5-fold CV results
├── longtail_comparison/   # Long-tail methods comparison
├── LOSO_results.csv       # LOSO validation results
├── comparison/            # Model comparison results
├── shap/                  # SHAP visualizations
└── baselines/            # Baseline training logs
```
