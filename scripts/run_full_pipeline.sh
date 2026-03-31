#!/bin/bash
# Full pipeline: Pre-training -> Training -> Evaluation -> Distillation
# Usage: ./scripts/run_full_pipeline.sh

echo "======================================"
echo "FluviCLIP Full Pipeline"
echo "======================================"
echo "This will run the complete workflow:"
echo "  1. MAE Pre-training (optional)"
echo "  2. FluviCLIP Training"
echo "  3. 5-Fold Cross-Validation"
echo "  4. LOSO Validation"
echo "  5. SHAP Analysis"
echo "  6. Knowledge Distillation"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

cd "$(dirname "$0")/.."

# Step 1: Pre-training (optional)
read -p "Run MAE pre-training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./scripts/pretrain_mae.sh
fi

# Step 2: Train FluviCLIP
echo ""
echo "Step 2: Training FluviCLIP..."
./scripts/train_fluviclip.sh

# Step 3: 5-Fold CV
echo ""
echo "Step 3: Running 5-Fold Cross-Validation..."
./scripts/run_5fold_cv.sh

# Step 4: LOSO
echo ""
echo "Step 4: Running LOSO Validation..."
./scripts/run_loso.sh

# Step 5: SHAP Analysis
echo ""
echo "Step 5: Running SHAP Analysis..."
./scripts/analyze_with_shap.sh

# Step 6: Distillation
echo ""
echo "Step 6: Knowledge Distillation..."
./scripts/distill_model.sh

echo ""
echo "======================================"
echo "Full Pipeline Completed!"
echo "======================================"
echo "Check output/ directory for all results"
