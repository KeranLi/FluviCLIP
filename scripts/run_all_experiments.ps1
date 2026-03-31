# Run all experiments (PowerShell version)
# Usage: .\scripts\run_all_experiments.ps1

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $scriptDir
Set-Location $projectDir

$env:PYTHONPATH = "$env:PYTHONPATH;$projectDir\code"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "FluviCLIP - All Experiments" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Create output directory
New-Item -ItemType Directory -Force -Path "output" | Out-Null

# 1. Train FluviCLIP
Write-Host "`n[1/6] Training FluviCLIP..." -ForegroundColor Yellow
python code\Train_FluviCLIP.py

# 2. 5-Fold CV
Write-Host "`n[2/6] Running 5-Fold Cross-Validation..." -ForegroundColor Yellow
python code\scripts\cross_validation_5fold.py

# 3. LOSO
Write-Host "`n[3/6] Running LOSO Validation..." -ForegroundColor Yellow
python code\scripts\evaluate_loso.py

# 4. Long-tail methods comparison
Write-Host "`n[4/6] Comparing Long-Tail Methods..." -ForegroundColor Yellow
python code\scripts\Train_longtail_methods.py

# 5. Model comparison
Write-Host "`n[5/6] Running Model Comparison..." -ForegroundColor Yellow
python code\scripts\compare_all_models.py

# 6. SHAP Analysis
Write-Host "`n[6/6] Running SHAP Analysis..." -ForegroundColor Yellow
python code\scripts\analyze_shap.py

Write-Host "`n======================================" -ForegroundColor Green
Write-Host "All Experiments Completed!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host "Check output\ directory for results" -ForegroundColor White
