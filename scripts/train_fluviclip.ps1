# Train FluviCLIP model (main experiment)
# Usage: .\scripts\train_fluviclip.ps1

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Training FluviCLIP" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $scriptDir
Set-Location $projectDir

# Set environment variables
$env:PYTHONPATH = "$env:PYTHONPATH;$projectDir\code"

# Train FluviCLIP
python code\Train_FluviCLIP.py

Write-Host "Training completed!" -ForegroundColor Green
