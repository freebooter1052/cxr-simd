# test_all.ps1
# Runs the CXR-SIMD all-in-one test suite:
#  1. NEON Kernels + QEMU ARM64 Emulation (inside WSL2)
#  2. Full Disease Prediction & Clinical Classification across all 4 model tiers (FP32, INT8-PTQ, INT8-QAT, INT4)
$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "       CXR-SIMD: RUNNING ALL-IN-ONE PIPELINE & DISEASE PREDICTION       " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Run Kernel validation & benchmark inside WSL2
Write-Host "`n[STEP 1/2] Running ARM64 NEON Kernels & Benchmarks under QEMU..." -ForegroundColor Yellow
wsl -d Ubuntu -- bash -c "cd /mnt/d/coa/cxr-simd && tr -d '\r' < test_all.sh > /tmp/test_all.sh && chmod +x /tmp/test_all.sh && /tmp/test_all.sh"

# 2. Run End-to-End Disease Classification in Python
Write-Host "`n[STEP 2/2] Running Multi-Disease Clinical Diagnosis across Model Variants..." -ForegroundColor Yellow
python diagnose.py

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "                ALL TESTS & DIAGNOSIS COMPLETE!                       " -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
