#!/usr/bin/env bash
# test_all.sh
# ==============================================================================
# All-In-One Test & Benchmark Suite for CXR-SIMD
#
# Supports:
#   1. PC (WSL2 / Linux x86_64) via ARM64 cross-compilation + QEMU emulation
#   2. Raspberry Pi 4/5 (native aarch64 Linux)
#
# Usage:
#   bash test_all.sh
# ==============================================================================

set -e

# Detect script root directory
if [ -d "/mnt/d/coa/cxr-simd" ]; then
    ROOT_DIR="/mnt/d/coa/cxr-simd"
else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

cd "$ROOT_DIR"

echo "======================================================================"
echo "          CXR-SIMD: All-In-One Validation & Benchmark Suite           "
echo "======================================================================"

ARCH="$(uname -m)"
echo "[INFO] Working Directory: $ROOT_DIR"
echo "[INFO] Host Architecture: $ARCH"

mkdir -p "$ROOT_DIR/bench"

# ------------------------------------------------------------------------------
# 1. Build & Run ARM64 NEON Kernels
# ------------------------------------------------------------------------------
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo ""
    echo "--- [1/3] Building Native ARM64 NEON Kernels ---"
    cd "$ROOT_DIR/kernels"
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc 2>/dev/null || echo 4)

    echo ""
    echo "--- [2/3] Running NEON Numerical Validation (Native) ---"
    ./validate

    echo ""
    echo "--- [3/3] Running NEON Micro-Benchmarks (Native) ---"
    ./bench | tee "$ROOT_DIR/bench/results_kernels.csv"
else
    echo ""
    echo "--- [1/4] Cross-Compiling ARM64 NEON Kernels with GCC aarch64 ---"
    cd "$ROOT_DIR/kernels"
    mkdir -p build_arm64 && cd build_arm64
    cmake .. -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
             -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
             -DCMAKE_SYSTEM_NAME=Linux \
             -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
             -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc 2>/dev/null || echo 4)

    echo ""
    echo "--- [2/4] Running NEON Numerical Validation (QEMU Emulation) ---"
    qemu-aarch64-static -L /usr/aarch64-linux-gnu ./validate

    echo ""
    echo "--- [3/4] Running NEON Micro-Benchmarks (QEMU Emulation) ---"
    qemu-aarch64-static -L /usr/aarch64-linux-gnu ./bench | tee "$ROOT_DIR/bench/results_arm64_qemu.csv"

    echo ""
    echo "--- [4/4] Validating x86 Scalar Fallback (Native) ---"
    cd "$ROOT_DIR/kernels"
    mkdir -p build_x86 && cd build_x86
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc 2>/dev/null || echo 4)
    ./validate
fi

# ------------------------------------------------------------------------------
# 2. Python Tensor Repacking & Disease Prediction Smoke Tests
# ------------------------------------------------------------------------------
echo ""
echo "--- Running Tensor Layout & Repacking Smoke Test (Python) ---"
cd "$ROOT_DIR"
if python3 -c "import torch" 2>/dev/null; then
    python3 kernels/repack.py
elif python -c "import torch" 2>/dev/null; then
    python kernels/repack.py
else
    echo "[INFO] PyTorch not installed in this Linux environment; skipping repack smoke test."
fi

echo ""
echo "--- Running End-to-End Disease Classification (Python) ---"
if python3 -c "import onnxruntime, PIL" 2>/dev/null; then
    python3 diagnose.py
elif python -c "import onnxruntime, PIL" 2>/dev/null; then
    python diagnose.py
else
    echo "[INFO] onnxruntime not in WSL environment (available in Windows host). Run 'python diagnose.py' on Windows."
fi

echo ""
echo "======================================================================"
echo "          SUCCESS: ALL KERNEL TESTS & BENCHMARKS PASSED!              "
echo "======================================================================"
