# Phase 4 — SIMD-Aware Optimization Deliverables

## Overview

Phase 4 implements the custom ARM NEON INT8 kernel layer for the CXR-SIMD pipeline.
The deliverables are:

- **Hand-written, numerically-validated NEON kernels** (C++17, <arm_neon.h>)
- **Scalar reference twins** for every kernel (for CI validation on any architecture)
- **Tensor repacking utilities** (Python, NCHW→NHWC blocked layout)
- **Python ctypes bridge** to call compiled kernels from Python
- **CMake build system** (produces alidate, ench, libneon_kernels.so)
- **Pi deployment scripts** (setup + inference runner)

---

## Directory Structure Created

`
cxr-simd/
├── kernels/
│   ├── neon_kernels.h              ← public API (4 kernels)
│   ├── neon_kernels.cpp            ← NEON implementations + scalar fallback
│   ├── repack.py                   ← NCHW→NHWC, weight packing, quantize utils
│   ├── neon_bridge.py              ← Python ctypes wrapper for libneon_kernels.so
│   ├── bench_kernels.py            ← Python micro-benchmark (cross-platform)
│   ├── CMakeLists.txt              ← builds validate + bench + libneon_kernels.so
│   ├── scalar_ref/
│   │   ├── scalar_kernels.h
│   │   └── scalar_kernels.cpp      ← pure-C reference (no SIMD)
│   └── tests/
│       ├── validate_kernels.cpp    ← exact int32 equality checks
│       └── bench_kernels.cpp       ← C++ micro-benchmark with CSV output
├── deploy/
│   ├── run_inference.py            ← Pi inference runner (onnxruntime + neon)
│   └── pi_setup.sh                 ← Pi bootstrap script
├── bench/
│   └── results_kernels.csv         ← (populated after Pi run)
├── results/                        ← (plots go here in Phase 6)
└── report/                         ← (writeup goes here)
`

---

## Kernels Implemented

| # | Kernel | NEON Strategy | Scalar Reference |
|---|--------|---------------|-----------------|
| 1 | simd_dot_int8 | 32-element unrolled loop, mull_s8 + mull_high_s8 + padalq_s16, horizontal reduction with padd_s32 | scalar_dot_int8 |
| 2 | simd_depthwise_conv3x3_int8 | 16-wide output columns per iteration, movl_s8 + mlal_n_s16 per kernel weight, st1q_s32 | scalar_depthwise_conv3x3_int8 |
| 3 | simd_pointwise_gemm_int8 | K-loop vectorised with mull_s8 + padalq_s16, outer M×N loop | scalar_pointwise_gemm_int8 |
| 4 | simd_requantize | 16-element blocks, cvtq_f32_s32 + mulq_f32 + rndnq_f32 + saturating narrow qmovn_s32 → qmovn_s16 | scalar_requantize |

---

## Tensor Layout Decision: NHWC with 16-channel blocking

**Why NHWC (not NCHW)?**
- ARM NEON's ld1q_s8 loads 16 consecutive int8 values from memory.
- In **NCHW**, all H×W values for one channel are contiguous — perfect for GPU coalescing but requires N_ch/16 separate loads per pixel for NEON.
- In **NHWC**, all C channels for one pixel are contiguous — a single ld1q_s8 loads 16 channels at once, saturating the 128-bit NEON pipeline.
- Channels are padded to multiples of 16 (pad_channels_to_multiple(x, 16)).

**Weight packing:**
- Depthwise: [C, 1, KH, KW] → [C, KH*KW] — one flat kernel vector per channel.
- Pointwise: [OC, IC, 1, 1] → [IC, OC] — transposed to match GEMM layout A[M,K] @ B[K,N].

---

## Smoke Tests (run on x86, scalar fallback path)

`
=== repack.py smoke test ===
NCHW (1, 32, 7, 7)  ->  NHWC (1, 7, 7, 32)  (C-contig=True)
Round-trip max abs diff: 0.00e+00  (expect 0.00e+00)
Quantization max error:  0.0136  (expect < scale/2 = 0.0136)
Depthwise weight pack:   torch.Size([32, 1, 3, 3]) -> (32, 9)
Pointwise weight pack:   torch.Size([64, 32, 1, 1]) -> (32, 64)  [IC, OC]
Channel pad:             C=30 -> C=32 (multiple of 16)
All smoke tests OK.
`

---

## x86 Scalar Baseline (results_kernels.csv — this machine)

| Kernel | Param | Scalar Latency (µs) |
|--------|-------|---------------------|
| dot_int8 | N=256 | 1.6 |
| dot_int8 | N=4096 | 5.4 |
| depthwise_conv3x3 | 34×34 | 2988 |
| depthwise_conv3x3 | 18×18 | 746 |
| pointwise_gemm | 49×16×96 | 35.5 |
| pointwise_gemm | 196×32×96 | 237 |

*NEON speedup numbers will be added after Pi run (expected 4–12× for dot, 3–8× for conv).*

---

## Pi Run Instructions

### Step 1 — Copy to Pi
`ash
# From dev machine:
rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
    d:/coa/cxr-simd/ pi@raspberrypi:~/cxr-simd/
`

### Step 2 — Build & Validate kernels
`ash
# On Pi (aarch64, Raspberry Pi OS 64-bit):
cd ~/cxr-simd
bash deploy/pi_setup.sh
`

Expected output:
`
=== Kernel Numerical Validation ===
[PASS] dot_int8  N=256  ref=XXXX simd=XXXX
[PASS] dot_int8  N=48   ref=XXXX simd=XXXX
[PASS] depthwise_conv3x3_int8 stride=1 18x18
[PASS] depthwise_conv3x3_int8 stride=2 9x9
[PASS] pointwise_gemm_int8 16x32x16
[PASS] requantize N=128
===================================
ALL TESTS PASSED
`

### Step 3 — Kernel micro-benchmark
`ash
./kernels/build/bench
# CSV output: kernel,variant,param_n,latency_us_mean,speedup_vs_scalar
`

### Step 4 — Full model inference benchmark
`ash
# ONNXRuntime backend (reference):
python deploy/run_inference.py \
    --model quantize/onnx/int8_ptq.onnx \
    --backend onnxruntime --runs 100

# Compare both backends:
python deploy/run_inference.py \
    --model quantize/onnx/fp32.onnx \
    --compare --runs 50
`

### Step 5 — perf stat (cache + SIMD utilisation)
`ash
perf stat -e cache-misses,cache-references,instructions,cycles \
    ./kernels/build/bench 2>&1 | tee bench/perf_kernels.txt
`

---

## Phase 4b Decision: Hand-written vs. ARM Compute Library

**Decision: Hybrid approach**

| Layer type | Implementation |
|------------|---------------|
| simd_dot_int8 | ✅ Hand-written NEON |
| simd_depthwise_conv3x3_int8 | ✅ Hand-written NEON |
| simd_pointwise_gemm_int8 | ✅ Hand-written NEON |
| Full-model INT8 inference (Phase 5) | ONNXRuntime (ARM NEON EP) |

**Rationale:** Writing and validating one dot product + one depthwise conv + one GEMM kernel from scratch demonstrates mastery of NEON intrinsics and provides publishable speedup numbers. For full-model end-to-end inference, ONNXRuntime's execution provider is the production-grade baseline — this is the *comparison target*, not a cop-out. The novel contribution is the repacking methodology + kernel-level speedup measurements vs. the scalar baseline.

---

## Deliverables Checklist

- [x] kernels/neon_kernels.h — public API, 4 kernels
- [x] kernels/neon_kernels.cpp — NEON implementations
- [x] kernels/scalar_ref/scalar_kernels.{h,cpp} — scalar twins
- [x] kernels/tests/validate_kernels.cpp — exact equality validation
- [x] kernels/tests/bench_kernels.cpp — C++ micro-benchmark
- [x] kernels/CMakeLists.txt — builds validate + bench + .so
- [x] kernels/repack.py — NCHW→NHWC, weight packing, quantize
- [x] kernels/neon_bridge.py — Python ctypes API
- [x] kernels/bench_kernels.py — cross-platform Python benchmark
- [x] deploy/run_inference.py — Pi inference runner (ort + neon)
- [x] deploy/pi_setup.sh — one-shot Pi bootstrap
- [x] ench/results_kernels.csv — x86 scalar baseline numbers
- [x] PC QEMU validation run → ALL TESTS PASSED (ARM64 NEON verified via qemu-aarch64-static)
- [x] PC QEMU benchmark CSV with NEON speedup numbers (bench/results_arm64_qemu.csv)
- [ ] Physical Pi benchmark validation (run on hardware if/when connected)
