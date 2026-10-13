# -*- coding: utf-8 -*-
"""
kernels/bench_kernels.py
-------------------------
Python-side micro-benchmark harness.

Runs on x86 (using numpy scalar path) as a dry-run, and on Pi with the
compiled .so for the real NEON numbers.

Usage:
    python kernels/bench_kernels.py              # numpy scalar baseline
    python kernels/bench_kernels.py --neon       # with libneon_kernels.so

Output: bench/results_kernels.csv (one row per kernel/size/variant).
"""

import argparse
import csv
import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_CSV = ROOT / "bench" / "results_kernels.csv"
RUNS = 500


# ---------------------------------------------------------------------------
# Numpy scalar implementations (always available, cross-platform)
# ---------------------------------------------------------------------------

def np_dot_int8(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.dot(a.astype(np.int32), b.astype(np.int32)))


def np_depthwise_conv3x3(inp: np.ndarray, kernel: np.ndarray,
                          stride: int = 1) -> np.ndarray:
    H, W = inp.shape
    OH = (H - 2 + stride - 1) // stride
    OW = (W - 2 + stride - 1) // stride
    out = np.zeros((OH, OW), dtype=np.int32)
    k = kernel.reshape(3, 3).astype(np.int32)
    for oh in range(OH):
        for ow in range(OW):
            ih, iw = oh * stride, ow * stride
            patch = inp[ih:ih+3, iw:iw+3].astype(np.int32)
            out[oh, ow] = np.sum(patch * k)
    return out


def np_pointwise_gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A.astype(np.int32) @ B.astype(np.int32))


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def time_fn(fn, runs=RUNS):
    # warm-up
    for _ in range(max(1, runs // 20)):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    elapsed = time.perf_counter() - t0
    return elapsed / runs * 1e6  # microseconds per call


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

def run_benchmarks(neon_kernels=None):
    rng = np.random.default_rng(42)
    rows = []

    def row(kernel, variant, param, lat_us):
        rows.append({
            "kernel": kernel,
            "variant": variant,
            "param": param,
            "latency_us": f"{lat_us:.3f}",
        })
        return lat_us

    # ---- 1. dot product ----
    for N in [256, 4096]:
        a = rng.integers(-127, 127, N, dtype=np.int8)
        b = rng.integers(-127, 127, N, dtype=np.int8)

        scalar_us = row("dot_int8", "numpy_scalar", f"N={N}",
                         time_fn(lambda: np_dot_int8(a, b)))

        if neon_kernels is not None:
            neon_us = row("dot_int8", "neon", f"N={N}",
                           time_fn(lambda: neon_kernels.dot(a, b)))
            speedup = scalar_us / neon_us
            print(f"dot N={N:5d}:  scalar={scalar_us:.1f}us  neon={neon_us:.1f}us  speedup={speedup:.2f}x")
        else:
            print(f"dot N={N:5d}:  scalar={scalar_us:.1f}us  (neon not available)")

    # ---- 2. depthwise conv 3x3 ----
    for H, W in [(34, 34), (18, 18)]:
        inp = rng.integers(-127, 127, (H, W), dtype=np.int8)
        kern = rng.integers(-127, 127, 9, dtype=np.int8)

        scalar_us = row("depthwise_conv3x3", "numpy_scalar", f"{H}x{W}",
                         time_fn(lambda: np_depthwise_conv3x3(inp, kern)))

        if neon_kernels is not None:
            neon_us = row("depthwise_conv3x3", "neon", f"{H}x{W}",
                           time_fn(lambda: neon_kernels.depthwise_conv3x3(inp, kern)))
            speedup = scalar_us / neon_us
            print(f"dw_conv {H}x{W}:  scalar={scalar_us:.1f}us  neon={neon_us:.1f}us  speedup={speedup:.2f}x")
        else:
            print(f"dw_conv {H}x{W}:  scalar={scalar_us:.1f}us  (neon not available)")

    # ---- 3. pointwise GEMM ----
    for M, K, N in [(49, 16, 96), (196, 32, 96)]:
        A = rng.integers(-127, 127, (M, K), dtype=np.int8)
        B = rng.integers(-127, 127, (K, N), dtype=np.int8)

        scalar_us = row("pointwise_gemm", "numpy_scalar", f"{M}x{K}x{N}",
                         time_fn(lambda: np_pointwise_gemm(A, B)))

        if neon_kernels is not None:
            neon_us = row("pointwise_gemm", "neon", f"{M}x{K}x{N}",
                           time_fn(lambda: neon_kernels.pointwise_gemm(A, B)))
            speedup = scalar_us / neon_us
            print(f"ptwise {M}x{K}x{N}:  scalar={scalar_us:.1f}us  neon={neon_us:.1f}us  speedup={speedup:.2f}x")
        else:
            print(f"ptwise {M}x{K}x{N}:  scalar={scalar_us:.1f}us  (neon not available)")

    return rows


# ---------------------------------------------------------------------------
# Validate NEON numerics against numpy
# ---------------------------------------------------------------------------

def validate_neon(nk):
    rng = np.random.default_rng(0)
    ok = True

    # dot
    a = rng.integers(-127, 127, 256, dtype=np.int8)
    b = rng.integers(-127, 127, 256, dtype=np.int8)
    ref  = np_dot_int8(a, b)
    simd = nk.dot(a, b)
    match = ref == simd
    ok &= match
    print(f"[{'PASS' if match else 'FAIL'}] dot_int8 N=256  ref={ref}  neon={simd}")

    # depthwise
    inp  = rng.integers(-127, 127, (20, 20), dtype=np.int8)
    kern = rng.integers(-127, 127, 9,        dtype=np.int8)
    ref  = np_depthwise_conv3x3(inp, kern)
    simd = nk.depthwise_conv3x3(inp, kern)
    match = np.array_equal(ref, simd)
    ok &= match
    print(f"[{'PASS' if match else 'FAIL'}] depthwise_conv3x3 20x20")
    if not match:
        diff = np.abs(ref.astype(np.int64) - simd.astype(np.int64))
        print(f"  max diff = {diff.max()}  at {np.unravel_index(diff.argmax(), diff.shape)}")

    # GEMM
    A = rng.integers(-127, 127, (16, 32), dtype=np.int8)
    B = rng.integers(-127, 127, (32, 16), dtype=np.int8)
    ref  = np_pointwise_gemm(A, B)
    simd = nk.pointwise_gemm(A, B)
    match = np.array_equal(ref, simd)
    ok &= match
    print(f"[{'PASS' if match else 'FAIL'}] pointwise_gemm 16x32x16")

    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neon", action="store_true",
                        help="Load libneon_kernels.so and benchmark NEON path")
    parser.add_argument("--validate", action="store_true",
                        help="Validate NEON kernels against numpy reference")
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()
    RUNS = args.runs

    nk = None
    if args.neon:
        from kernels.neon_bridge import NeonKernels
        nk = NeonKernels()
        if args.validate:
            print("\n=== Numerical Validation ===")
            all_ok = validate_neon(nk)
            print(f"\n{'ALL NEON KERNELS VALIDATED OK' if all_ok else 'VALIDATION FAILURES DETECTED'}")
            if not all_ok:
                sys.exit(1)
            print()

    print("=== Micro-benchmark Results ===")
    rows = run_benchmarks(nk)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kernel", "variant", "param", "latency_us"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to: {RESULTS_CSV}")
