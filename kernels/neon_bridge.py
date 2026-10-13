# -*- coding: utf-8 -*-
"""
kernels/neon_bridge.py
-----------------------
Python ctypes bridge to the compiled libneon_kernels.so.

This module loads the shared library built by CMakeLists.txt and exposes
each NEON kernel as a callable Python function with proper type signatures.

On Raspberry Pi, build the library first:
    cd kernels && mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j4           # produces build/libneon_kernels.so

Then import this module from Python:
    from kernels.neon_bridge import NeonKernels
    nk = NeonKernels()          # auto-finds the .so
    result = nk.dot(a, b)
"""

import ctypes
import numpy as np
import os
from pathlib import Path

_LIB_SEARCH_PATHS = [
    Path(__file__).parent / "build" / "libneon_kernels.so",
    Path(__file__).parent / "build" / "neon_kernels.dll",   # Windows (scalar fallback)
    Path("/usr/local/lib/libneon_kernels.so"),               # system install
]


class NeonKernels:
    """
    Python wrapper around compiled NEON kernel shared library.

    Usage
    -----
    nk = NeonKernels()
    a = np.random.randint(-127, 127, 256, dtype=np.int8)
    b = np.random.randint(-127, 127, 256, dtype=np.int8)
    print(nk.dot(a, b))
    """

    def __init__(self, lib_path: str = None):
        if lib_path is None:
            lib_path = self._find_lib()
        self._lib = ctypes.CDLL(str(lib_path))
        self._setup_signatures()
        print(f"[NeonKernels] Loaded: {lib_path}")

    @staticmethod
    def _find_lib():
        for p in _LIB_SEARCH_PATHS:
            if p.exists():
                return p
        raise FileNotFoundError(
            "libneon_kernels.so not found. Build it first:\n"
            "  cd kernels && mkdir build && cd build\n"
            "  cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4"
        )

    def _setup_signatures(self):
        lib = self._lib

        # int32_t simd_dot_int8(const int8_t* a, const int8_t* b, int n)
        lib.simd_dot_int8.restype  = ctypes.c_int32
        lib.simd_dot_int8.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
        ]

        # void simd_depthwise_conv3x3_int8(...)
        lib.simd_depthwise_conv3x3_int8.restype  = None
        lib.simd_depthwise_conv3x3_int8.argtypes = [
            ctypes.POINTER(ctypes.c_int8),   # input
            ctypes.POINTER(ctypes.c_int8),   # kernel
            ctypes.POINTER(ctypes.c_int32),  # output
            ctypes.c_int, ctypes.c_int,      # H, W
            ctypes.c_int, ctypes.c_int,      # OH, OW
            ctypes.c_int,                    # stride
        ]

        # void simd_pointwise_gemm_int8(...)
        lib.simd_pointwise_gemm_int8.restype  = None
        lib.simd_pointwise_gemm_int8.argtypes = [
            ctypes.POINTER(ctypes.c_int8),   # A [M, K]
            ctypes.POINTER(ctypes.c_int8),   # B [K, N]
            ctypes.POINTER(ctypes.c_int32),  # C [M, N]
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # M, K, N
        ]

        # void simd_requantize(...)
        lib.simd_requantize.restype  = None
        lib.simd_requantize.argtypes = [
            ctypes.POINTER(ctypes.c_int32),  # in
            ctypes.POINTER(ctypes.c_int8),   # out
            ctypes.c_int,                    # n
            ctypes.c_float,                  # scale
            ctypes.c_int32,                  # zero_point
        ]

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def dot(self, a: np.ndarray, b: np.ndarray) -> int:
        """INT8 dot product. a and b must be 1D int8, same length."""
        assert a.dtype == np.int8 and b.dtype == np.int8
        assert a.ndim == 1 and a.shape == b.shape
        n = len(a)
        return int(self._lib.simd_dot_int8(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ctypes.c_int(n),
        ))

    def depthwise_conv3x3(self,
                           input: np.ndarray,   # [H, W] int8
                           kernel: np.ndarray,  # [9]    int8
                           stride: int = 1) -> np.ndarray:
        """Depthwise 3×3 conv on a single channel. Returns int32 [OH, OW]."""
        assert input.dtype == np.int8 and kernel.dtype == np.int8
        H, W = input.shape
        OH = (H - 2 + stride - 1) // stride
        OW = (W - 2 + stride - 1) // stride
        out = np.zeros(OH * OW, dtype=np.int32)
        inp_c = np.ascontiguousarray(input).reshape(-1)
        ker_c = np.ascontiguousarray(kernel.reshape(9))
        self._lib.simd_depthwise_conv3x3_int8(
            inp_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ker_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            H, W, OH, OW, stride,
        )
        return out.reshape(OH, OW)

    def pointwise_gemm(self,
                        A: np.ndarray,   # [M, K] int8
                        B: np.ndarray,   # [K, N] int8
                        ) -> np.ndarray: # [M, N] int32
        """Pointwise GEMM: C = A @ B  (int8 × int8 -> int32)."""
        assert A.dtype == np.int8 and B.dtype == np.int8
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        C = np.zeros(M * N, dtype=np.int32)
        A_c = np.ascontiguousarray(A)
        B_c = np.ascontiguousarray(B)
        self._lib.simd_pointwise_gemm_int8(
            A_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            B_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            M, K, N,
        )
        return C.reshape(M, N)

    def requantize(self, inp: np.ndarray, scale: float,
                   zero_point: int = 0) -> np.ndarray:
        """Requantize int32 array to int8. inp must be 1D int32."""
        assert inp.dtype == np.int32
        inp_c = np.ascontiguousarray(inp.reshape(-1))
        out = np.zeros_like(inp_c, dtype=np.int8)
        self._lib.simd_requantize(
            inp_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            len(inp_c),
            ctypes.c_float(scale),
            ctypes.c_int32(zero_point),
        )
        return out.reshape(inp.shape)
