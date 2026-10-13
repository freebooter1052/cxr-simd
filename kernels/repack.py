# -*- coding: utf-8 -*-
"""
kernels/repack.py
-----------------
Tensor layout repacking utilities: converts PyTorch-default NCHW tensors
(or weight tensors in [OC, IC, KH, KW]) to SIMD-friendly blocked layouts.

Background
----------
ARM NEON processes 16 x int8 per register (128-bit).  For INT8 inference
to achieve maximum lane utilisation, activations and weights must be laid
out so that consecutive memory addresses correspond to consecutive channels
that can be loaded with a single vld1q_s8 instruction.

This module provides:
  1. nchw_to_nhwc(x)           - activation layout conversion
  2. pack_depthwise_weights(w)  - [C, 1, KH, KW] -> flat [C, KH*KW]
  3. pack_pointwise_weights(w)  - [OC, IC, 1, 1] -> [IC, OC] (for GEMM)
  4. quantize_to_int8(x, scale, zp) - float tensor -> int8 ndarray
  5. unpack_nhwc_to_nchw(x, C) - round-trip for debugging

All functions work on numpy arrays for easy ctypes interop.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1.  Activation layout: NCHW -> NHWC
# ---------------------------------------------------------------------------

def nchw_to_nhwc(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a float/int8 PyTorch tensor from NCHW to NHWC layout.

    Parameters
    ----------
    tensor : torch.Tensor   shape [N, C, H, W]

    Returns
    -------
    numpy.ndarray   shape [N, H, W, C],  dtype=float32 or int8
                    C-contiguous (row-major)

    Why NHWC for NEON?
    ------------------
    In NHWC, all channels for a given (n, h, w) pixel are contiguous.
    A single vld1q_s8 loads 16 channels at once, enabling the vectorised
    depthwise and pointwise kernels to process full channel-chunks without
    gather operations.  NCHW puts all H×W values for one channel
    contiguously — great for GPU coalescing, bad for NEON channel loops.
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    return tensor.permute(0, 2, 3, 1).contiguous().numpy()


def nhwc_to_nchw(arr: np.ndarray) -> torch.Tensor:
    """Round-trip: NHWC numpy -> NCHW torch (for debugging only)."""
    t = torch.from_numpy(arr)           # [N, H, W, C]
    return t.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# 2.  Depthwise weight packing: [C, 1, KH, KW] -> [C, KH*KW]
# ---------------------------------------------------------------------------

def pack_depthwise_weights(weight: torch.Tensor) -> np.ndarray:
    """
    Pack depthwise conv weights for the simd_depthwise_conv3x3_int8 kernel.

    PyTorch stores depthwise weights as [C_out, C_in=1, KH, KW].
    The kernel expects a flat [KH*KW] int8 array per channel.

    Parameters
    ----------
    weight : torch.Tensor  shape [C, 1, KH, KW],  dtype float32 or int8

    Returns
    -------
    numpy.ndarray  shape [C, KH*KW],  dtype int8  (C-contiguous)
    """
    C, _, KH, KW = weight.shape
    w = weight.squeeze(1)               # [C, KH, KW]
    w = w.reshape(C, KH * KW)
    arr = w.detach().numpy()
    if arr.dtype != np.int8:
        arr = arr.astype(np.int8)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# 3.  Pointwise weight packing: [OC, IC, 1, 1] -> [IC, OC]
# ---------------------------------------------------------------------------

def pack_pointwise_weights(weight: torch.Tensor) -> np.ndarray:
    """
    Transpose pointwise conv weights for GEMM layout [K=IC, N=OC].

    In simd_pointwise_gemm_int8(A, B, C, M, K, N):
      A = activations  [M=pixels, K=IC]
      B = weights      [K=IC,     N=OC]  <- this function produces B

    PyTorch stores pointwise weights as [OC, IC, 1, 1], so we squeeze
    the spatial dims and transpose.

    Parameters
    ----------
    weight : torch.Tensor  shape [OC, IC, 1, 1]

    Returns
    -------
    numpy.ndarray  shape [IC, OC],  dtype int8  (C-contiguous)
    """
    OC, IC, _, _ = weight.shape
    w = weight.squeeze(-1).squeeze(-1)  # [OC, IC]
    w = w.t()                           # [IC, OC]
    arr = w.detach().numpy()
    if arr.dtype != np.int8:
        arr = arr.astype(np.int8)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# 4.  Float -> INT8 quantization (symmetric, per-tensor)
# ---------------------------------------------------------------------------

def quantize_to_int8(tensor: torch.Tensor,
                     scale: float = None,
                     zero_point: int = 0) -> tuple:
    """
    Quantize a float tensor to int8.

    Parameters
    ----------
    tensor     : torch.Tensor  (any shape), float32
    scale      : float         per-tensor scale; if None, computed from max-abs
    zero_point : int           default 0 (symmetric quantization)

    Returns
    -------
    arr_int8   : numpy.ndarray  int8, same shape
    scale      : float          (scale used, for requantize output)
    zero_point : int
    """
    arr = tensor.detach().float().numpy()
    if scale is None:
        max_abs = np.max(np.abs(arr))
        scale = float(max_abs / 127.0) if max_abs > 0 else 1.0

    q = np.round(arr / scale).astype(np.int32) + zero_point
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scale, zero_point


def dequantize_from_int8(arr: np.ndarray, scale: float,
                          zero_point: int = 0) -> np.ndarray:
    """Inverse of quantize_to_int8, returns float32 numpy array."""
    return (arr.astype(np.float32) - zero_point) * scale


# ---------------------------------------------------------------------------
# 5.  Channel-blocking utility (for NEON 16-wide processing)
# ---------------------------------------------------------------------------

def pad_channels_to_multiple(arr: np.ndarray, multiple: int = 16) -> np.ndarray:
    """
    Pad the channel dimension (last axis for NHWC) to a multiple of multiple.
    Padding value is 0 (zero-point for symmetric quantization).

    Parameters
    ----------
    arr      : numpy.ndarray  [..., C]
    multiple : int            16 for NEON int8 (128-bit / 8-bit)

    Returns
    -------
    numpy.ndarray  [..., C_padded]
    """
    C = arr.shape[-1]
    pad = (-C) % multiple
    if pad == 0:
        return arr
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, pad)]
    return np.pad(arr, pad_width, mode='constant', constant_values=0)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== repack.py smoke test ===")

    # Create a fake NCHW activation tensor
    x = torch.randn(1, 32, 7, 7)
    nhwc = nchw_to_nhwc(x)
    print(f"NCHW {tuple(x.shape)}  ->  NHWC {nhwc.shape}  (C-contig={nhwc.flags['C_CONTIGUOUS']})")

    # Round-trip
    x_back = nhwc_to_nchw(nhwc)
    max_diff = (x - x_back).abs().max().item()
    print(f"Round-trip max abs diff: {max_diff:.2e}  (expect 0.00e+00)")

    # Quantize
    q, scale, zp = quantize_to_int8(x)
    dq = dequantize_from_int8(q, scale, zp)
    quant_err = np.max(np.abs(x.numpy() - dq))
    print(f"Quantization max error:  {quant_err:.4f}  (expect < scale/2 = {scale/2:.4f})")

    # Depthwise weight pack
    dw = torch.randint(-127, 127, (32, 1, 3, 3), dtype=torch.int8).float()
    packed_dw = pack_depthwise_weights(dw.to(torch.float32))
    print(f"Depthwise weight pack:   {dw.shape} -> {packed_dw.shape}")

    # Pointwise weight pack
    pw = torch.randint(-127, 127, (64, 32, 1, 1), dtype=torch.int8).float()
    packed_pw = pack_pointwise_weights(pw)
    print(f"Pointwise weight pack:   {pw.shape} -> {packed_pw.shape}  [IC, OC]")

    # Pad channels
    fake_nhwc = np.zeros((1, 7, 7, 30), dtype=np.int8)
    padded = pad_channels_to_multiple(fake_nhwc, 16)
    print(f"Channel pad:             C=30 -> C={padded.shape[-1]} (multiple of 16)")

    print("\nAll smoke tests OK.")
