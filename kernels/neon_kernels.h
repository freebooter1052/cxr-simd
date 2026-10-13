/**
 * kernels/neon_kernels.h
 * ----------------------
 * Public API for all hand-written ARM NEON INT8 kernels used in the
 * CXR-SIMD inference pipeline.
 *
 * Design notes:
 *   - All weight and activation buffers must be 16-byte aligned.
 *   - Activations are in NHWC layout; weights are in [OC, KH, KW, IC] for
 *     depthwise and [OC, IC] for pointwise (transposed to row-major GEMM).
 *   - Accumulators are always int32 to avoid overflow; the caller is
 *     responsible for requantisation (scale + zero-point shift back to int8).
 *   - Every function has a scalar twin in scalar_ref.h for validation.
 *
 * Compile with (on Pi):
 *   aarch64-linux-gnu-g++ -O3 -mcpu=native -std=c++17 -ffast-math
 */

#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  1. INT8 dot product  (validates the fundamental SIMD operation)   */
/* ------------------------------------------------------------------ */

/**
 * simd_dot_int8 - vectorised dot product of two int8 arrays.
 * n must be a multiple of 16; pad inputs if needed.
 * Accumulator is int32 - no saturation for typical CNN ranges.
 */
int32_t simd_dot_int8(const int8_t* __restrict__ a,
                      const int8_t* __restrict__ b,
                      int n);

/* ------------------------------------------------------------------ */
/*  2. Depthwise INT8 convolution  (3x3, stride 1 or 2)              */
/* ------------------------------------------------------------------ */

/**
 * simd_depthwise_conv3x3_int8
 *
 * Single-channel depthwise 3x3 convolution.
 * Input  : [H, W]   int8  (pre-padded; no boundary check inside kernel)
 * Kernel : [3, 3]   int8  (row-major, 9 bytes)
 * Output : [OH, OW] int32 (caller allocates; not scaled/quantised)
 * stride : 1 or 2
 */
void simd_depthwise_conv3x3_int8(const int8_t*  __restrict__ input,
                                  const int8_t*  __restrict__ kernel,
                                  int32_t*       __restrict__ output,
                                  int H, int W,
                                  int OH, int OW,
                                  int stride);

/* ------------------------------------------------------------------ */
/*  3. Pointwise (1x1) INT8 GEMM                                      */
/* ------------------------------------------------------------------ */

/**
 * simd_pointwise_gemm_int8
 *
 * C[m,n] = sum_k( A[m,k] * B[k,n] )   (all int8 inputs, int32 output)
 * A : [M, K]  activations
 * B : [K, N]  weights  (row-major)
 * C : [M, N]  int32 output
 */
void simd_pointwise_gemm_int8(const int8_t*  __restrict__ A,
                               const int8_t*  __restrict__ B,
                               int32_t*       __restrict__ C,
                               int M, int K, int N);

/* ------------------------------------------------------------------ */
/*  4. Requantisation  (int32 accumulator -> int8 output)             */
/* ------------------------------------------------------------------ */

/**
 * simd_requantize
 * out[i] = clamp( round(in[i] * scale) + zero_point, -128, 127 )
 */
void simd_requantize(const int32_t* __restrict__ in,
                     int8_t*        __restrict__ out,
                     int            n,
                     float          scale,
                     int32_t        zero_point);

#ifdef __cplusplus
} /* extern "C" */
#endif
