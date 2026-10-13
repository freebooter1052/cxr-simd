/**
 * kernels/scalar_ref/scalar_kernels.h
 * ------------------------------------
 * Pure-C scalar reference implementations that are numerically identical
 * to the NEON kernels but have zero SIMD dependency.
 *
 * Used for:
 *   (a) Cross-platform CI (x86 build agents).
 *   (b) Numerical validation on the Pi itself:
 *       SIMD output must match scalar output element-wise (exact, no ULP tolerance).
 */

#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

int32_t scalar_dot_int8(const int8_t* a, const int8_t* b, int n);

void scalar_depthwise_conv3x3_int8(const int8_t* input,
                                   const int8_t* kernel,
                                   int32_t*      output,
                                   int H, int W,
                                   int OH, int OW,
                                   int stride);

void scalar_pointwise_gemm_int8(const int8_t* A,
                                const int8_t* B,
                                int32_t*      C,
                                int M, int K, int N);

void scalar_requantize(const int32_t* in,
                       int8_t*        out,
                       int            n,
                       float          scale,
                       int32_t        zero_point);

#ifdef __cplusplus
}
#endif
