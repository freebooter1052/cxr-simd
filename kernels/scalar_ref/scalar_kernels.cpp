/**
 * kernels/scalar_ref/scalar_kernels.cpp
 * --------------------------------------
 * Scalar (no-SIMD) reference implementations.
 * These are deliberately simple and easy to verify by inspection.
 */

#include "scalar_kernels.h"
#include <cmath>
#include <algorithm>

int32_t scalar_dot_int8(const int8_t* a, const int8_t* b, int n)
{
    int32_t acc = 0;
    for (int i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

void scalar_depthwise_conv3x3_int8(const int8_t* input,
                                   const int8_t* kernel,
                                   int32_t*      output,
                                   int H, int W,
                                   int OH, int OW,
                                   int stride)
{
    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            int32_t s = 0;
            int ih = oh * stride;
            int iw = ow * stride;
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++)
                    s += (int32_t)input[(ih + ki) * W + (iw + kj)]
                       * (int32_t)kernel[ki * 3 + kj];
            output[oh * OW + ow] = s;
        }
    }
}

void scalar_pointwise_gemm_int8(const int8_t* A,
                                const int8_t* B,
                                int32_t*      C,
                                int M, int K, int N)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int32_t)A[m * K + k] * (int32_t)B[k * N + n];
            C[m * N + n] = s;
        }
}

void scalar_requantize(const int32_t* in,
                       int8_t*        out,
                       int            n,
                       float          scale,
                       int32_t        zero_point)
{
    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)roundf((float)in[i] * scale) + zero_point;
        v = v < -128 ? -128 : (v > 127 ? 127 : v);
        out[i] = (int8_t)v;
    }
}
