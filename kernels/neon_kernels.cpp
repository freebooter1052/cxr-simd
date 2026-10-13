/**
 * kernels/neon_kernels.cpp
 * ------------------------
 * ARM NEON INT8 kernel implementations.
 *
 * Build on Raspberry Pi 4/5 (aarch64 Raspberry Pi OS 64-bit):
 *   g++ -O3 -mcpu=native -std=c++17 -ffast-math \
 *       neon_kernels.cpp scalar_ref/scalar_kernels.cpp \
 *       -o neon_kernels_test
 *
 * The NEON intrinsics are guarded by __ARM_NEON so the file can also
 * compile on x86 (to scalar fallback) for CI/unit-test purposes.
 */

#include "neon_kernels.h"
#include <cstring>
#include <cmath>
#include <algorithm>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ================================================================== */
/*  1.  simd_dot_int8                                                  */
/* ================================================================== */

int32_t simd_dot_int8(const int8_t* __restrict__ a,
                      const int8_t* __restrict__ b,
                      int n)
{
#ifdef __ARM_NEON
    /*
     * Each NEON register holds 16 x int8.
     * vmull_s8  : multiply low 8 lanes  -> int16x8
     * vmull_high_s8 : multiply high 8 lanes -> int16x8  (A64 only)
     * vpadalq_s16   : pairwise-add-and-accumulate int16 -> int32
     *
     * We process 32 elements per outer iteration (2 x vld1q_s8 each side)
     * to hide load latency on the Cortex-A72/A76 pipeline.
     */
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);

    int i = 0;
    for (; i <= n - 32; i += 32) {
        int8x16_t va0 = vld1q_s8(a + i);
        int8x16_t vb0 = vld1q_s8(b + i);
        int8x16_t va1 = vld1q_s8(a + i + 16);
        int8x16_t vb1 = vld1q_s8(b + i + 16);

        acc0 = vpadalq_s16(acc0, vmull_s8(vget_low_s8(va0),  vget_low_s8(vb0)));
        acc0 = vpadalq_s16(acc0, vmull_high_s8(va0, vb0));
        acc1 = vpadalq_s16(acc1, vmull_s8(vget_low_s8(va1),  vget_low_s8(vb1)));
        acc1 = vpadalq_s16(acc1, vmull_high_s8(va1, vb1));
    }
    /* handle remaining 16-element chunk */
    for (; i <= n - 16; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        acc0 = vpadalq_s16(acc0, vmull_s8(vget_low_s8(va),  vget_low_s8(vb)));
        acc0 = vpadalq_s16(acc0, vmull_high_s8(va, vb));
    }

    /* horizontal reduction */
    int32x4_t total = vaddq_s32(acc0, acc1);
    int32x2_t lo    = vget_low_s32(total);
    int32x2_t hi    = vget_high_s32(total);
    int32x2_t sum2  = vadd_s32(lo, hi);
    int32_t   result = vget_lane_s32(vpadd_s32(sum2, sum2), 0);

    /* scalar tail (< 16 elements) */
    for (; i < n; i++) result += (int32_t)a[i] * b[i];
    return result;

#else  /* scalar fallback for non-ARM builds */
    int32_t acc = 0;
    for (int i = 0; i < n; i++) acc += (int32_t)a[i] * b[i];
    return acc;
#endif
}

/* ================================================================== */
/*  2.  simd_depthwise_conv3x3_int8                                    */
/* ================================================================== */

void simd_depthwise_conv3x3_int8(const int8_t*  __restrict__ input,
                                  const int8_t*  __restrict__ kernel,
                                  int32_t*       __restrict__ output,
                                  int H, int W,
                                  int OH, int OW,
                                  int stride)
{
    /*
     * Strategy
     * --------
     * For each output row (oh), we load three input rows (r0,r1,r2).
     * For each output column, we multiply each kernel weight against
     * 16 consecutive elements from the shifted input row and
     * accumulate into int32.
     *
     * When OW >= 16 we use a vectorised inner loop; the tail is scalar.
     *
     * Layout: input[h*W + w], output[oh*OW + ow]
     */
#ifdef __ARM_NEON
    /* Broadcast each kernel weight into a lane */
    int8_t k[9];
    for (int i = 0; i < 9; i++) k[i] = kernel[i];

    for (int oh = 0; oh < OH; oh++) {
        int ih = oh * stride;
        const int8_t* r0 = input + (ih    ) * W;
        const int8_t* r1 = input + (ih + 1) * W;
        const int8_t* r2 = input + (ih + 2) * W;
        int32_t*      out_row = output + oh * OW;

        int ow = 0;
        for (; ow <= OW - 16; ow += 16) {
            int iw = ow * stride;

            /* load 18 bytes from each row (3 kernel cols * 16 output cols,
               overlapping; for stride=1 we need iw .. iw+17) */
            int8x16_t r0_0 = vld1q_s8(r0 + iw);
            int8x16_t r0_1 = vld1q_s8(r0 + iw + 1);
            int8x16_t r0_2 = vld1q_s8(r0 + iw + 2);
            int8x16_t r1_0 = vld1q_s8(r1 + iw);
            int8x16_t r1_1 = vld1q_s8(r1 + iw + 1);
            int8x16_t r1_2 = vld1q_s8(r1 + iw + 2);
            int8x16_t r2_0 = vld1q_s8(r2 + iw);
            int8x16_t r2_1 = vld1q_s8(r2 + iw + 1);
            int8x16_t r2_2 = vld1q_s8(r2 + iw + 2);

            /* Widen to int16 and multiply with kernel scalars */
            int16x8_t a0 = vmovl_s8(vget_low_s8(r0_0));
            int16x8_t a1 = vmovl_high_s8(r0_0);
            int16x8_t b0 = vmovl_s8(vget_low_s8(r0_1));
            int16x8_t b1 = vmovl_high_s8(r0_1);
            int16x8_t c0 = vmovl_s8(vget_low_s8(r0_2));
            int16x8_t c1 = vmovl_high_s8(r0_2);

            int32x4_t acc0 = vmull_n_s16(vget_low_s16(a0),  (int16_t)k[0]);
            int32x4_t acc1 = vmull_n_s16(vget_high_s16(a0), (int16_t)k[0]);
            int32x4_t acc2 = vmull_n_s16(vget_low_s16(a1),  (int16_t)k[0]);
            int32x4_t acc3 = vmull_n_s16(vget_high_s16(a1), (int16_t)k[0]);

            acc0 = vmlal_n_s16(acc0, vget_low_s16(b0),  (int16_t)k[1]);
            acc1 = vmlal_n_s16(acc1, vget_high_s16(b0), (int16_t)k[1]);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(b1),  (int16_t)k[1]);
            acc3 = vmlal_n_s16(acc3, vget_high_s16(b1), (int16_t)k[1]);

            acc0 = vmlal_n_s16(acc0, vget_low_s16(c0),  (int16_t)k[2]);
            acc1 = vmlal_n_s16(acc1, vget_high_s16(c0), (int16_t)k[2]);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(c1),  (int16_t)k[2]);
            acc3 = vmlal_n_s16(acc3, vget_high_s16(c1), (int16_t)k[2]);

            /* Row 1 */
            a0 = vmovl_s8(vget_low_s8(r1_0)); a1 = vmovl_high_s8(r1_0);
            b0 = vmovl_s8(vget_low_s8(r1_1)); b1 = vmovl_high_s8(r1_1);
            c0 = vmovl_s8(vget_low_s8(r1_2)); c1 = vmovl_high_s8(r1_2);

            acc0 = vmlal_n_s16(acc0, vget_low_s16(a0),  (int16_t)k[3]);
            acc1 = vmlal_n_s16(acc1, vget_high_s16(a0), (int16_t)k[3]);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(a1),  (int16_t)k[3]);
            acc3 = vmlal_n_s16(acc3, vget_high_s16(a1), (int16_t)k[3]);

            acc0 = vmlal_n_s16(acc0, vget_low_s16(b0),  (int16_t)k[4]);
            acc1 = vmlal_n_s16(acc1, vget_high_s16(b0), (int16_t)k[4]);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(b1),  (int16_t)k[4]);
            acc3 = vmlal_n_s16(acc3, vget_high_s16(b1), (int16_t)k[4]);

            acc0 = vmlal_n_s16(acc0, vget_low_s16(c0),  (int16_t)k[5]);
            acc1 = vmlal_n_s16(acc1, vget_high_s16(c0), (int16_t)k[5]);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(c1),  (int16_t)k[5]);
            acc3 = vmlal_n_s16(acc3, vget_high_s16(c1), (int16_t)k[5]);

            /* Row 2 */
            a0 = vmovl_s8(vget_low_s8(r2_0)); a1 = vmovl_high_s8(r2_0);
            b0 = vmovl_s8(vget_low_s8(r2_1)); b1 = vmovl_high_s8(r2_1);
            c0 = vmovl_s8(vget_low_s8(r2_2)); c1 = vmovl_high_s8(r2_2);

            acc0 = vmlal_n_s16(acc0, vget_low_s16(a0),  (int16_t)k[6]);
            acc1 = vmlal_n_s16(acc1, vget_high_s16(a0), (int16_t)k[6]);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(a1),  (int16_t)k[6]);
            acc3 = vmlal_n_s16(acc3, vget_high_s16(a1), (int16_t)k[6]);

            acc0 = vmlal_n_s16(acc0, vget_low_s16(b0),  (int16_t)k[7]);
            acc1 = vmlal_n_s16(acc1, vget_high_s16(b0), (int16_t)k[7]);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(b1),  (int16_t)k[7]);
            acc3 = vmlal_n_s16(acc3, vget_high_s16(b1), (int16_t)k[7]);

            acc0 = vmlal_n_s16(acc0, vget_low_s16(c0),  (int16_t)k[8]);
            acc1 = vmlal_n_s16(acc1, vget_high_s16(c0), (int16_t)k[8]);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(c1),  (int16_t)k[8]);
            acc3 = vmlal_n_s16(acc3, vget_high_s16(c1), (int16_t)k[8]);

            vst1q_s32(out_row + ow,      acc0);
            vst1q_s32(out_row + ow + 4,  acc1);
            vst1q_s32(out_row + ow + 8,  acc2);
            vst1q_s32(out_row + ow + 12, acc3);
        }

        /* scalar tail */
        for (; ow < OW; ow++) {
            int iw = ow * stride;
            int32_t s = 0;
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++)
                    s += (int32_t)r0[(ki==0?0:(ki==1?W:2*W)) + iw + kj]
                       * (int32_t)k[ki * 3 + kj];
            /* above indexing was wrong for variable row - fix: */
            out_row[ow] = 0; /* will be recomputed in scalar below */
        }
        /* redo scalar tail cleanly */
        for (int cow = (OW / 16) * 16; cow < OW; cow++) {
            int iw = cow * stride;
            int32_t s = 0;
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++)
                    s += (int32_t)input[(ih + ki) * W + iw + kj]
                       * (int32_t)k[ki * 3 + kj];
            out_row[cow] = s;
        }
    }

#else  /* scalar fallback */
    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            int32_t s = 0;
            int ih = oh * stride, iw = ow * stride;
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++)
                    s += (int32_t)input[(ih + ki) * W + iw + kj]
                       * (int32_t)kernel[ki * 3 + kj];
            output[oh * OW + ow] = s;
        }
    }
#endif
}

/* ================================================================== */
/*  3.  simd_pointwise_gemm_int8                                       */
/* ================================================================== */

void simd_pointwise_gemm_int8(const int8_t*  __restrict__ A,
                               const int8_t*  __restrict__ B,
                               int32_t*       __restrict__ C,
                               int M, int K, int N)
{
    /*
     * Naive vectorised GEMM: for each (m, n) pair, dot-product row m of A
     * against column n of B.  We process K in chunks of 16 using NEON.
     *
     * Note: B is stored row-major [K, N], so column n starts at B[0*N+n]
     * with stride N — not contiguous.  For small N (typical pointwise in
     * MobileNetV3-Small: N <= 96) this is acceptable.
     * For production, B should be pre-transposed to [N, K] for locality.
     */
#ifdef __ARM_NEON
    for (int m = 0; m < M; m++) {
        const int8_t* a_row = A + m * K;
        for (int n = 0; n < N; n++) {
            int32x4_t acc = vdupq_n_s32(0);
            int k = 0;
            for (; k <= K - 16; k += 16) {
                int8x16_t va = vld1q_s8(a_row + k);
                /* gather B column n — must read strided */
                int8_t tmp[16];
                for (int t = 0; t < 16; t++) tmp[t] = B[(k + t) * N + n];
                int8x16_t vb = vld1q_s8(tmp);
                acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va),  vget_low_s8(vb)));
                acc = vpadalq_s16(acc, vmull_high_s8(va, vb));
            }
            int32_t s = vaddvq_s32(acc);
            for (; k < K; k++) s += (int32_t)a_row[k] * B[k * N + n];
            C[m * N + n] = s;
        }
    }
#else
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++) s += (int32_t)A[m*K+k] * B[k*N+n];
            C[m*N+n] = s;
        }
#endif
}

/* ================================================================== */
/*  4.  simd_requantize                                                 */
/* ================================================================== */

void simd_requantize(const int32_t* __restrict__ in,
                     int8_t*        __restrict__ out,
                     int            n,
                     float          scale,
                     int32_t        zero_point)
{
#ifdef __ARM_NEON
    float32x4_t vscale = vdupq_n_f32(scale);
    int32x4_t   vzp    = vdupq_n_s32(zero_point);

    int i = 0;
    for (; i <= n - 16; i += 16) {
        /* Load 16 int32 values */
        int32x4_t v0 = vld1q_s32(in + i);
        int32x4_t v1 = vld1q_s32(in + i + 4);
        int32x4_t v2 = vld1q_s32(in + i + 8);
        int32x4_t v3 = vld1q_s32(in + i + 12);

        /* Convert to float, scale, round, add zero_point */
        v0 = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vcvtq_f32_s32(v0), vscale)));
        v1 = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vcvtq_f32_s32(v1), vscale)));
        v2 = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vcvtq_f32_s32(v2), vscale)));
        v3 = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vcvtq_f32_s32(v3), vscale)));

        v0 = vaddq_s32(v0, vzp);
        v1 = vaddq_s32(v1, vzp);
        v2 = vaddq_s32(v2, vzp);
        v3 = vaddq_s32(v3, vzp);

        /* Saturating narrow: int32 -> int16 -> int8 */
        int16x8_t s01 = vcombine_s16(vqmovn_s32(v0), vqmovn_s32(v1));
        int16x8_t s23 = vcombine_s16(vqmovn_s32(v2), vqmovn_s32(v3));
        int8x16_t r   = vcombine_s8 (vqmovn_s16(s01), vqmovn_s16(s23));
        vst1q_s8(out + i, r);
    }
    /* scalar tail */
    for (; i < n; i++) {
        int32_t v = (int32_t)roundf((float)in[i] * scale) + zero_point;
        v = v < -128 ? -128 : v > 127 ? 127 : v;
        out[i] = (int8_t)v;
    }
#else
    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)roundf((float)in[i] * scale) + zero_point;
        v = v < -128 ? -128 : v > 127 ? 127 : v;
        out[i] = (int8_t)v;
    }
#endif
}
