/**
 * kernels/tests/validate_kernels.cpp
 * -----------------------------------
 * Numerical validation harness: runs every NEON kernel and its scalar twin
 * on randomised inputs, verifying exact integer equality.
 *
 * Build on Pi:
 *   g++ -O3 -mcpu=native -std=c++17 \
 *       ../neon_kernels.cpp \
 *       ../scalar_ref/scalar_kernels.cpp \
 *       validate_kernels.cpp \
 *       -o validate && ./validate
 *
 * Expected output: all "PASS" lines, then "ALL TESTS PASSED".
 * Any FAIL line indicates a silent correctness bug in the NEON path.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

#include "../neon_kernels.h"
#include "../scalar_ref/scalar_kernels.h"

/* ---- helpers ---- */
static std::mt19937 rng(42);
static auto rng_int8 = []() -> int8_t {
    return (int8_t)(rng() % 256 - 128);
};

static bool all_pass = true;

static void check(const char* name, bool ok) {
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) all_pass = false;
}

/* ================================================================== */
/*  Test 1: dot product                                                */
/* ================================================================== */
static void test_dot() {
    const int N = 256;
    std::vector<int8_t> a(N), b(N);
    for (auto& x : a) x = rng_int8();
    for (auto& x : b) x = rng_int8();

    int32_t ref  = scalar_dot_int8(a.data(), b.data(), N);
    int32_t simd = simd_dot_int8  (a.data(), b.data(), N);
    char name[64]; snprintf(name, sizeof(name), "dot_int8  N=%d  ref=%d simd=%d", N, ref, simd);
    check(name, ref == simd);

    /* also test non-multiple-of-32 */
    const int N2 = 48;
    ref  = scalar_dot_int8(a.data(), b.data(), N2);
    simd = simd_dot_int8  (a.data(), b.data(), N2);
    snprintf(name, sizeof(name), "dot_int8  N=%d  ref=%d simd=%d", N2, ref, simd);
    check(name, ref == simd);
}

/* ================================================================== */
/*  Test 2: depthwise conv 3x3                                         */
/* ================================================================== */
static void test_depthwise() {
    /* 20x20 input -> 18x18 output with stride=1, 1-padded for clarity */
    const int H = 20, W = 20, OH = 18, OW = 18;
    std::vector<int8_t>  input (H * W),  kern(9);
    std::vector<int32_t> ref_out(OH * OW), simd_out(OH * OW);

    for (auto& x : input) x = rng_int8();
    for (auto& x : kern)  x = rng_int8();

    scalar_depthwise_conv3x3_int8(input.data(), kern.data(), ref_out.data(),
                                   H, W, OH, OW, 1);
    simd_depthwise_conv3x3_int8  (input.data(), kern.data(), simd_out.data(),
                                   H, W, OH, OW, 1);

    bool ok = (memcmp(ref_out.data(), simd_out.data(), OH*OW*sizeof(int32_t)) == 0);
    check("depthwise_conv3x3_int8 stride=1 18x18", ok);
    if (!ok) {
        /* print first mismatch */
        for (int i = 0; i < OH * OW; i++)
            if (ref_out[i] != simd_out[i]) {
                printf("  First mismatch at [%d]: ref=%d simd=%d\n",
                       i, ref_out[i], simd_out[i]);
                break;
            }
    }

    /* stride=2 */
    const int OH2 = 9, OW2 = 9;
    std::vector<int32_t> ref2(OH2*OW2), simd2(OH2*OW2);
    scalar_depthwise_conv3x3_int8(input.data(), kern.data(), ref2.data(),
                                   H, W, OH2, OW2, 2);
    simd_depthwise_conv3x3_int8  (input.data(), kern.data(), simd2.data(),
                                   H, W, OH2, OW2, 2);
    ok = (memcmp(ref2.data(), simd2.data(), OH2*OW2*sizeof(int32_t)) == 0);
    check("depthwise_conv3x3_int8 stride=2 9x9", ok);
}

/* ================================================================== */
/*  Test 3: pointwise GEMM                                             */
/* ================================================================== */
static void test_pointwise_gemm() {
    const int M = 16, K = 32, N = 16;
    std::vector<int8_t>  A(M*K), B(K*N);
    std::vector<int32_t> ref_C(M*N), simd_C(M*N);

    for (auto& x : A) x = rng_int8();
    for (auto& x : B) x = rng_int8();

    scalar_pointwise_gemm_int8(A.data(), B.data(), ref_C.data(),  M, K, N);
    simd_pointwise_gemm_int8  (A.data(), B.data(), simd_C.data(), M, K, N);

    bool ok = (memcmp(ref_C.data(), simd_C.data(), M*N*sizeof(int32_t)) == 0);
    check("pointwise_gemm_int8 16x32x16", ok);
    if (!ok) {
        for (int i = 0; i < M*N; i++)
            if (ref_C[i] != simd_C[i]) {
                printf("  First mismatch at [%d]: ref=%d simd=%d\n",
                       i, ref_C[i], simd_C[i]);
                break;
            }
    }
}

/* ================================================================== */
/*  Test 4: requantize                                                 */
/* ================================================================== */
static void test_requantize() {
    const int N = 128;
    std::vector<int32_t> in(N);
    std::vector<int8_t>  ref_out(N), simd_out(N);

    for (auto& x : in) x = (int32_t)(rng() % 65536) - 32768;

    float scale = 0.00392f;  /* ~1/255 */
    int32_t zp  = -1;

    scalar_requantize(in.data(), ref_out.data(),  N, scale, zp);
    simd_requantize  (in.data(), simd_out.data(), N, scale, zp);

    bool ok = (memcmp(ref_out.data(), simd_out.data(), N) == 0);
    check("requantize N=128", ok);
    if (!ok) {
        for (int i = 0; i < N; i++)
            if (ref_out[i] != simd_out[i]) {
                printf("  First mismatch at [%d]: in=%d ref=%d simd=%d\n",
                       i, in[i], (int)ref_out[i], (int)simd_out[i]);
                break;
            }
    }
}

/* ================================================================== */
int main() {
    printf("=== Kernel Numerical Validation ===\n");
    test_dot();
    test_depthwise();
    test_pointwise_gemm();
    test_requantize();
    printf("===================================\n");
    printf("%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
