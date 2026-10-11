/**
 * kernels/tests/bench_kernels.cpp
 * --------------------------------
 * Micro-benchmark: measures wall-clock throughput of each kernel
 * (NEON vs scalar) using clock_gettime(CLOCK_MONOTONIC).
 *
 * Build on Pi:
 *   g++ -O3 -mcpu=native -std=c++17 \
 *       ../neon_kernels.cpp \
 *       ../scalar_ref/scalar_kernels.cpp \
 *       bench_kernels.cpp \
 *       -o bench && ./bench
 *
 * Output (CSV rows): kernel,variant,n,latency_us,throughput_gops
 * These rows are consumed by bench/run_latency.py for the final table.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <time.h>

#include "../neon_kernels.h"
#include "../scalar_ref/scalar_kernels.h"

static std::mt19937 rng(42);

/* ---- timing ---- */
static double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

#define RUNS 500

static void print_header() {
    printf("kernel,variant,param_n,latency_us_mean,speedup_vs_scalar\n");
}

/* ================================================================== */
/*  1. dot product                                                      */
/* ================================================================== */
static void bench_dot(int N) {
    std::vector<int8_t> a(N), b(N);
    for (auto& x : a) x = (int8_t)(rng() % 256 - 128);
    for (auto& x : b) x = (int8_t)(rng() % 256 - 128);

    /* warm-up */
    volatile int32_t sink = 0;
    for (int i = 0; i < 10; i++) sink += simd_dot_int8(a.data(), b.data(), N);

    double t0 = now_us();
    for (int r = 0; r < RUNS; r++) sink += simd_dot_int8(a.data(), b.data(), N);
    double simd_us = (now_us() - t0) / RUNS;

    t0 = now_us();
    for (int r = 0; r < RUNS; r++) sink += scalar_dot_int8(a.data(), b.data(), N);
    double scalar_us = (now_us() - t0) / RUNS;
    (void)sink;

    printf("dot_int8,neon,%d,%.3f,%.2fx\n", N, simd_us, scalar_us / simd_us);
    printf("dot_int8,scalar,%d,%.3f,1.00x\n", N, scalar_us);
}

/* ================================================================== */
/*  2. depthwise conv 3x3                                              */
/* ================================================================== */
static void bench_depthwise(int H, int W) {
    int OH = H - 2, OW = W - 2;
    std::vector<int8_t>  input(H*W), kern(9);
    std::vector<int32_t> out(OH*OW);
    for (auto& x : input) x = (int8_t)(rng() % 256 - 128);
    for (auto& x : kern)  x = (int8_t)(rng() % 256 - 128);

    /* warm-up */
    for (int i = 0; i < 5; i++)
        simd_depthwise_conv3x3_int8(input.data(), kern.data(), out.data(),
                                     H, W, OH, OW, 1);

    double t0 = now_us();
    for (int r = 0; r < RUNS; r++)
        simd_depthwise_conv3x3_int8(input.data(), kern.data(), out.data(),
                                     H, W, OH, OW, 1);
    double simd_us = (now_us() - t0) / RUNS;

    t0 = now_us();
    for (int r = 0; r < RUNS; r++)
        scalar_depthwise_conv3x3_int8(input.data(), kern.data(), out.data(),
                                       H, W, OH, OW, 1);
    double scalar_us = (now_us() - t0) / RUNS;

    char tag[32]; snprintf(tag, sizeof(tag), "%dx%d", H, W);
    printf("depthwise_conv3x3,neon,%s,%.3f,%.2fx\n",   tag, simd_us, scalar_us/simd_us);
    printf("depthwise_conv3x3,scalar,%s,%.3f,1.00x\n", tag, scalar_us);
}

/* ================================================================== */
/*  3. pointwise GEMM                                                  */
/* ================================================================== */
static void bench_gemm(int M, int K, int N) {
    std::vector<int8_t>  A(M*K), B(K*N);
    std::vector<int32_t> C(M*N);
    for (auto& x : A) x = (int8_t)(rng() % 256 - 128);
    for (auto& x : B) x = (int8_t)(rng() % 256 - 128);

    for (int i = 0; i < 3; i++)
        simd_pointwise_gemm_int8(A.data(), B.data(), C.data(), M, K, N);

    double t0 = now_us();
    for (int r = 0; r < RUNS; r++)
        simd_pointwise_gemm_int8(A.data(), B.data(), C.data(), M, K, N);
    double simd_us = (now_us() - t0) / RUNS;

    t0 = now_us();
    for (int r = 0; r < RUNS; r++)
        scalar_pointwise_gemm_int8(A.data(), B.data(), C.data(), M, K, N);
    double scalar_us = (now_us() - t0) / RUNS;

    char tag[32]; snprintf(tag, sizeof(tag), "%dx%dx%d", M, K, N);
    printf("pointwise_gemm,neon,%s,%.3f,%.2fx\n",   tag, simd_us, scalar_us/simd_us);
    printf("pointwise_gemm,scalar,%s,%.3f,1.00x\n", tag, scalar_us);
}

/* ================================================================== */
int main() {
    print_header();
    bench_dot(256);
    bench_dot(4096);
    bench_depthwise(34, 34);   /* after first conv in MobileNetV3-Small */
    bench_depthwise(18, 18);
    bench_gemm(49, 16, 96);    /* typical pointwise: 7x7 spatial, 16->96 ch */
    bench_gemm(196, 32, 96);
    return 0;
}
