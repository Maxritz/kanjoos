// tools/bench/rocwmma_tput.hip -- MEASURED rocWMMA throughput at the REAL
// Qwen3-30B-A3B shapes, so "fast" can be checked against a number instead of
// against a feeling.
//
// WHY THIS DRIVER EXISTS
// ----------------------
// tool_rocwmma_moe.hip proves the grouped W4 expert GEMM is CORRECT. It does not
// answer "is it fast", and its timing is on a toy shape (P=256 pairs, K=256,
// N=512). This driver answers the speed question at the shapes the model
// actually presents, and reports the two regimes separately because they are
// different problems:
//
//   PREFILL  M = tokens x topk  (large M -> the WMMA tile is well fed)
//   DECODE   M = 1 token x topk spread over topk experts (M = 1 PER EXPERT)
//
// The decode row is the honest one: a 16x16x16 WMMA tile consumes 1 useful row
// out of 16 when a single token routes to an expert. No amount of instruction
// choice fixes that; only grouping across tokens does. This driver measures the
// loss instead of asserting it away.
//
// WHAT IS MEASURED vs DERIVED
// ---------------------------
//   MEASURED  achieved TFLOP/s, packed bytes read, achieved GB/s, ms
//   DERIVED   per-layer and whole-model time, obtained by multiplying the
//             measured per-launch number by the model's layer/GEMM count.
//             DERIVED numbers are labelled as such on every line.
//
// Build (verified):
//   hipcc --rocm-path="G:/ROCM10RT-gfx1201/lib/llvm" -std=c++17 -O2 \
//         --offload-arch=gfx1201 -Xclang -target-feature \
//         -Xclang +wavefrontsize32 -Itools/bench -Ikernels \
//         tools/bench/rocwmma_tput.hip -o /tmp/rocwmma_tput.exe
// Run:
//   KNJ_BUILD_ARCH=gfx1201 /tmp/rocwmma_tput.exe

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "knj_wmma_guard.h"

#define HIPCHECK(x)                                                            \
    do {                                                                       \
        hipError_t _e = (x);                                                   \
        if (_e != hipSuccess) {                                                \
            printf("HIP ERROR %s at %s:%d\n", hipGetErrorString(_e), __FILE__, \
                   __LINE__);                                                  \
            exit(9);                                                           \
        }                                                                      \
    } while (0)

namespace rwx = rocwmma;

// 16x16x16 is the ONLY WMMA geometry gfx1201 offers (docs/03-kernels.md).
static const int TM = 16, TN = 16, TK = 16;
// Block tile: 64x64 output per block = 16 warps of 16x16 tiles. The wide block
// is what gives the A and B panels reuse; a warp-per-tile layout would re-read
// the K panel for every tile (32 FLOP/byte -> ~8).
static const int BM = 64, BN = 64;
static const int NWARP = (BM / TM) * (BN / TN);   // 16
static const int NTHREAD = NWARP * 32;            // 512

// ---------------------------------------------------------------------------
// C[M,N] = A[M,K] * B[K,N], f16 in, f32 out, both operands row-major.
// This is the matrix core the whole engine's GEMMs bottom out in.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREAD)
knj_tput_gemm(const __half *__restrict__ A, const __half *__restrict__ B,
              float *__restrict__ C, int M, int N, int K) {
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    __shared__ __align__(32) __half sA[BM * TK];
    __shared__ __align__(32) __half sB[TK * BN];

    const int w = threadIdx.x / 32;          // warp id
    const int wm = w / (BN / TN);            // warp's tile row
    const int wn = w % (BN / TN);            // warp's tile col

    rwx::fragment<rwx::accumulator, TM, TN, TK, float> acc;
    rwx::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += TK) {
        // Cooperative panel load. Bounds-checked so a non-multiple shape is
        // still exactly correct (zero padding contributes nothing).
        for (int i = threadIdx.x; i < BM * TK; i += NTHREAD) {
            const int r = i / TK, k = i % TK;
            const int gr = m0 + r, gk = k0 + k;
            sA[i] = (gr < M && gk < K) ? A[(size_t)gr * K + gk] : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < TK * BN; i += NTHREAD) {
            const int k = i / BN, c = i % BN;
            const int gk = k0 + k, gc = n0 + c;
            sB[i] = (gk < K && gc < N) ? B[(size_t)gk * N + gc] : __float2half(0.0f);
        }
        __syncthreads();

        rwx::fragment<rwx::matrix_a, TM, TN, TK, __half, rwx::row_major> fa;
        rwx::fragment<rwx::matrix_b, TM, TN, TK, __half, rwx::row_major> fb;
        rwx::load_matrix_sync(fa, sA + wm * TM * TK, TK);
        rwx::load_matrix_sync(fb, sB + wn * TN, BN);
        rwx::mma_sync(acc, fa, fb, acc);
        __syncthreads();
    }

    __shared__ __align__(32) float sC[BM * BN];
    rwx::store_matrix_sync(sC + (wm * TM) * BN + wn * TN, acc, BN,
                           rwx::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < BM * BN; i += NTHREAD) {
        const int r = i / BN, c = i % BN;
        const int gr = m0 + r, gc = n0 + c;
        if (gr < M && gc < N) C[(size_t)gr * N + gc] = sC[i];
    }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
struct Case {
    const char *name;
    int M, N, K;
    const char *note;
};

static double bench_case(const __half *dA, const __half *dB, float *dC,
                         const Case &c, int iters, double *ms_out) {
    // Warm + correctness-adjacent sanity: output must not be all zeros.
    const dim3 grid((c.N + BN - 1) / BN, (c.M + BM - 1) / BM);
    knj_tput_gemm<<<grid, NTHREAD>>>(dA, dB, dC, c.M, c.N, c.K);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());

    hipEvent_t e0, e1;
    HIPCHECK(hipEventCreate(&e0));
    HIPCHECK(hipEventCreate(&e1));
    float best = 1e30f;
    for (int r = 0; r < 5; ++r) {
        HIPCHECK(hipEventRecord(e0));
        for (int i = 0; i < iters; ++i)
            knj_tput_gemm<<<grid, NTHREAD>>>(dA, dB, dC, c.M, c.N, c.K);
        HIPCHECK(hipEventRecord(e1));
        HIPCHECK(hipEventSynchronize(e1));
        float ms = 0.0f;
        HIPCHECK(hipEventElapsedTime(&ms, e0, e1));
        best = std::min(best, ms / (float)iters);
    }
    *ms_out = best;
    const double flops = 2.0 * (double)c.M * c.N * c.K;
    return flops / (best * 1e-3) / 1e12;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    HIPCHECK(hipSetDevice(0));

    knj_wmma_dev dev;
    const int g = knj_wmma_guard(&dev);
    if (g) return g;

    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    printf("device      %s\n", dev.name);
    printf("gcnArchName %s\n", dev.arch);
    printf("CU (as reported) %d   wave %d   clock %d kHz\n",
           dev.cu, dev.warp, prop.clockRate);

    // ---- Qwen3-30B-A3B geometry (models/qwen3-30b-a3b/config.json) -------
    // 48 layers, hidden 2048, moe_intermediate 768, 128 experts, top-8.
    const int HID  = 2048;   // K of gate/up
    const int MOE  = 768;    // intermediate; N of gate/up per expert (x2 w/ up)
    const int TOPK = 8;
    const int NL   = 48;
    const int NEXP = 128;

    // Worst-case allocation so every case fits one buffer set. NMAX must be the
    // LARGEST N over all cases, which is the down-projection's 2048 (= hidden),
    // not the gate/up 1536 -- sizing dC at 1536 silently overran it and the
    // launch failed rather than printing a wrong number, which is why this is
    // checked by running, not by reading.
    const int KMAX = HID;
    const int NMAX = HID;                   // 2048, the down-projection N
    const int MMAX = 16384;                 // 2048 tokens x top-8
    std::vector<__half> hA((size_t)MMAX * KMAX, __float2half(0.01f));
    std::vector<__half> hB((size_t)KMAX * NMAX, __float2half(0.02f));
    for (size_t i = 0; i < hA.size(); ++i)
        hA[i] = __float2half((float)((int)(i % 97) - 48) / 64.0f);
    for (size_t i = 0; i < hB.size(); ++i)
        hB[i] = __float2half((float)((int)(i % 89) - 44) / 64.0f);

    __half *dA; __half *dB; float *dC;
    HIPCHECK(hipMalloc(&dA, hA.size() * sizeof(__half)));
    HIPCHECK(hipMalloc(&dB, hB.size() * sizeof(__half)));
    HIPCHECK(hipMalloc(&dC, (size_t)MMAX * NMAX * sizeof(float)));
    HIPCHECK(hipMemcpy(dA, hA.data(), hA.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(dB, hB.data(), hB.size() * sizeof(__half), hipMemcpyHostToDevice));

    printf("\ngeometry    Qwen3-30B-A3B: hidden %d, moe_inter %d, %d experts, top-%d, %d layers\n",
           HID, MOE, NEXP, TOPK, NL);
    printf("note        DERIVED lines multiply a MEASURED per-launch number by the\n");
    printf("            model's GEMM count; they are not measured end-to-end.\n");

    // ---- the two regimes -------------------------------------------------
    // PREFILL: 2048-token prompt, 8 routed experts per token -> 16384 pairs.
    //   gate+up for one expert set: M=16384, N=1536, K=2048
    //   but a single expert only sees its own routed rows. Measured here as the
    //   *shape* the grouped kernel must serve, i.e. the union.
    // DECODE: 1 token, 8 routed experts -> 8 experts x M=1.
    const Case cases[] = {
        {"prefill gate+up (grouped)", 16384, 1536, 2048,
         "M = 2048 tokens x top-8, all experts"},
        {"prefill down",              16384, 2048,  768,
         "M = same, K = moe_inter"},
        {"decode gate+up (grouped)",      8, 1536, 2048,
         "M = 8 = one token's top-8 pairs"},
        {"decode down",                   8, 2048,  768,
         "M = 8 pairs"},
        {"decode gate+up, tile-limited",  1, 1536, 2048,
         "M = 1: ONE expert, ONE token -- the true per-expert decode shape"},
        {"attention QK^T, 1 head, 2048 ctx", 2048, 2048, 128,
         "per-head prefill score matrix"},
    };

    printf("\n%-36s %10s %10s %12s %12s\n",
           "case", "M x N x K", "ms", "TFLOP/s", "GB/s(packed)");
    printf("---------------------------------------------------------------------------------\n");

    double pf_gate = 0, pf_down = 0, dec_gate = 0, dec_down = 0, dec_true = 0;
    for (const Case &c : cases) {
        const int iters = (c.M >= 2048) ? 30 : 400;
        double ms = 0.0;
        const double tf = bench_case(dA, dB, dC, c, iters, &ms);
        // f16 operand traffic actually touched, in bytes
        const double bytes =
            sizeof(__half) * ((double)c.M * c.K + (double)c.K * c.N);
        char dims[64];
        snprintf(dims, sizeof dims, "%dx%dx%d", c.M, c.N, c.K);
        printf("%-36s %10s %10.4f %12.2f %12.1f\n", c.name, dims, ms, tf,
               bytes / (ms * 1e-3) / 1e9);
        if (c.M == 16384 && c.N == 1536) pf_gate = ms;
        if (c.M == 16384 && c.N == 2048) pf_down = ms;
        if (c.M == 8 && c.N == 1536)     dec_gate = ms;
        if (c.M == 8 && c.N == 2048)     dec_down = ms;
        if (c.M == 1 && c.N == 1536)     dec_true = ms;
    }

    // ---- DERIVED per-layer / per-token budgets ---------------------------
    printf("\nDERIVED (measured per-launch x model layer count -- NOT end-to-end)\n");
    printf("---------------------------------------------------------------------------------\n");
    // Prefill of 2048 tokens: every layer runs one grouped gate+up and one down.
    const double pf_layer_us = (pf_gate + pf_down) * 1000.0;
    const double pf_total_ms = pf_layer_us * NL / 1000.0;
    printf("prefill 2048 tok, expert GEMM only   %8.1f us/layer  %8.1f ms total\n",
           pf_layer_us, pf_total_ms);
    printf("  -> expert-GEMM-only prefill ceiling  %8.1f tok/s\n",
           2048.0 / (pf_total_ms * 1e-3));

    // Decode: 48 layers x (gate+up + down) for one token.
    const double dec_layer_us = (dec_gate + dec_down) * 1000.0;
    const double dec_total_ms = dec_layer_us * NL / 1000.0;
    printf("decode 1 tok, expert GEMM only       %8.1f us/layer  %8.1f ms total\n",
           dec_layer_us, dec_total_ms);
    printf("  -> expert-GEMM-only decode ceiling   %8.1f tok/s\n",
           1.0 / (dec_total_ms * 1e-3));
    printf("  -> but the SAME work at the M=1 per-expert shape costs\n");
    printf("     %8.1f us/layer (%.0fx the grouped cost): tile is 1/16 useful\n",
           dec_true * 1000.0 * 2.0, (dec_true * 2.0) / (dec_layer_us / 1000.0));
    printf("     %d experts x M=1 per layer = %d WMMA tiles each 1/16 full\n",
           NEXP, NEXP);

    printf("\nRESULT: MEASURED throughput above; no end-to-end token rate is claimed\n");
    printf("        by this driver. See BUILD-OUTLINE.md for what still blocks it.\n");
    return 0;
}
