// tools/bench/rocwmma_moe.hip -- grouped MoE expert GEMM done with rocWMMA.
//
// WHAT THIS REPLACES
// ------------------
// docs/03-kernels.md specified the gfx1201 expert W4 path as "unpack + WMMA f16"
// and docs/02 C17 as "per-token expert launches are prohibited; group or die on
// load; waves when the active set exceeds the slot pool". Both are still the
// contract. What changed is HOW the WMMA part is written: instead of the raw
// `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` whose per-lane operand
// mapping was the blocking unknown of docs/00 section 7.4a, this driver uses
// rocWMMA fragments, and the library owns the lane mapping.
//
// SHAPE
// -----
//   P pairs = T tokens x topk experts, each pair p = (token t, expert e)
//   grouped by expert: expert e holds the list of pairs routed to it
//   Y[p][n] = sum_k X[pairTok[p]][k] * W[e][n][k]     (the gate/up half)
//
// The output is per PAIR, not accumulated into the residual: accumulation is
// the weighted expert sum that comes after this kernel, and keeping it out means
// two experts routing the same token cannot race on Y.
//
// THREE THINGS THIS DRIVER PROVES
// -------------------------------
//  1. A rocWMMA W4-g128 grouped expert GEMM is CORRECT against a host oracle on
//     every output element, seeded from the repo's own pack layout (knj_pack.h),
//     so the kernel and the packer cannot drift.
//  2. The always-compiled SIMT fallback (docs/02 C16/C17) agrees with the WMMA
//     path on the same inputs -- so the WMMA path is A/B-able, not assumed.
//  3. The packed bytes are consumed DIRECTLY (I2: no dequantise-to-f16 staging
//     on the miss path). The only f16 materialisation is the 16x16 tile that
//     `load_matrix_sync` requires, and it happens in shared memory after the
//     bytes have already moved.
//
// Build (verified):
//   hipcc --rocm-path="G:/ROCM10RT-gfx1201/lib/llvm" -std=c++17 -O2 \
//         --offload-arch=gfx1201 -Xclang -target-feature \
//         -Xclang +wavefrontsize32 -Itools/bench -Ikernels \
//         tools/bench/rocwmma_moe.hip -o /tmp/rocwmma_moe.exe
// Run:
//   KNJ_BUILD_ARCH=gfx1201 /tmp/rocwmma_moe.exe

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "knj_wmma_guard.h"
#include "knj_rocwmma.h"
#include "knj_pack.h"

#define HIPCHECK(x)                                                            \
    do {                                                                       \
        hipError_t _e = (x);                                                   \
        if (_e != hipSuccess) {                                                \
            printf("HIP ERROR %s at %s:%d\n", hipGetErrorString(_e), __FILE__, \
                   __LINE__);                                                  \
            exit(9);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Problem shape
// ---------------------------------------------------------------------------
// Small enough to run and fully verify in seconds, but the structural
// properties are the real model's: hidden 2048 is the Qwen3-30B-A3B number, the
// routing is 8-of-16, and K/N are a scaled-down but not trivial K loop
// (K/16 = 16 mma steps, N/16 = 32 n-tiles).
static const int T_TOKENS  = 32;   // tokens in the batch
static const int TOPK      = 8;    // experts per token (Qwen3-30B-A3B is top-8)
static const int E_EXPERTS = 16;   // distinct experts in the active set
static const int K_DIM     = 256;  // reduction (hidden) -- 2 groups of 128
static const int N_DIM     = 512;  // output rows (gate+up for the tile)
static const int P_PAIRS   = T_TOKENS * TOPK;

static inline uint32_t rng_next(uint32_t &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

// ---------------------------------------------------------------------------
// Device: rocWMMA path
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(64)
knj_rw_w4_gemm(const __half *__restrict__ X,        // [T][K]
               const uint8_t *__restrict__ Wq,      // [E][N][gpr][68]
               const int *__restrict__ pairIdx,     // [E][maxM]
               const int *__restrict__ pairCnt,     // [E]
               const int *__restrict__ pairTok,     // [P]
               float *__restrict__ Y,               // [P][N]
               int N, int K, int idxStride) {
    namespace rw  = knjrw;
    namespace rwx = rocwmma;
    const int e  = blockIdx.z;
    const int m0 = blockIdx.x * rw::M;
    const int n0 = blockIdx.y * rw::N;
    const int m  = pairCnt[e];
    if (m0 >= m) return;

    const int gpr = K / rw::GROUP;                        // groups per weight row
    const uint8_t *wblob = Wq + (uint64_t)e * N * gpr * rw::GROUP_STRIDE;

    __shared__ __align__(32) __half sA[rw::M * rw::K];
    __shared__ __align__(32) __half sB[rw::K * rw::N];
    __shared__ __align__(32) float  sD[rw::M * rw::N];

    rwx::fragment<rwx::accumulator, rw::M, rw::N, rw::K, float> acc;
    rwx::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += rw::K) {
        // --- A tile: gather the routed token rows -------------------------
        for (int i = threadIdx.x; i < rw::M * rw::K; i += blockDim.x) {
            const int mm = i / rw::K, kk = i % rw::K;
            const int p  = (m0 + mm < m) ? pairIdx[e * idxStride + m0 + mm] : -1;
            sA[i] = (p >= 0) ? X[(uint64_t)pairTok[p] * K + k0 + kk]
                             : __float2half(0.0f);
        }
        // --- B tile: unpack nibbles straight out of the packed blob -------
        for (int i = threadIdx.x; i < rw::K * rw::N; i += blockDim.x) {
            const int nn = i / rw::K, kk = i % rw::K;   // col_major: (n,k) at n*ld+k
            const int k  = k0 + kk;
            const int g  = k / rw::GROUP;
            const uint8_t *row = wblob + (uint64_t)(n0 + nn) * gpr * rw::GROUP_STRIDE;
            const uint8_t *grp = row + (uint64_t)g * rw::GROUP_STRIDE;
            __half scale;
            uint32_t zp;
            rw::group_meta(row, g, &scale, &zp);
            sB[i] = rw::code_to_half(rw::nibble_at(grp, k % rw::GROUP), scale, zp);
        }
        __syncthreads();

        rwx::fragment<rwx::matrix_a, rw::M, rw::N, rw::K, __half, rwx::row_major> fa;
        rwx::fragment<rwx::matrix_b, rw::M, rw::N, rw::K, __half, rwx::col_major> fb;
        rwx::load_matrix_sync(fa, sA, rw::K);
        rwx::load_matrix_sync(fb, sB, rw::K);
        rwx::mma_sync(acc, fa, fb, acc);
        __syncthreads();
    }

    rwx::store_matrix_sync(sD, acc, rw::N, rwx::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < rw::M * rw::N; i += blockDim.x) {
        const int mm = i / rw::N, nn = i % rw::N;
        if (m0 + mm < m) {
            const int p = pairIdx[e * idxStride + m0 + mm];
            Y[(uint64_t)p * N + n0 + nn] = sD[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Device: SIMT fallback (always compiled -- docs/02 C16/C17)
// ---------------------------------------------------------------------------
// One thread per output element, no WMMA. This is the path that must exist so a
// future card without matrix units degrades rather than fails, and so the WMMA
// path can be A/B'd instead of trusted.
//
// `stride` is the grid stride, passed in from the host as gridX * blockX.
// It cannot be gridDim.x: gfx1201 lowers gridDim through __ockl_get_num_groups,
// which lives in the device runtime that -nogpulib skips, so a kernel that
// reads it fails to link with `undefined hidden symbol` -- not at runtime, at
// LINK time, which is how it was caught. The host already knows the launch
// shape; the device does not need to be told twice.
__global__ void knj_simt_w4_gemm(const __half *__restrict__ X,
                                 const uint8_t *__restrict__ Wq,
                                 const int *__restrict__ pairIdx,
                                 const int *__restrict__ pairCnt,
                                 const int *__restrict__ pairTok,
                                 float *__restrict__ Y,
                                 int N, int K, int idxStride, int stride) {
    const int e = blockIdx.z;
    const int m = pairCnt[e];
    const int total = m * N;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        const int mm = i / N, nn = i % N;
        const int pi = pairIdx[e * idxStride + mm];
        const __half *x = X + (uint64_t)pairTok[pi] * K;
        const int gpr = K / knjrw::GROUP;
        const uint8_t *row =
            Wq + ((uint64_t)e * N + nn) * gpr * knjrw::GROUP_STRIDE;
        float acc = 0.0f;
        for (int g = 0; g < gpr; ++g) {
            const uint8_t *grp = row + (uint64_t)g * knjrw::GROUP_STRIDE;
            __half scale;
            uint32_t zp;
            knjrw::group_meta(row, g, &scale, &zp);
            const float s = __half2float(scale);
            const int z   = (int)zp;
            const int kstart = g * knjrw::GROUP;
            for (int kk = 0; kk < knjrw::GROUP; kk += 8) {
                // `grp` is the GROUP BASE: the nibble index is group-local (kk),
                // the activation index is absolute (kstart + kk). Using the
                // absolute index on the packed payload walks off the end of every
                // group after the first and silently reads the next column's
                // metadata as weights.
                float w[8];
                for (int i = 0; i < 8; ++i)
                    w[i] = (float)((int)knjrw::nibble_at(grp, kk + i) - z) * s;
                for (int i = 0; i < 8; ++i)
                    acc = fmaf(__half2float(x[kstart + kk + i]), w[i], acc);
            }
        }
        Y[(uint64_t)pi * N + nn] = acc;
    }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
static const float SENTINEL = -12345.0f;  // proves every output was written

struct Verdict {
    double worst_row_rel = 0.0;
    double worst_abs     = 0.0;
    int    unwritten     = 0;
    int    mismatches    = 0;
    int    checked       = 0;
};

static Verdict check(const std::vector<float> &got, const std::vector<double> &ref,
                     int P, int N, double tol) {
    Verdict v;
    for (int p = 0; p < P; ++p) {
        double rowmax = 0.0;
        for (int n = 0; n < N; ++n) rowmax = std::max(rowmax, std::fabs(ref[(size_t)p * N + n]));
        double worst = 0.0;
        for (int n = 0; n < N; ++n) {
            const float g = got[(size_t)p * N + n];
            if (g == SENTINEL) { v.unwritten++; continue; }
            const double d = std::fabs((double)g - ref[(size_t)p * N + n]);
            v.worst_abs = std::max(v.worst_abs, d);
            worst = std::max(worst, d / (rowmax > 0 ? rowmax : 1.0));
            v.checked++;
        }
        if (worst > tol) v.mismatches++;
        v.worst_row_rel = std::max(v.worst_row_rel, worst);
    }
    return v;
}

int main() {
    knj_wmma_dev dev;
    const int g = knj_wmma_guard(&dev);
    if (g) return g;
    printf("device      %s\n", dev.name);
    printf("gcnArchName %s\n", dev.arch);
    printf("cu          %d   wave %d\n", dev.cu, dev.warp);

    const int gpr   = K_DIM / knjrw::GROUP;       // 2
    const int maxM  = T_TOKENS;                   // worst case: all tokens -> one expert
    const int idxStride = 4096;                   // fixed, shared by both kernels

    std::vector<__half> X((size_t)T_TOKENS * K_DIM);
    std::vector<uint8_t> Wq((size_t)E_EXPERTS * N_DIM * gpr * knjrw::GROUP_STRIDE);
    std::vector<int> pairIdx((size_t)E_EXPERTS * idxStride, -1);
    std::vector<int> pairCnt(E_EXPERTS, 0);
    std::vector<int> pairTok(P_PAIRS);
    std::vector<int> pairExp(P_PAIRS);

    uint32_t s = 0x1234567u;
    for (auto &x : X) x = __float2half((float)((int)(rng_next(s) % 2001) - 1000) / 1000.0f);

    // Pack W4 g128 with the repo's own packer convention, writing the metadata
    // INTO the blob exactly the way knj_pack.h describes it.
    for (int e = 0; e < E_EXPERTS; ++e) {
        for (int n = 0; n < N_DIM; ++n) {
            uint8_t *row = Wq.data() + ((size_t)e * N_DIM + n) * gpr * knjrw::GROUP_STRIDE;
            for (int gg = 0; gg < gpr; ++gg) {
                uint8_t *grp = row + (size_t)gg * knjrw::GROUP_STRIDE;
                std::vector<uint8_t> codes(knjrw::GROUP);
                for (int k = 0; k < knjrw::GROUP; ++k)
                    codes[k] = (uint8_t)(rng_next(s) % 16);
                knj_pack_group_u4(grp, codes.data(), knjrw::GROUP);
                const __half scale = __float2half(0.004f + (float)(rng_next(s) % 100) / 10000.0f);
                const uint32_t zp  = rng_next(s) % 16;
                memcpy(grp + (knjrw::GROUP * 4 / 8), &scale, sizeof(scale));
                grp[knjrw::GROUP * 4 / 8 + 2] = (uint8_t)zp;
            }
        }
    }

    // Routing: deterministic, deliberately uneven (some experts cold, some hot)
    // so the wave/tile padding paths are exercised.
    for (int t = 0; t < T_TOKENS; ++t) {
        bool taken[E_EXPERTS] = {false};
        for (int i = 0; i < TOPK; ++i) {
            int e;
            do { e = (int)(rng_next(s) % E_EXPERTS); } while (taken[e]);
            taken[e] = true;
            const int p = t * TOPK + i;
            pairTok[p] = t;
            pairExp[p] = e;
            pairIdx[(size_t)e * idxStride + pairCnt[e]++] = p;
        }
    }

    // Host oracle: dequantise from the SAME blob bytes using the SAME formula,
    // accumulate in double over every k. No shortcut, no sampling.
    std::vector<double> ref((size_t)P_PAIRS * N_DIM, 0.0);
    std::vector<double> wdeq((size_t)N_DIM * K_DIM);
    for (int p = 0; p < P_PAIRS; ++p) {
        const int e = pairExp[p], t = pairTok[p];
        for (int n = 0; n < N_DIM; ++n) {
            const uint8_t *row = Wq.data() + ((size_t)e * N_DIM + n) * gpr * knjrw::GROUP_STRIDE;
            for (int gg = 0; gg < gpr; ++gg) {
                const uint8_t *grp = row + (size_t)gg * knjrw::GROUP_STRIDE;
                __half scale;
                uint32_t zp;
                knjrw::group_meta(row, gg, &scale, &zp);
                const double sc = (double)__half2float(scale);
                for (int k = 0; k < knjrw::GROUP; ++k) {
                    const int kk = gg * knjrw::GROUP + k;
                    // same code->half rounding the device does, then to double
                    const double w = (double)__half2float(
                        knjrw::code_to_half(knjrw::nibble_at(grp, k), scale, zp));
                    wdeq[(size_t)n * K_DIM + kk] = w;
                    (void)sc;
                }
            }
        }
        for (int n = 0; n < N_DIM; ++n) {
            double a = 0.0;
            for (int k = 0; k < K_DIM; ++k)
                a += (double)__half2float(X[(size_t)t * K_DIM + k]) * wdeq[(size_t)n * K_DIM + k];
            ref[(size_t)p * N_DIM + n] = a;
        }
    }

    // --- device buffers ---------------------------------------------------
    __half *dX; uint8_t *dW; int *dIdx, *dCnt, *dTok; float *dY;
    HIPCHECK(hipMalloc(&dX, X.size() * sizeof(__half)));
    HIPCHECK(hipMalloc(&dW, Wq.size()));
    HIPCHECK(hipMalloc(&dIdx, pairIdx.size() * sizeof(int)));
    HIPCHECK(hipMalloc(&dCnt, pairCnt.size() * sizeof(int)));
    HIPCHECK(hipMalloc(&dTok, pairTok.size() * sizeof(int)));
    HIPCHECK(hipMalloc(&dY, (size_t)P_PAIRS * N_DIM * sizeof(float)));
    HIPCHECK(hipMemcpy(dX, X.data(), X.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(dW, Wq.data(), Wq.size(), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(dIdx, pairIdx.data(), pairIdx.size() * sizeof(int), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(dCnt, pairCnt.data(), pairCnt.size() * sizeof(int), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(dTok, pairTok.data(), pairTok.size() * sizeof(int), hipMemcpyHostToDevice));

    const dim3 grid((maxM + knjrw::M - 1) / knjrw::M, N_DIM / knjrw::N, E_EXPERTS);
    // One block group per expert (blockIdx.z), sweeping that expert's rows x cols.
    const dim3 simtGrid((maxM * N_DIM + 255) / 256, 1, E_EXPERTS);

    std::vector<float> yWmma((size_t)P_PAIRS * N_DIM, SENTINEL);
    std::vector<float> ySimt((size_t)P_PAIRS * N_DIM, SENTINEL);

    const double tol = 2e-3;  // row-relative

    // --- rocWMMA path -----------------------------------------------------
    HIPCHECK(hipMemset(dY, 0, (size_t)P_PAIRS * N_DIM * sizeof(float)));
    for (int i = 0; i < P_PAIRS * N_DIM; ++i) yWmma[i] = SENTINEL;  // sentinel prefill host-side
    {
        std::vector<float> sent((size_t)P_PAIRS * N_DIM, SENTINEL);
        HIPCHECK(hipMemcpy(dY, sent.data(), sent.size() * sizeof(float), hipMemcpyHostToDevice));
    }
    knj_rw_w4_gemm<<<grid, 64>>>(dX, dW, dIdx, dCnt, dTok, dY, N_DIM, K_DIM, idxStride);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(yWmma.data(), dY, yWmma.size() * sizeof(float), hipMemcpyDeviceToHost));

    // --- SIMT fallback ----------------------------------------------------
    {
        std::vector<float> sent((size_t)P_PAIRS * N_DIM, SENTINEL);
        HIPCHECK(hipMemcpy(dY, sent.data(), sent.size() * sizeof(float), hipMemcpyHostToDevice));
    }
    knj_simt_w4_gemm<<<simtGrid, 256>>>(dX, dW, dIdx, dCnt, dTok, dY, N_DIM, K_DIM,
                                        idxStride, simtGrid.x * 256);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(ySimt.data(), dY, ySimt.size() * sizeof(float), hipMemcpyDeviceToHost));

    const Verdict vw = check(yWmma, ref, P_PAIRS, N_DIM, tol);
    const Verdict vs = check(ySimt, ref, P_PAIRS, N_DIM, tol);

    // WMMA vs SIMT agreement on the same packed bytes.
    double worst_x = 0.0;
    int    diff_x  = 0;
    for (int p = 0; p < P_PAIRS; ++p) {
        double rm = 0.0;
        for (int n = 0; n < N_DIM; ++n) rm = std::max(rm, std::fabs((double)ref[(size_t)p * N_DIM + n]));
        for (int n = 0; n < N_DIM; ++n) {
            const double d = std::fabs((double)yWmma[(size_t)p * N_DIM + n] -
                                       (double)ySimt[(size_t)p * N_DIM + n]);
            if (d / (rm > 0 ? rm : 1.0) > tol) diff_x++;
            worst_x = std::max(worst_x, d / (rm > 0 ? rm : 1.0));
        }
    }

    // --- timing (launch-inclusive, best of N) -----------------------------
    const int ITERS = 200;
    hipEvent_t ev0, ev1;
    HIPCHECK(hipEventCreate(&ev0));
    HIPCHECK(hipEventCreate(&ev1));
    knj_rw_w4_gemm<<<grid, 64>>>(dX, dW, dIdx, dCnt, dTok, dY, N_DIM, K_DIM, idxStride);  // warm
    HIPCHECK(hipDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < 5; ++i) {
        HIPCHECK(hipEventRecord(ev0));
        for (int j = 0; j < ITERS; ++j)
            knj_rw_w4_gemm<<<grid, 64>>>(dX, dW, dIdx, dCnt, dTok, dY, N_DIM, K_DIM, idxStride);
        HIPCHECK(hipEventRecord(ev1));
        HIPCHECK(hipEventSynchronize(ev1));
        float ms = 0.0f;
        HIPCHECK(hipEventElapsedTime(&ms, ev0, ev1));
        best = std::min(best, ms / ITERS);
    }
    const double flops = 2.0 * (double)P_PAIRS * N_DIM * K_DIM;
    const double tflops = flops / ((double)best * 1e-3) / 1e12;

    // --- report -----------------------------------------------------------
    printf("\n-- grouped MoE expert GEMM, rocWMMA (gfx1201) --\n");
    printf("shape       P=%d pairs, E=%d experts, K=%d, N=%d, W4 g%d\n",
           P_PAIRS, E_EXPERTS, K_DIM, N_DIM, knjrw::GROUP);
    printf("packed      %zu bytes (%.4f B/weight incl. 3 B/group metadata)\n",
           Wq.size(), (double)Wq.size() / ((double)E_EXPERTS * N_DIM * K_DIM));
    printf("\nWMMA path\n");
    printf("  worst row-relative err  %.3e   (tol %.0e)\n", vw.worst_row_rel, tol);
    printf("  worst absolute err      %.3e\n", vw.worst_abs);
    printf("  outputs checked         %d / %d\n", vw.checked, P_PAIRS * N_DIM);
    printf("  unwritten (sentinel)    %d\n", vw.unwritten);
    printf("  mismatching rows        %d\n", vw.mismatches);
    printf("\nSIMT fallback (always compiled)\n");
    printf("  worst row-relative err  %.3e\n", vs.worst_row_rel);
    printf("  outputs checked         %d / %d\n", vs.checked, P_PAIRS * N_DIM);
    printf("  unwritten (sentinel)    %d\n", vs.unwritten);
    printf("  mismatching rows        %d\n", vs.mismatches);
    printf("\nWMMA vs SIMT\n");
    printf("  worst row-relative diff %.3e, %d cells over tol\n", worst_x, diff_x);
    printf("\ntiming (MEASURED, launch-inclusive, best of 5 x %d)\n", ITERS);
    printf("  %.4f ms   %.2f TFLOP/s effective\n", best, tflops);

    const bool pass = (vw.unwritten == 0 && vw.checked == P_PAIRS * N_DIM &&
                       vw.worst_row_rel < tol && vs.unwritten == 0 &&
                       vs.worst_row_rel < tol && diff_x == 0);
    printf("\nRESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 5;
}
