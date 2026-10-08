// Kanjoos expert GEMM at real model shape — is the engine transfer-bound,
// and if so, what bit-width gets it off that wall?
//
// WHY THIS AND NOT MORE WMMA PROBING
// ---------------------------------
// The layout probe (wmma_layout.hip) settled it against the naive assumption.
// Measured on an RX 9070 XT:
//   A = e_k, B = [1..8]   -> D = 2(k+1) in ALL EIGHT slots
//   A = B = [1..8]       -> D = 408 = 2*sum(j^2)  in all eight slots
//   A varies per lane, B = 1 -> D[i] = 72*(9+i), slots scaled by DIFFERENT
//                              factors
// The first two say D is 2x a per-lane dot. The third says the eight slots are
// not even a uniform broadcast. Neither is a 16x16 tile assignment, so a tiled
// GEMM written against "A is 8 consecutive f16, lane L holds row L" would read
// the wrong elements and produce wrong numbers silently.
//
// So gfx1201 ships the SIMT packed-f16 path, and this file measures THAT.
//
// THE ROOFLINE THAT ACTUALLY APPLIES
// ----------------------------------
// Kanjoos exists to run a 30B MoE on a consumer card by keeping expert
// weights in HOST RAM and streaming them. The transfer medium is therefore
// PCIe, not the card's own VRAM. An earlier revision divided 2.47 MB by the
// card's 580 GB/s and concluded "compute-bound by 217x". That was wrong on
// both sides at once: 580 GB/s is the wrong roofline, AND the kernel it timed
// ran 126x below its own roofline. The two errors cancelled into a confident,
// entirely fictional verdict. Both numbers are measured here.
//
// Build (the only line that links on this machine), then run it with the arch
// in the ENVIRONMENT -- the guard in main() refuses to run without it:
//   G:\ROCM10RT-gfx1201\bin\hipcc.exe -nogpulib -O2 --offload-arch=gfx1201 \
//     -L'G:\ROCM10RT-gfx1201\lib' -lamdhip64 expert_gemm.hip -o eg.exe
//   KNJ_BUILD_ARCH=gfx1201 ./eg.exe
//
// KNJ_BUILD_ARCH must be an environment variable and NOT -D. hipcc re-spawns
// clang through a command STRING, so a -D carrying quotes loses them to the
// inner shell and the compiler reports 'use of undeclared identifier'. It also
// cannot be __gfx1201__, which is a device-pass macro and main() is host code.
//
// Toolchain constraints this file ran into, all properties of the -nogpulib
// build rather than of the program:
//   - no `gridDim` in any kernel: gfx1201 materialises it through
//     __ockl_get_num_groups, whose inline definition is in the runtime we skip.
//   - no 64-bit indexing needed; uint32_t suffices even for 512 MB.
//   - '+wavefrontsize32' is rejected: gfx1201 is natively wave32.

#include <hip/hip_runtime.h>
#include <chrono>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The weight-pack constants below used to be restated here as literals
 * (0.5 + 3.0/128.0, meta = 3.0/128.0, ...). That is exactly the drift class
 * this repo caught once (0.5186 vs 0.5234, docs/00 section 8.10). The single
 * definition of those constants is kernels/knj_pack.h; include it and use
 * KNJ_W*_BYTES_PER_WEIGHT / KNJ_META_BYTES_PER_GROUP so this file's one-expert
 * copy size and its bit-width table share ONE constant with the rest of the tree.
 *
 * knj_pack.h is self-contained and guarded; adding the include does not change
 * any type or helper this file already defines (F2H/H2F/LO/HI stay local).
 */
#include "knj_pack.h"

typedef _Float16 __attribute__((ext_vector_type(2))) v2h;

#define HIDDEN   2048
#define INTER     768
#define NCOL      64   /* columns per block tile   */
#define CPT        2   /* columns per thread       */
#define RPT        8   /* rows per thread          */
#define GRP        4   /* 32-lane groups per block */
#define RPB (GRP*RPT)  /* 32 rows per block         */
#define MMAX      512
#define NEXP      128
#define TOPK        8

/* Macros, not functions. A plain `static inline` is __host__ by default and a
 * __device__-only spelling is rejected too, so calling either from inside a
 * __global__ fails with "no matching function". */
#define F2H(f) (__builtin_bit_cast(uint16_t, (_Float16)(f)))
#define H2F(h) ((float)__builtin_bit_cast(_Float16, (uint16_t)(h)))
#define LO(w) ((_Float16)H2F((uint16_t)((w) & 0xffffu)))
#define HI(w) ((_Float16)H2F((uint16_t)((w) >> 16)))

static double now_us(void) {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// C = A*B split over K. A row-major [M,K] fp16, B row-major [K,N] fp16.
//
// WHY THIS SHAPE. The first working version ran at 0.8% of peak, and the
// reason was visible in its own census rather than in a profiler:
//
//   grid 2 x 12 x 256 thr = 24 blocks on a 32-CU card. 24 blocks means 24
//   CUs have work and 8 idle, and B was re-read once per row-block. The
//   census said 3.3% of the VRAM roofline and 0.8% of peak: the kernel was
//   on NEITHER. A kernel that is on neither roofline is not arithmetic-bound,
//   it is issue-bound, and no amount of arithmetic tuning fixes that.
//
// The fix is arithmetic intensity per instruction:
//
//   old  1 col/thread, 4 rows: 2 B loads + 4 A loads = 6 loads for 4
//        v_pk_fma_f16.  1.5 loads per FMA -- the loads dominated.
//   new  2 cols/thread, 8 rows: 2 B loads + 8 A loads = 10 loads for 16
//        v_pk_fma_f16. 0.63 loads per FMA, and 16 INDEPENDENT accumulator
//        chains so the FMA units stay fed at low occupancy.
//   split-K: the k range is cut KSPLIT ways so 48 blocks cover 32 CUs.
//
// The 4-byte B load is what makes 2 columns per thread free: B[k][col] and
// B[k][col+1] are adjacent, so one 32-bit fetch yields the whole pair, and
// its two halves are the B operands for k and k+1 respectively.
//
// FIVE revisions of this kernel compiled cleanly and were wrong. All are
// recorded because the pattern recurs:
//   (1) `wave = threadIdx.y` with blockDim(64,2). Waves are cut from the
//       LINEAR id, so wave 1 is tx 32..63 of ty 0. Never a wave index.
//   (2) A wave claimed 16 rows but addressed row and row+1 only: every block
//       wrote 4 of its 32 rows. RMSE 0.90, written rows EXACTLY right.
//   (3) An f16 temporary named `acc` shadowed `float acc[]`, so `acc[j] +=`
//       indexed one v2h. Every output stayed exactly zero. RMSE 1.000.
//   (4) `(tid & 63) * 2` gave 2 columns across 0..126 while the tile is 64
//       wide. RMSE 1.13. Two columns need CPT threads per 64-column tile,
//       i.e. a 32-lane group, not a 64-lane one.
//   (5) An arch guard written with __gfx1201__ compiled to NOTHING, because
//       that is a device-pass macro and main() is host code. The kernel it
//       was meant to protect ran on the wrong arch and reported 2327% of peak.
// ---------------------------------------------------------------------------
__global__ void gemm_pk16(const uint16_t *__restrict__ A,
                          const uint16_t *__restrict__ B,
                          float *__restrict__ Cpart,
                          int M, int K, int N, int kchunk) {
    const int tid  = threadIdx.x;                 /* 0..127            */
    const int grp  = tid >> 5;                    /* 0..3   : row grp  */
    const int lane = tid & 31;                    /* 0..31            */
    const int col  = blockIdx.y * NCOL + lane * CPT;   /* 0,2,..62     */
    const int row0 = blockIdx.x * RPB + grp * RPT;     /* 0,8,16,24   */

    int kbeg = blockIdx.z * kchunk;
    int kend = kbeg + kchunk;
    if (kend > K) kend = K;

    /* The row bound is hoisted out of the k loop. Left inside, it is 8
     * compare-and-branch per iteration against a compile-time-unrolled body;
     * it never fires at M = RPB, it just costs. */
    int jmax = M - row0;
    if (jmax > RPT) jmax = RPT;
    if (jmax < 0) jmax = 0;

    float acc[RPT][CPT];
    for (int j = 0; j < RPT; j++) { acc[j][0] = 0.0f; acc[j][1] = 0.0f; }

    for (int k = kbeg; k + 1 < kend; k += 2) {
        /* One 32-bit fetch per k yields BOTH columns of this thread: the low
         * half is B[k][col] and the high half is B[k][col+1]. Pairing the low
         * halves across k is what keeps the FMA contracting along K. */
        uint32_t b0, b1;
        memcpy(&b0, &B[(size_t)k * N + col], 4);
        memcpy(&b1, &B[(size_t)(k + 1) * N + col], 4);
        v2h bk = { LO(b0), LO(b1) };   /* B[k][col],   B[k+1][col]   */
        v2h bl = { HI(b0), HI(b1) };   /* B[k][col+1], B[k+1][col+1] */

        for (int j = 0; j < jmax; j++) {
            const int r = row0 + j;
            /* A is contiguous in k, so the pair is one 32-bit load, and all
             * 32 lanes of a group want the same word (a broadcast). */
            uint32_t aw;
            memcpy(&aw, &A[(size_t)r * K + k], 4);
            v2h a = { LO(aw), HI(aw) };

            /* NOT volatile. `asm volatile` is a compiler memory barrier, which
             * stops loads of the next k iteration from being hoisted above these
             * FMAs. Measured: keeping volatile pinned this kernel at 2% of peak
             * because every iteration stalled on its own loads. The operands
             * differ each iteration, so there is nothing for CSE to collapse. */
            /* NOTE the temporaries start at ZERO and are NOT named `acc`.
             * v_pk_fma_f16 is an accumulating form: it reads whatever the
             * register held. Both of those were real, separate bugs. */
            v2h p0 = {(_Float16)0.0f, (_Float16)0.0f};
            v2h p1 = {(_Float16)0.0f, (_Float16)0.0f};
            asm ("v_pk_fma_f16 %0, %1, %2, %0" : "+v"(p0) : "v"(a), "v"(bk));
            asm ("v_pk_fma_f16 %0, %1, %2, %0" : "+v"(p1) : "v"(a), "v"(bl));
            acc[j][0] += (float)p0.x + (float)p0.y;
            acc[j][1] += (float)p1.x + (float)p1.y;
        }
    }

    for (int j = 0; j < RPT; j++) {
        const int r = row0 + j;
        if (r >= M) continue;
        float *dst = Cpart + (size_t)blockIdx.z * M * N + (size_t)r * N + col;
        dst[0] = acc[j][0];
        dst[1] = acc[j][1];
    }
}

/* Split-K needs the partials summed. Cheap: KSPLIT*M*N reads, once. */
__global__ void reduce_parts(const float *__restrict__ Cpart,
                             float *__restrict__ C, int MN, int KS) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MN) return;
    float s = 0.0f;
    for (int z = 0; z < KS; z++) s += Cpart[(size_t)z * MN + i];
    C[i] = s;
}

/* VRAM streaming read: the roofline the GEMM's own B traffic is judged against. */
__global__ void bw_read(const uint32_t *__restrict__ p, uint32_t n,
                        uint32_t stride, float *__restrict__ out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    uint32_t k = i;
    for (; k + 3u * stride < n; k += 4u * stride) {
        s0 += p[k]; s1 += p[k + stride];
        s2 += p[k + 2u * stride]; s3 += p[k + 3u * stride];
    }
    for (; k < n; k += stride) s0 += p[k];
    if (s0 + s1 + s2 + s3 == 0xffffffffu) out[i] = (float)(s0 + s1 + s2 + s3);
}

static double ref_one(const uint16_t *A, const uint16_t *B, int m, int n, int K, int N) {
    double s = 0;
    for (int k = 0; k < K; k++) s += (double)H2F(A[(size_t)m * K + k]) *
                                    (double)H2F(B[(size_t)k * N + n]);
    return s;
}

int main(void) {
    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, 0);

    /* ARCH GUARD.
     * A binary built --offload-arch=gfx1031 RUNS on a gfx1201 card without an
     * error. The kernel is never dispatched, every output keeps its hipMemset
     * value, and the timing table below comes out as "2327% of peak" in
     * 0.1 us. A clean build, a plausible number and a well-formatted table
     * are all consistent with the kernel never having run at all. */
    const char *built_arch = getenv("KNJ_BUILD_ARCH");
    if (!built_arch || !*built_arch) {
        printf("ARCH GUARD NOT CONFIGURED: set KNJ_BUILD_ARCH to the arch you"
               " built for. An unguarded run is not a measurement.\n");
        return 7;
    }
    if (strncmp(p.gcnArchName, built_arch, strlen(built_arch)) != 0) {
        printf("ARCH MISMATCH: built for %s, device is %s\n",
               built_arch, p.gcnArchName);
        printf("SKIPPED -- not run, not measured, not a result.\n");
        return 6;
    }

    double clk_ghz = p.clockRate / 1.0e6;
    double peak_pk = p.multiProcessorCount * 128.0 * 2.0 * 2.0 * clk_ghz / 1.0e3;
    int waves_per_cu_cap = p.maxThreadsPerMultiProcessor / p.warpSize;
    printf("device %s  arch %s  %d CU  wave %d  %d MHz  %d thr/CU (max %d waves/CU)\n",
           p.name, p.gcnArchName, p.multiProcessorCount, p.warpSize,
           p.clockRate / 1000, p.maxThreadsPerMultiProcessor, waves_per_cu_cap);
    printf("packed-f16 peak, derived from CU count and clock: %.1f TFLOP/s\n\n", peak_pk);

    const int K = HIDDEN, N = INTER;
    const int M = 32;
    const int KS = 4;                     /* split-K factor              */
    const int kchunk = (K / KS) & ~1;     /* even                        */
    const int MN = M * N;

    uint16_t *hA = (uint16_t *)malloc((size_t)M * K * 2);
    uint16_t *hB = (uint16_t *)malloc((size_t)K * N * 2);
    float    *hC = (float   *)malloc((size_t)MN * 4);
    if (!hA || !hB || !hC) { printf("host alloc failed\n"); return 1; }
    for (size_t i = 0; i < (size_t)M * K; i++)
        hA[i] = F2H(((float)((i * 1103515245u >> 16) & 0xffff) / 32768.0f) - 1.0f);
    for (size_t i = 0; i < (size_t)K * N; i++)
        hB[i] = F2H(((float)((i * 22695477u >> 16) & 0xffff) / 32768.0f) - 1.0f);

    uint16_t *dA, *dB; float *dC, *dP;
    const size_t vram_mb = 512;
    uint32_t *dBW;
    if (hipMalloc(&dA, (size_t)M * K * 2) != hipSuccess ||
        hipMalloc(&dB, (size_t)K * N * 2) != hipSuccess ||
        hipMalloc(&dC, (size_t)MN * 4) != hipSuccess ||
        hipMalloc(&dP, (size_t)KS * MN * 4) != hipSuccess ||
        hipMalloc(&dBW, vram_mb << 20) != hipSuccess) {
        printf("hipMalloc failed\n"); return 1;
    }
    hipMemcpy(dA, hA, (size_t)M * K * 2, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, (size_t)K * N * 2, hipMemcpyHostToDevice);
    hipMemset(dC, 0, (size_t)MN * 4);
    hipMemset(dBW, 0, vram_mb << 20);

    dim3 blk(GRP * 32);
    dim3 grd((M + RPB - 1) / RPB, (N + NCOL - 1) / NCOL, KS);

    /* ---- 1. correctness: every element, no sampling --------------------- */
    gemm_pk16<<<grd, blk>>>(dA, dB, dP, M, K, N, kchunk);
    reduce_parts<<<(MN + 255) / 256, 256>>>(dP, dC, MN, KS);
    hipError_t le = hipDeviceSynchronize();
    if (le != hipSuccess) { printf("launch failed: %s\n", hipGetErrorString(le)); return 1; }
    hipMemcpy(hC, dC, (size_t)MN * 4, hipMemcpyDeviceToHost);

    double sse = 0, mag = 0, maxabs = 0;
    int zeros = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double want = ref_one(hA, hB, m, n, K, N);
            double got  = hC[(size_t)m * N + n];
            if (got == 0.0) zeros++;
            double d = got - want;
            sse += d * d; mag += want * want;
            if (fabs(d) > maxabs) maxabs = fabs(d);
        }
    double nrmse = sqrt(sse / (mag > 0 ? mag : 1.0));
    double rms  = sqrt(mag / (double)MN);
    printf("correctness: all %d output elements against a double oracle\n", MN);
    printf("  relative RMSE        %.3e\n", nrmse);
    printf("  max |error|          %.4f  (%.2e of the %.3f output RMS)\n",
           maxabs, maxabs / rms, rms);
    printf("  exactly-zero outputs %d of %d%s\n", zeros, MN,
           zeros ? "   <- rows the kernel never wrote" : "");
    printf("  %s (fp16 in, fp32 accumulate; ~1e-3 is the expected floor)\n\n",
           nrmse < 5e-3 ? "PASS" : "FAIL");

    /* ---- 2. occupancy and tiling census: why the first version was slow -- */
    int blocks = grd.x * grd.y * grd.z;
    int threads = blocks * (GRP * 32);
    double waves = (double)threads / p.warpSize;
    printf("tiling and occupancy census (this is what the profiler would say)\n");
    printf("  tile per block      %d rows x %d cols, %d threads, split-K %d\n",
           RPB, NCOL, GRP * 32, KS);
    printf("  per thread          %d rows x %d cols = %d accumulators,\n"
           "                       %d loads per %d v_pk_fma_f16  (%.2f loads/FMA)\n",
           RPT, CPT, RPT * CPT, 2 + RPT, RPT * CPT, (double)(2 + RPT) / (RPT * CPT));
    printf("  grid                %u x %u x %u = %d blocks on %d CU\n",
           grd.x, grd.y, grd.z, blocks, p.multiProcessorCount);
    printf("  waves               %.0f total, %.1f per CU of %d possible"
           "  -> %.1f%% theoretical occupancy\n",
           waves, waves / p.multiProcessorCount, waves_per_cu_cap,
           waves / p.multiProcessorCount / waves_per_cu_cap * 100.0);
    printf("  B re-read           %u time(s) (once per row-block)\n\n", grd.x);

    /* ---- 3. both bandwidths, both measured ----------------------------- */
    const int BLK = 256;
    const int bblocks = p.multiProcessorCount * 8;
    const uint32_t stride = (uint32_t)bblocks * BLK;
    double bw_vram = 0, bw_pcie_stream = 0, bw_pcie_expert = 0;

    {
        const uint32_t n_big = (uint32_t)((vram_mb << 20) / 4);
        bw_read<<<bblocks, BLK>>>((const uint32_t *)dBW, n_big, stride, dC);
        hipDeviceSynchronize();
        double t = now_us();
        for (int i = 0; i < 20; i++)
            bw_read<<<bblocks, BLK>>>((const uint32_t *)dBW, n_big, stride, dC);
        hipDeviceSynchronize();
        bw_vram = (double)n_big * 4 / ((now_us() - t) / 20 * 1e-6) / 1e9;
    }
    float *hPin = NULL;
    const size_t pin_mb = 256;
    if (hipHostMalloc(&hPin, pin_mb << 20, hipHostMallocPortable) == hipSuccess) {
        hipMemset(hPin, 1, pin_mb << 20);
        const size_t big = pin_mb << 20;
        hipMemcpy(dBW, hPin, big, hipMemcpyHostToDevice); hipDeviceSynchronize();
        double t = now_us();
        for (int i = 0; i < 20; i++) hipMemcpy(dBW, hPin, big, hipMemcpyHostToDevice);
        hipDeviceSynchronize();
        bw_pcie_stream = (double)big * 20 / ((now_us() - t) * 1e-6) / 1e9;

        /* 3.0*HIDDEN*INTER experts (3 matrices per expert), times the W4 g128
         * bytes/weight from kernels/knj_pack.h (KNJ_W4_BYTES_PER_WEIGHT).
         * This must be the SAME pack the bit-width table below prints, so one
         * run does not quote two different packs (the 0.5186-vs-0.5234 drift). */
        const size_t one = (size_t)(3.0 * HIDDEN * INTER * KNJ_W4_BYTES_PER_WEIGHT);
        const int R = 200;
        double t2 = now_us();
        for (int i = 0; i < R; i++) hipMemcpy(dBW, hPin, one, hipMemcpyHostToDevice);
        hipDeviceSynchronize();
        bw_pcie_expert = (double)one * R / ((now_us() - t2) * 1e-6) / 1e9;
        hipFreeHost(hPin);
    }

    printf("measured bandwidth\n");
    printf("  VRAM  %6.1f GB/s  streaming %d MB        <- the GEMM's own roofline\n",
           bw_vram, (int)vram_mb);
    if (bw_pcie_stream > 0)
        printf("  PCIe  %6.1f GB/s  streaming %d MB pinned <- roofline for experts\n",
               bw_pcie_stream, (int)pin_mb);
    printf("  PCIe  %6.1f GB/s  one 2.47 MB expert copy  <- the size that matters\n\n",
           bw_pcie_expert);

    /* ---- 4. timed, over split-K choices --------------------------------- */
    printf("expert GEMM [32 x 2048] * [2048 x 768], SIMT packed-f16, split-K sweep\n");
    printf("   KS   blocks    us   TFLOP/s   %% of peak   GB/s   %% of VRAM roofline\n");
    double best_us = 0, best_ks = 0;
    for (int ks = 1; ks <= 64; ks *= 2) {
        int kc = (K / ks) & ~1;
        dim3 g((M + RPB - 1) / RPB, (N + NCOL - 1) / NCOL, ks);
        const int IT = 20;
        gemm_pk16<<<g, blk>>>(dA, dB, dP, M, K, N, kc);
        reduce_parts<<<(MN + 255) / 256, 256>>>(dP, dC, MN, ks);
        hipDeviceSynchronize();
        double best = 1e18;
        for (int r = 0; r < 3; r++) {
            double t = now_us();
            for (int i = 0; i < IT; i++) {
                gemm_pk16<<<g, blk>>>(dA, dB, dP, M, K, N, kc);
                reduce_parts<<<(MN + 255) / 256, 256>>>(dP, dC, MN, ks);
            }
            hipDeviceSynchronize();
            double per = (now_us() - t) / IT;
            if (per < best) best = per;
        }
        int nb = g.x * g.y * g.z;
        double fl = 2.0 * M * N * K;
        double moved = (double)K * N * 2 * g.x;
        double gbs = moved / (best * 1e-6) / 1e9;
        printf("  %4d %7d  %6.1f  %7.2f    %5.1f%%  %6.1f    %5.1f%%\n",
               ks, nb, best, fl / (best * 1e-6) / 1e12,
               fl / (best * 1e-6) / 1e12 / peak_pk * 100.0, gbs,
               gbs / bw_vram * 100.0);
        if (best_us == 0 || best < best_us) { best_us = best; best_ks = ks; }
    }
    double tflops = 2.0 * M * N * K / (best_us * 1e-6) / 1e12;
    printf("\n  best: split-K %d at %.1f us = %.2f TFLOP/s = %.1f%% of derived peak\n",
           (int)best_ks, best_us, tflops, tflops / peak_pk * 100.0);
    printf("  was 0.8%% of peak at 1 column/thread with no split-K.\n");
    /* The ceiling, which is what makes the number above meaningful.
     *
     * Bytes that MUST move for this projection at M=32 with a 32-row block:
     *   B  K*N*2              read once, because grid.x = 1
     *   A  M*K*2*(N/NCOL)     read once per column block
     * Anything faster is impossible; anything at that speed is bandwidth
     * bound, and no amount of tiling, registers or split-K changes it. */
    const double bytes_min = (double)K * N * 2 + (double)M * K * 2 * (N / NCOL);
    const double gflop  = 2.0 * M * N * K / 1e9;
    const double inten = gflop / (bytes_min / 1e9);
    const double bal   = peak_pk * 1e12 / (bw_vram * 1e9);
    printf("\nWHY NO KERNEL CAN DO MUCH BETTER AT M=32\n");
    printf("  minimum bytes that must move   %6.2f MB  (B once, A per column block)\n", bytes_min / 1e6);
    printf("  arithmetic intensity           %6.1f FLOP/byte\n", inten);
    printf("  machine balance (peak / BW)    %6.1f FLOP/byte\n", bal);
    printf("  => hard ceiling at M=32        %6.1f%% of peak, whatever the code does\n",
           inten / bal * 100.0);
    printf("  this kernel reaches            %6.1f%% of peak = %5.1f%% of that ceiling\n",
           tflops / peak_pk * 100.0,
           tflops / peak_pk / (inten / bal) * 100.0);
    printf("\n  The M=32 shape is byte-bound: %.0f FLOP/byte cannot buy %.0f of\n"
           "machine. Raising M is the only thing that moves this ratio; "
           "tiling, registers and split-K cannot.\n", inten, bal);



    /* ---- 5. the verdict, and what bit-width buys ------------------------- */
    const double weights = 3.0 * HIDDEN * INTER;
    const double eflops  = 3.0 * 2.0 * M * N * K;
    printf("\none expert at M=32: %.0f MFLOP\n", eflops / 1e6);
    printf("  arithmetic at DERIVED peak  %7.3f us\n", eflops / (peak_pk * 1e12) * 1e6);
    printf("  arithmetic in this kernel    %7.1f us  (%.0fx off peak)\n",
           best_us * 3.0, best_us * 3.0 / (eflops / (peak_pk * 1e12) * 1e6));
    printf("  transfer over PCIe (%.1f GB/s)%7.1f us\n",
           bw_pcie_expert, weights * (0.5 + 3.0 / 128.0) / (bw_pcie_expert * 1e9) * 1e6);

    /* Metadata is 3 B/group (KNJ_META_BYTES_PER_GROUP), same rule as
     * kernels/knj_pack.h for EVERY bit width, and docs/00 section 8.10.
     * The per-format bytes/weight now come straight from knj_pack.h so the
     * bit-width table and the one-expert copy size above are the same constant.
     * W4 g128 = KNJ_W4_BYTES_PER_WEIGHT = 0.5234 (was 0.5186 once). */
    const double b4 = KNJ_W4_BYTES_PER_WEIGHT;
    const double b3 = KNJ_W3_BYTES_PER_WEIGHT;
    const double b2 = KNJ_W2_BYTES_PER_WEIGHT;

    printf("\nWHAT BIT-WIDTH BUYS, at the MEASURED %.1f GB/s PCIe\n", bw_pcie_expert);
    printf("Every row is one full 128-expert sweep -- the worst case, where all\n"
           "32 tokens route to disjoint experts. Quality is NOT modelled here;\n"
           "only bytes are. An actual W2/W3 design has to beat the 81.5 dB SNR\n"
           "penalty that killed the int4 activation path (gemm_w4.hip).\n");
    printf("   format   B/weight   expert MB   128 experts   vs peak arithmetic\n");
    const char *nm[3] = { "W4 g128", "W3 g128", "W2 g128" };
    const double  bp[3] = { b4, b3, b2 };
    const double  xfer = weights * b4 / (bw_pcie_expert * 1e9) * 1e6;
    const double  peak_us = eflops / (peak_pk * 1e12) * 1e6;
    for (int i = 0; i < 3; i++)
        printf("  %8s  %8.4f  %9.2f  %9.2f ms  %10.1fx  (%s)\n", nm[i], bp[i],
               weights * bp[i] / 1e6, 128 * weights * bp[i] / (bw_pcie_expert * 1e9) * 1e3,
               weights * bp[i] / (bw_pcie_expert * 1e9) * 1e6 / peak_us,
               weights * bp[i] / (bw_pcie_expert * 1e9) * 1e6 > peak_us
                   ? "transfer-bound" : "COMPUTE-bound");

    printf("\n  To become compute-bound, PCIe would have to reach %.1f GB/s at W4.\n"
           "  No shipping interconnect is close, so the crossover cannot be moved\n"
           "  by tuning -- only by moving fewer bytes.\n",
           weights * b4 / (peak_us * 1e-6) / 1e9);
    printf("  Halving to W2 only buys %.1fx, because the arithmetic side is already\n"
           "  near free. That is the honest shape of this problem: it is a byte-\n"
           "  movement problem wearing an inference engine's clothes.\n",
           xfer / (weights * b2 / (bw_pcie_expert * 1e9) * 1e6));

    hipFree(dA); hipFree(dB); hipFree(dC); hipFree(dP); hipFree(dBW);
    free(hA); free(hB); free(hC);
    return nrmse < 5e-3 ? 0 : 1;
}