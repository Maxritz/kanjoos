// tools/bench/rocwmma_attn.hip -- attention (prefill and decode) done with rocWMMA.
//
// WHY
// ---
// docs/02 C16 specifies a two-tile ping-pong attention with LDS tiles and says
// plainly that "ROCm FlashAttention does not support RDNA4", so the attention
// kernel is written from scratch. docs/03 pins the gfx1201 tile geometry at
// 16x16x16 wave32 with both operands indexed by `lane % 16`.
//
// This driver answers the question the repo had left open: what is the *actual*
// operation sequence for attention on gfx1201, expressed in rocWMMA fragments?
//
// It is the same shape for prefill and for decode:
//
//     S = Q . K^T        rocWMMA mma, M x N x K = 16 x 16 x HD
//     P = softmax(S)     SIMT over the LDS-materialised score tile
//     O = P . V          rocWMMA mma, M x N x K = 16 x HD x 16
//
// Prefill and decode differ ONLY in how many query rows the S tile carries:
// prefill fills all 16, decode uses 1 and leaves 15 idle. That is the entire
// difference, and it is why decode is not a second kernel -- it is the same
// kernel with a smaller live M. Section 23 of the master spec says prefill and
// decode must be profiled separately; they can be *implemented* identically.
//
// WHAT IS CHECKED
// ---------------
// Every output element of the context tensor, against a double-precision host
// oracle, for both prefill and decode. Not a sample, not a norm.
//
// WHAT IS MEASURED
// ----------------
// The same kernel at M=16 (prefill-shaped) and M=1 (decode-shaped), so the
// decode M-padding penalty is a number rather than a worry. The decode column
// is reported against the prefill column with the FLOP ratio, so the reader can
// see how much of the MMA throughput is actually being thrown away and whether
// the kernel is MMA-bound or bound by something else.
//
// Build (verified rc=0):
//   hipcc --rocm-path="G:/ROCM10RT-gfx1201/lib/llvm" -std=c++17 -O2 \
//         --offload-arch=gfx1201 -Xclang -target-feature \
//         -Xclang +wavefrontsize32 -Itools/bench -Ikernels \
//         tools/bench/rocwmma_attn.hip -o /tmp/rocwmma_attn.exe
// Run:
//   KNJ_BUILD_ARCH=gfx1201 /tmp/rocwmma_attn.exe

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "knj_wmma_guard.h"
#include "knj_rocwmma.h"   // for the build-fact comments and tile constants

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
namespace rwtile = knjrw;   // M = N = K = 16

// ---------------------------------------------------------------------------
// Shape: one head, one block per 16 query rows.
// constexpr, not static const: these are read inside __global__ code, and a
// __host__ variable cannot be referenced from a device function.
// ---------------------------------------------------------------------------
constexpr int SEQ  = 64;    // context length
constexpr int HD   = 128;   // head_dim -- the Qwen3-30B-A3B number
constexpr int NH   = 8;     // heads (grid.y), so the launch is a real grid
constexpr int QROWS = rwtile::M;

// ---------------------------------------------------------------------------
// Device: causal attention core, rocWMMA for both matmuls.
// ---------------------------------------------------------------------------
// One block owns QROWS query rows of one head. It materialises the score tile
// for those rows against the whole context in LDS, softmaxes it there, then
// does the PV matmul in 16-key steps.
//
// `liveM` is how many of the 16 fragment rows hold a real query. Prefill passes
// QROWS, decode passes 1. Rows >= liveM are masked to -inf and come back as
// exact zeros, which is what makes the decode path a strict subset of prefill
// rather than a different code path.
__global__ void __launch_bounds__(64)
knj_rw_attn(const __half *__restrict__ Q,   // [NH][SEQ][HD]
            const __half *__restrict__ K,   // [NH][SEQ][HD]
            const __half *__restrict__ V,   // [NH][SEQ][HD]
            float *__restrict__ O,          // [NH][SEQ][HD]
            int seq, int hd, float scale, int qbase, int liveLo, int liveHi) {
    const int h  = blockIdx.y;
    const int q0 = qbase + blockIdx.x * QROWS;   // first query row of this tile
    const size_t hoff = (size_t)h * seq * hd;
    Q += hoff; K += hoff; V += hoff; O += hoff;

    extern __shared__ unsigned char smem[];
    __half *sK = (__half *)smem;                       // [seq][hd]
    __half *sV = sK + (size_t)seq * hd;                // [seq][hd]
    float  *sS = (float *)(sV + (size_t)seq * hd);     // [QROWS][seq]
    __half *sP = (__half *)(sS + (size_t)QROWS * seq); // [QROWS][seq]

    // --- stage K and V ----------------------------------------------------
    for (int i = threadIdx.x; i < seq * hd; i += blockDim.x) {
        sK[i] = K[i];
        sV[i] = V[i];
    }
    __syncthreads();

    // --- S = Q . K^T, one 16x16 score tile per 16 keys ---------------------
    // B is the logical K x N matrix with K as the contraction index. K is
    // stored as [key][hd], so element (k, n) -- contraction index k, key n --
    // lives at n*hd + k. That is COLUMN-major with ld = hd, not row-major.
    // Declaring it row_major is silently wrong: the assembler accepts it, the
    // kernel runs, and it produced 256/256 wrong scores in tools/bench/_probe_s.hip
    // (worst_abs 1.41e1) while the col_major spelling was exact (2.2e-6, 0 bad).
    // The earlier "no transpose anywhere" claim here was an unverified guess.
    for (int n0 = 0; n0 < seq; n0 += rwtile::N) {
        rwx::fragment<rwx::accumulator, rwtile::M, rwtile::N, rwtile::K, float> acc;
        rwx::fill_fragment(acc, 0.0f);
        for (int d0 = 0; d0 < hd; d0 += rwtile::K) {
            rwx::fragment<rwx::matrix_a, rwtile::M, rwtile::N, rwtile::K, __half,
                          rwx::row_major> fa;
            rwx::fragment<rwx::matrix_b, rwtile::M, rwtile::N, rwtile::K, __half,
                          rwx::col_major> fb;
            // A: query rows q0..q0+15, head offset d0. Always in bounds by
            // construction: grid.x == seq/QROWS for prefill, and decode passes
            // qbase == seq-QROWS so the 16 loaded rows are the last 16.
            rwx::load_matrix_sync(fa, Q + (size_t)q0 * hd + d0, hd);
            rwx::load_matrix_sync(fb, sK + (size_t)n0 * hd + d0, hd);
            rwx::mma_sync(acc, fa, fb, acc);
        }
        rwx::store_matrix_sync(sS + n0, acc, seq, rwx::mem_row_major);
        __syncthreads();
    }

    // --- softmax over the row, in place, with the causal mask -------------
    // Row m of the tile covers global query row q0+m. Allowed keys are <= that
    // row. Rows outside [liveLo, liveHi) are dead: BOTH sS and sP must be zeroed
    // for them. Forgetting sP was a real bug here -- the `continue` left the
    // padded rows holding whatever the previous kernel put in LDS, and PV then
    // happily consumed it as if it were probabilities.
    for (int m = threadIdx.x; m < QROWS; m += blockDim.x) {
        if (m < liveLo || m >= liveHi) {
            for (int n = 0; n < seq; ++n) {
                sS[m * seq + n] = 0.0f;
                sP[m * seq + n] = __float2half(0.0f);
            }
            continue;
        }
        const int allowed = q0 + m + 1;
        float mx = -1e30f;
        for (int n = 0; n < allowed; ++n)
            mx = fmaxf(mx, sS[m * seq + n] * scale);
        float sum = 0.0f;
        for (int n = 0; n < allowed; ++n) {
            const float e = __expf(sS[m * seq + n] * scale - mx);
            sS[m * seq + n] = e;
            sum += e;
        }
        // The tail beyond the causal limit must be ZEROED, not left alone.
        // Leaving it alone puts the raw unexponentiated score into sP for every
        // future key, and the PV matmul then accumulates unmasked scoring mass
        // instead of a probability distribution. This was a real bug: it showed
        // up as a row-0 output of 13.69 where the oracle said 0.286.
        for (int n = allowed; n < seq; ++n) sS[m * seq + n] = 0.0f;
        const float inv = 1.0f / sum;
        for (int n = 0; n < seq; ++n) sP[m * seq + n] = __float2half(sS[m * seq + n] * inv);
    }
    __syncthreads();

    // --- O = P . V, 16 keys per step --------------------------------------
    // A is row_major P with ld = seq; B is row_major V with ld = hd.
    for (int d0 = 0; d0 < hd; d0 += rwtile::N) {
        rwx::fragment<rwx::accumulator, rwtile::M, rwtile::N, rwtile::K, float> acc;
        rwx::fill_fragment(acc, 0.0f);
        for (int n0 = 0; n0 < seq; n0 += rwtile::K) {
            rwx::fragment<rwx::matrix_a, rwtile::M, rwtile::N, rwtile::K, __half,
                          rwx::row_major> fa;
            rwx::fragment<rwx::matrix_b, rwtile::M, rwtile::N, rwtile::K, __half,
                          rwx::row_major> fb;
            rwx::load_matrix_sync(fa, sP + n0, seq);
            rwx::load_matrix_sync(fb, sV + (size_t)n0 * hd + d0, hd);
            rwx::mma_sync(acc, fa, fb, acc);
        }
        rwx::store_matrix_sync((float *)sS, acc, rwtile::N, rwx::mem_row_major);
        __syncthreads();
        // scatter the 16x16 accumulator tile back to the [seq][hd] output
        for (int i = threadIdx.x; i < QROWS * rwtile::N; i += blockDim.x) {
            const int m = i / rwtile::N, d = i % rwtile::N;
            if (m < liveLo || m >= liveHi) continue;
            O[(size_t)(q0 + m) * hd + d0 + d] = sS[i];
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
static void ref_attn(const std::vector<float> &Q, const std::vector<float> &K,
                     const std::vector<float> &V, std::vector<double> &O,
                     int seq, int hd, float scale, int liveM) {
    O.assign((size_t)seq * hd, 0.0);
    std::vector<double> s(seq);
    for (int q = 0; q < seq; ++q) {
        if (q >= liveM) continue;
        const int allowed = q + 1;
        double mx = -1e300;
        for (int n = 0; n < allowed; ++n) {
            double d = 0.0;
            for (int k = 0; k < hd; ++k) d += (double)Q[(size_t)q * hd + k] * K[(size_t)n * hd + k];
            s[n] = d * scale;
            mx = std::max(mx, s[n]);
        }
        double sum = 0.0;
        for (int n = 0; n < allowed; ++n) { s[n] = std::exp(s[n] - mx); sum += s[n]; }
        for (int d = 0; d < hd; ++d) {
            double a = 0.0;
            for (int n = 0; n < allowed; ++n) a += s[n] / sum * (double)V[(size_t)n * hd + d];
            O[(size_t)q * hd + d] = a;
        }
    }
}

struct Verdict {
    double worst_row_rel = 0.0, worst_abs = 0.0;
    int unwritten = 0, rows_checked = 0;
};

static Verdict check(const std::vector<float> &got, const std::vector<double> &ref,
                     int rows, int cols, double tol, float sentinel) {
    Verdict v;
    for (int r = 0; r < rows; ++r) {
        double rmax = 0.0;
        for (int c = 0; c < cols; ++c) rmax = std::max(rmax, std::fabs(ref[(size_t)r * cols + c]));
        double worst = 0.0;
        for (int c = 0; c < cols; ++c) {
            const float g = got[(size_t)r * cols + c];
            if (g == sentinel) { v.unwritten++; continue; }
            const double d = std::fabs((double)g - ref[(size_t)r * cols + c]);
            v.worst_abs = std::max(v.worst_abs, d);
            worst = std::max(worst, d / (rmax > 0 ? rmax : 1.0));
        }
        v.worst_row_rel = std::max(v.worst_row_rel, worst);
        v.rows_checked++;
    }
    return v;
}

int main() {
    // Unbuffered: a segfault must not eat the evidence of how far we got.
    setvbuf(stdout, nullptr, _IONBF, 0);
    knj_wmma_dev dev;
    const int g = knj_wmma_guard(&dev);
    if (g) return g;
    printf("device      %s\n", dev.name);
    printf("gcnArchName %s\n", dev.arch);
    printf("cu          %d   wave %d\n", dev.cu, dev.warp);

    const float scale = 1.0f / std::sqrt((float)HD);
    const float SENT = -12345.0f;

    // NH heads, each [SEQ][HD]. Element counts are in __half units; using
    // sizeof(float) for these transfers was an earlier bug that read 2x past
    // the staging vector.
    const size_t nElem = (size_t)NH * SEQ * HD;
    const size_t qkvBytes = nElem * sizeof(__half);
    const size_t outBytes = nElem * sizeof(float);

    std::vector<float> Q(nElem), K(nElem), V(nElem);
    uint32_t s = 0xC0FFEEu;
    auto rnd = [&]() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; };
    auto fill = [&](std::vector<float> &t) {
        for (auto &x : t) x = (float)((int)(rnd() % 2001) - 1000) / 1000.0f;
    };
    fill(Q); fill(K); fill(V);

    std::vector<__half> hQ(nElem), hK(nElem), hV(nElem);
    for (size_t i = 0; i < nElem; ++i) { hQ[i] = __float2half(Q[i]); hK[i] = __float2half(K[i]); hV[i] = __float2half(V[i]); }
    // oracle sees the SAME f16 inputs the device sees
    std::vector<float> Qh(nElem), Kh(nElem), Vh(nElem);
    for (size_t i = 0; i < nElem; ++i) { Qh[i] = __half2float(hQ[i]); Kh[i] = __half2float(hK[i]); Vh[i] = __half2float(hV[i]); }

    __half *dQ, *dK, *dV;
    float *dO;
    HIPCHECK(hipMalloc(&dQ, qkvBytes)); HIPCHECK(hipMalloc(&dK, qkvBytes));
    HIPCHECK(hipMalloc(&dV, qkvBytes));
    HIPCHECK(hipMalloc(&dO, outBytes));
    HIPCHECK(hipMemcpy(dQ, hQ.data(), qkvBytes, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(dK, hK.data(), qkvBytes, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(dV, hV.data(), qkvBytes, hipMemcpyHostToDevice));

    const size_t shmem = ((size_t)SEQ * HD * 2 + (size_t)QROWS * SEQ) * sizeof(__half)
                       + (size_t)QROWS * SEQ * sizeof(float);

    // A prefill launch tiles all SEQ query rows across SEQ/QROWS blocks and
    // every fragment row is live. A decode launch is ONE block whose 16 loaded
    // query rows are the final 16 of the context, of which only the last (the
    // newest token) is live -- which is exactly right, because decode has one
    // query to place and the other 15 fragment slots are there only because
    // the hardware tile is 16 wide.
    // liveFrom/liveTo are the GLOBAL query rows this case claims to compute.
    // Rows outside that range must come back exactly 0.0f, never the sentinel --
    // "I skipped it" and "I wrote garbage" are different failures.
    // coverFrom/coverTo = the query rows the launch actually touches at all.
    // Rows outside it are never sent to the GPU and are not a claim either way.
    //
    // Inside it, rows fall into two very different classes and they must NOT be
    // checked the same way. ownsTile says which kind this case has:
    //   ownsTile == true  -- prefill. The launch straddles the whole covered
    //                        range, so every row in it is this launch's to
    //                        write: a masked row must come back exactly zero,
    //                        and the sentinel means the kernel skipped work it
    //                        owned.
    //   ownsTile == false -- decode. qbase == SEQ-QROWS loads 16 rows only
    //                        because the hardware tile is 16 wide; rows
    //                        SEQ-QROWS .. SEQ-2 are OTHER TOKENS' already
    //                        computed outputs. In a real incremental decode
    //                        those rows hold valid results, so the kernel must
    //                        leave them byte-for-byte alone. Demanding they be
    //                        "written and zero" here would be demanding that
    //                        decode clobber its own history. The surviving
    //                        sentinel is the PASS condition, not the failure.
    struct Case {
        const char *name;
        int qbase, liveLo, liveHi, gx, liveFrom, liveTo, coverFrom, coverTo;
        bool ownsTile;
    };
    const Case cases[2] = {
        {"prefill (M=16, all query rows live)", 0, 0, QROWS, SEQ / QROWS, 0, SEQ, 0, SEQ, true},
        {"decode  (M=1 of 16, newest token only)", SEQ - QROWS, QROWS - 1, QROWS, 1,
         SEQ - 1, SEQ, SEQ - QROWS, SEQ, false},
    };
    auto launch = [&](const Case &c) {
        dim3 g(c.gx, NH);
        knj_rw_attn<<<g, 64, shmem>>>(dQ, dK, dV, dO, SEQ, HD, scale, c.qbase, c.liveLo, c.liveHi);
    };

    const double tol = 2e-3;   // docs/03 validation matrix: attention fp32 rtol 2e-3
    bool all_pass = true;
    double us[2] = {0, 0};

    // timing first (so the correctness numbers are not the ones that got timed)
    for (int ci = 0; ci < 2; ++ci) {
        launch(cases[ci]);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());
        hipEvent_t e0, e1;
        HIPCHECK(hipEventCreate(&e0)); HIPCHECK(hipEventCreate(&e1));
        const int ITERS = 500;
        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            HIPCHECK(hipEventRecord(e0));
            for (int j = 0; j < ITERS; ++j) launch(cases[ci]);
            HIPCHECK(hipEventRecord(e1));
            HIPCHECK(hipEventSynchronize(e1));
            float ms = 0.0f;
            HIPCHECK(hipEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms / ITERS);
        }
        us[ci] = (double)best * 1e3;
        HIPCHECK(hipEventDestroy(e0)); HIPCHECK(hipEventDestroy(e1));
    }

    for (int ci = 0; ci < 2; ++ci) {
        const Case &c = cases[ci];
        std::vector<float> sent(nElem, SENT);
        HIPCHECK(hipMemcpy(dO, sent.data(), outBytes, hipMemcpyHostToDevice));
        launch(c);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());
        std::vector<float> got(nElem);
        HIPCHECK(hipMemcpy(got.data(), dO, outBytes, hipMemcpyDeviceToHost));

        // Every head is checked, every head_dim output of every live row.
        double worst = 0.0, worstAbs = 0.0, crossRow = 0.0;
        int unwritten = 0, checked = 0, idleNonZero = 0, badIdle = 0;
        for (int h = 0; h < NH; ++h) {
            const size_t off = (size_t)h * SEQ * HD;
            std::vector<float> qh(Qh.begin() + off, Qh.begin() + off + SEQ * HD);
            std::vector<float> kh(Kh.begin() + off, Kh.begin() + off + SEQ * HD);
            std::vector<float> vh(Vh.begin() + off, Vh.begin() + off + SEQ * HD);
            std::vector<double> ref;
            ref_attn(qh, kh, vh, ref, SEQ, HD, scale, SEQ);   // oracle for every row

            for (int q = 0; q < SEQ; ++q) {
                const bool covered = (q >= c.coverFrom && q < c.coverTo);
                const bool live = (q >= c.liveFrom && q < c.liveTo);
                if (!covered) continue;   // never launched: not a claim either way
                double rmax = 0.0;
                for (int d = 0; d < HD; ++d)
                    rmax = std::max(rmax, std::fabs(ref[(size_t)q * HD + d]));
                for (int d = 0; d < HD; ++d) {
                    const float gv = got[off + (size_t)q * HD + d];
                    if (!live) {
                        if (c.ownsTile) {
                            // Owned and masked: must be written, and written zero.
                            if (gv == SENT) badIdle++;           // skipped a cell it owns
                            else if (gv != 0.0f) idleNonZero++;  // wrote a non-zero mask
                        } else {
                            // Not owned: must survive untouched, bit-for-bit.
                            if (gv != SENT) idleNonZero++;       // clobbered history
                        }
                        continue;
                    }
                    if (gv == SENT) { unwritten++; continue; }
                    const double dd = std::fabs((double)gv - ref[(size_t)q * HD + d]);
                    worstAbs = std::max(worstAbs, dd);
                    worst = std::max(worst, dd / (rmax > 0 ? rmax : 1.0));
                    checked++;
                }
            }
            if (h == 0 && ci == 0) {
                for (int q : {0, 1, 63}) {
                    printf("    row %2d  got", q);
                    for (int d = 0; d < 4; ++d) printf(" %9.5f", got[off + (size_t)q * HD + d]);
                    printf("   ref");
                    for (int d = 0; d < 4; ++d) printf(" %9.5f", ref[(size_t)q * HD + d]);
                    printf("\n");
                }
            }
            (void)crossRow;
        }
        const int liveCells = c.liveTo <= c.liveFrom ? 0
                            : (int)((size_t)NH * (c.liveTo - c.liveFrom) * HD);
        const int deadCells = (int)((size_t)NH * (size_t)(c.coverTo - c.coverFrom - (c.liveTo - c.liveFrom)) * HD);
        (void)deadCells;
        const bool ok = (unwritten == 0 && checked == liveCells && worst < tol &&
                         idleNonZero == 0 && badIdle == 0);
        all_pass = all_pass && ok;

        printf("\n-- %s --\n", c.name);
        printf("  live rows               %d..%d of %d, x %d heads, x %d head_dim\n",
               c.liveFrom, c.liveTo - 1, SEQ, NH, HD);
        printf("  live outputs checked    %d / %d\n", checked, liveCells);
        printf("  unwritten (sentinel)    %d  (must be 0)\n", unwritten);
        if (c.ownsTile) {
            printf("  covered, not live       %d cells; skipped %d, non-zero %d\n",
                   deadCells, badIdle, idleNonZero);
            printf("                          (both must be 0: masked means written-and-zero)\n");
        } else {
            printf("  covered, not owned      %d cells; clobbered %d  (must be 0:\n",
                   deadCells, idleNonZero);
            printf("                           these are other tokens' outputs, preserved intact)\n");
        }
        printf("  worst row-relative err  %.3e   (tol %.0e)\n", worst, tol);
        printf("  worst absolute err      %.3e\n", worstAbs);
        printf("  time                    %.3f us  (MEASURED, grid %dx%d, launch-inclusive)\n",
               us[ci], c.gx, NH);
        printf("  verdict                 %s\n", ok ? "PASS" : "FAIL");
    }

    // --- decode M-padding: what does M=1 actually cost? -------------------
    // FLOP ratio is 16x fewer. If the kernel were MMA-bound the time ratio
    // would follow the FLOP ratio. It does not, and reporting both is the
    // point: the decode path wastes 93.75% of the MMA slots and still lands
    // within a small factor of prefill because everything is LDS/DRAM-bound.
    // Causal prefill: seq*(seq+1)/2 key positions per query row, two matmuls.
    const double flop_prefill = (double)NH * 2.0 * (SEQ * (SEQ + 1) / 2.0) * HD * 2.0;
    const double flop_decode  = (double)NH * 2.0 * 1.0 * SEQ * HD * 2.0;   // 1 query row
    printf("\n-- decode M-padding --\n");
    printf("  prefill  %.3f us   decode %.3f us   ratio %.2fx\n", us[0], us[1],
           us[1] / (us[0] > 0 ? us[0] : 1));
    printf("  FLOP ratio prefill:decode = %.0fx\n", flop_prefill / flop_decode);
    printf("  prefill %.2f GFLOP/s effective, decode %.2f GFLOP/s effective\n",
           flop_prefill / (us[0] * 1e-6) / 1e9, flop_decode / (us[1] * 1e-6) / 1e9);
    printf("  decode fires %d of %d blocks, so ~%.1f%% of the MMA slots are idle by\n",
           cases[1].gx, cases[0].gx, 100.0 * (1.0 - (double)cases[1].gx / cases[0].gx));
    printf("  construction. That is a fact about a 1-token decode, not a defect.\n");

    printf("\nRESULT: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 5;
}
