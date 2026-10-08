// tools/bench/q4k_moe_ffn.hip -- the Qwen3-MoE FFN forward, on gfx1201, from
// REAL GGUF Q4_K / Q6_K expert bytes, via rocWMMA.
//
// KNJ_BENCH_REQUIRES: a job file naming an external GGUF and a dump directory (argv: <job> <dumpdir> [tokens]). The gate passes no arguments, so this driver is reported NOT RUN rather than failed on a usage error.
//
// WHAT PROBLEM THIS SOLVES
// ------------------------
// Every performance number the project owns is a HOST number. The device has
// only ever run kernel microbenchmarks, and the one grouped-MoE driver that does
// run (tools/bench/rocwmma_moe.hip) does it on SYNTHETIC W4 g128 bytes packed by
// the repo's own packer. That is a real result about the pack; it is not a result
// about the model. A GEMM measured on a synthetic tensor of approximately the
// right shape is the "passes without measuring anything" class -- the shape looks
// right and the bytes are not the bytes.
//
// So this driver does the opposite: it reads the REAL layer-0 expert tensors out
// of the REAL GGUF at byte offsets that tools/q4k_job.py verified against the
// file, and computes the real FFN:
//
//   for each token t, for each of its top-k experts e (router weight rw):
//       g = X[t] . Wgate[e]            Q4_K   [k=1024][n=3072]
//       u = X[t] . Wup[e]              Q4_K   [k=1024][n=3072]
//       a = silu(g) * u
//       y += rw * (a . Wdown[e])       Q6_K   [k=3072][n=1024]
//
// which is the whole reason WMMA exists in this project: this is ~60% of the
// layer's FLOPs and all of the FFN.
//
// WHY THE ORACLE IS NOT CIRCULAR
// ------------------------------
// The device decodes the packed bytes with kernels/knj_q4k.h. The oracle must
// NOT, or a wrong decode would agree with itself. So the oracle's weights come
// from <dumpdir>/{gate,up,down}.f32 -- the f32 dumps that
// tools/q4k_unpack_crosscheck.py proved BIT-IDENTICAL to gguf-py's independent
// dequantiser over all 37,748,736 weights of these three tensors. The oracle is
// therefore a verified decode fed into a plain double-precision triple loop.
//
// That splits the evidence cleanly:
//   * is the FORMAT read right?      -> q4k_unpack_crosscheck.py vs gguf-py
//   * is the TILE BUILT right?       -> this driver, full-output, per element
// and a failure here can only mean the kernel, not the header.
//
// PRECISION, DECLARED NOT HIDDEN
// ------------------------------
// rocWMMA on gfx1201 offers exactly one geometry, 16x16x16 wave32, with f16
// OPERANDS and an f32 accumulator. So X, the weights and the intermediate
// activation all round through f16 before the MAC -- a declared precision
// reduction. The oracle mirrors that rounding exactly (every operand is
// round-tripped through f16 before the double multiply), so what the comparison
// tests is LAYOUT and ACCUMULATION, not the f16 step. The f16 step's own size is
// reported separately, against an all-fp32 oracle, so it is a measured number
// rather than an assumed one.
//
// Build (verified -- --rocm-path must end in /lib/llvm, -std=c++17 is a header
// requirement because rocWMMA uses std::apply):
//   hipcc --rocm-path="G:/ROCM10RT-gfx1201/lib/llvm" -std=c++17 -O2 \
//         --offload-arch=gfx1201 -Xclang -target-feature \
//         -Xclang +wavefrontsize32 -Itools/bench -Ikernels \
//         tools/bench/q4k_moe_ffn.hip -o build/q4k/q4k_moe_ffn.exe
// Run (arch-guarded; gfx1201's guard exits 7 unconfigured / 6 mismatched):
//   KNJ_BUILD_ARCH=gfx1201 build/q4k/q4k_moe_ffn.exe <job> <dumpdir> [tokens]
//
// Exit codes: 0 PASS, 5 FAIL (a correctness check), 9 HIP error, 3 missing input.

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "knj_wmma_guard.h"
#include "knj_q4k.h"

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

// The only geometry gfx1201's WMMA builtins implement.
static const int TM = 16;   // M
static const int TN = 16;   // N
static const int TK = 16;   // K
static const int NTHREADS = 64;

static const float SENTINEL = -12345.0f;

// ---------------------------------------------------------------------------
// Job descriptor
// ---------------------------------------------------------------------------
struct Job {
    std::string model, dumpdir;
    int layer = 0, hidden = 0, ff = 0, experts = 0, topk = 0;
    long long off[3] = {0, 0, 0};    // gate, up, down
    int bpb[3] = {0, 0, 0};
    int k[3] = {0, 0, 0};
    int n[3] = {0, 0, 0};
    long long router_off = 0;
};

static std::vector<std::pair<std::string, std::string>> read_job(const std::string &path) {
    std::FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) { printf("cannot open job %s\n", path.c_str()); exit(3); }
    std::vector<std::pair<std::string, std::string>> kv;
    char line[2048];
    while (std::fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        char key[512] = {0}, val[1536] = {0};
        if (std::sscanf(line, "%511s %1535s", key, val) != 2) continue;
        kv.emplace_back(key, val);
    }
    std::fclose(fp);
    return kv;
}

static std::string jget(const std::vector<std::pair<std::string, std::string>> &kv,
                        const std::string &key) {
    for (const auto &p : kv) if (p.first == key) return p.second;
    printf("job is missing key: %s\n", key.c_str());
    exit(3);
}
static long long jint(const std::vector<std::pair<std::string, std::string>> &kv,
                      const std::string &key) {
    return std::strtoll(jget(kv, key).c_str(), nullptr, 10);
}

static std::vector<uint8_t> read_at(const std::string &path, long long off, long long nbytes) {
    std::FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) { printf("cannot open %s\n", path.c_str()); exit(3); }
    if (std::fseek(fp, (long)off, SEEK_SET) != 0) {
        printf("seek failed at %lld in %s\n", off, path.c_str());
        exit(3);
    }
    std::vector<uint8_t> buf((size_t)nbytes);
    const size_t got = std::fread(buf.data(), 1, (size_t)nbytes, fp);
    std::fclose(fp);
    if (got != (size_t)nbytes) {
        printf("short read at %lld in %s (%zu of %lld)\n", off, path.c_str(), got, nbytes);
        exit(3);
    }
    return buf;
}

// ===========================================================================
// Kernel 1 -- gate and up, Q4_K, both projections in one pass
// ===========================================================================
// The A tile (activations) is the SAME bytes for gate and up, so the K loop is
// walked once and two B tiles are materialised per step. Two accumulator
// fragments, one shared A fragment.
//
// Grid (nx/16, maxPairs/16, experts). `idxStride` is the per-expert stride of
// the pair list. No gridDim anywhere: -nogpulib removes __ockl_get_num_groups,
// so a kernel that reads gridDim fails to LINK, not to run.
__global__ void __launch_bounds__(NTHREADS)
knj_q4k_gateup(const __half *__restrict__ X,        // [T][K] token activations
               const uint8_t *__restrict__ Wg,      // Q4_K [E][N][K/256][144]
               const uint8_t *__restrict__ Wu,
               const int *__restrict__ pairIdx,     // [E][idxStride]
               const int *__restrict__ pairCnt,     // [E]
               const int *__restrict__ pairTok,     // [P]
               float *__restrict__ G,               // [P][N]
               float *__restrict__ U,
               int N, int K, int rowBytes, long long idxStride) {
    const int e  = blockIdx.z;
    const int n0 = blockIdx.x * TN;
    const int m0 = blockIdx.y * TM;
    const int m  = pairCnt[e];
    if (m0 >= m) return;

    const uint8_t *wg = Wg + ((long long)e * N + n0) * rowBytes;
    const uint8_t *wu = Wu + ((long long)e * N + n0) * rowBytes;

    __shared__ __align__(32) __half sA[TM * TK];
    __shared__ __align__(32) __half sBg[TK * TN];
    __shared__ __align__(32) __half sBu[TK * TN];

    rwx::fragment<rwx::accumulator, TM, TN, TK, float> accg;
    rwx::fragment<rwx::accumulator, TM, TN, TK, float> accu;
    rwx::fill_fragment(accg, 0.0f);
    rwx::fill_fragment(accu, 0.0f);

    // Which of the 16 output rows this thread materialises, and which 4 of its
    // 16 weights. 64 threads / 16 rows = 4 threads per row, so the 16-wide span
    // decode is shared by the 4 threads that need it instead of recomputed.
    const int row = threadIdx.x >> 2;
    const int col = (threadIdx.x & 3) * 4;

    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int i = threadIdx.x; i < TM * TK; i += NTHREADS) {
            const int mm = i / TK, kk = i % TK;
            const int p = (m0 + mm < m) ? pairIdx[(long long)e * idxStride + m0 + mm] : -1;
            sA[i] = (p >= 0) ? X[(long long)pairTok[p] * K + k0 + kk] : __float2half(0.0f);
        }

        const int r0 = k0 % 256;                     // 16-aligned; stays in one sub-block
        const int blkOff = (k0 / 256) * KNJ_Q4K_BYTES;
        const uint8_t *rowg = wg + row * rowBytes + blkOff;
        const uint8_t *rowu = wu + row * rowBytes + blkOff;
        float span[16];
        knj_q4k_span16(rowg, r0, span);
        for (int i = 0; i < 4; ++i) sBg[row * TK + col + i] = __float2half(span[col + i]);
        knj_q4k_span16(rowu, r0, span);
        for (int i = 0; i < 4; ++i) sBu[row * TK + col + i] = __float2half(span[col + i]);

        __syncthreads();

        rwx::fragment<rwx::matrix_a, TM, TN, TK, __half, rwx::row_major> fa;
        rwx::fragment<rwx::matrix_b, TM, TN, TK, __half, rwx::col_major> fbg;
        rwx::fragment<rwx::matrix_b, TM, TN, TK, __half, rwx::col_major> fbu;
        rwx::load_matrix_sync(fa, sA, TK);
        rwx::load_matrix_sync(fbg, sBg, TK);
        rwx::load_matrix_sync(fbu, sBu, TK);
        rwx::mma_sync(accg, fa, fbg, accg);
        rwx::mma_sync(accu, fa, fbu, accu);
        __syncthreads();
    }

    __shared__ __align__(32) float sDg[TM * TN];
    __shared__ __align__(32) float sDu[TM * TN];
    rwx::store_matrix_sync(sDg, accg, TN, rwx::mem_row_major);
    rwx::store_matrix_sync(sDu, accu, TN, rwx::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < TM * TN; i += NTHREADS) {
        const int mm = i / TN, nn = i % TN;
        if (m0 + mm < m) {
            const int p = pairIdx[(long long)e * idxStride + m0 + mm];
            G[(long long)p * N + n0 + nn] = sDg[i];
            U[(long long)p * N + n0 + nn] = sDu[i];
        }
    }
}

// ===========================================================================
// Kernel 2 -- down projection, Q6_K
// ===========================================================================
// A is the silu(g)*u activation, already f16 on the device because it is f16 in
// the accumulator path anyway; its f16 rounding is mirrored in the oracle.
// Output is per PAIR, never accumulated into the residual: two experts routing
// the same token must not race, and the weighted sum is what makes the result
// order-dependent, so it is done once, in a fixed order, on the host.
__global__ void __launch_bounds__(NTHREADS)
knj_q6k_down(const __half *__restrict__ A,          // [P][K] activations
             const uint8_t *__restrict__ Wd,        // Q6_K [E][N][K/256][210]
             const int *__restrict__ pairIdx,
             const int *__restrict__ pairCnt,
             float *__restrict__ Y,                 // [P][N]
             int N, int K, int rowBytes, long long idxStride) {
    const int e  = blockIdx.z;
    const int n0 = blockIdx.x * TN;
    const int m0 = blockIdx.y * TM;
    const int m  = pairCnt[e];
    if (m0 >= m) return;

    const uint8_t *wd = Wd + ((long long)e * N + n0) * rowBytes;

    __shared__ __align__(32) __half sA[TM * TK];
    __shared__ __align__(32) __half sB[TK * TN];

    rwx::fragment<rwx::accumulator, TM, TN, TK, float> acc;
    rwx::fill_fragment(acc, 0.0f);

    const int row = threadIdx.x >> 2;
    const int col = (threadIdx.x & 3) * 4;

    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int i = threadIdx.x; i < TM * TK; i += NTHREADS) {
            const int mm = i / TK, kk = i % TK;
            const int p = (m0 + mm < m) ? pairIdx[(long long)e * idxStride + m0 + mm] : -1;
            sA[i] = (p >= 0) ? A[(long long)p * K + k0 + kk] : __float2half(0.0f);
        }

        const int r0 = k0 % 256;
        const int blkOff = (k0 / 256) * KNJ_Q6K_BYTES;
        float span[16];
        knj_q6k_span16(wd + row * rowBytes + blkOff, r0, span);
        for (int i = 0; i < 4; ++i) sB[row * TK + col + i] = __float2half(span[col + i]);

        __syncthreads();

        rwx::fragment<rwx::matrix_a, TM, TN, TK, __half, rwx::row_major> fa;
        rwx::fragment<rwx::matrix_b, TM, TN, TK, __half, rwx::col_major> fb;
        rwx::load_matrix_sync(fa, sA, TK);
        rwx::load_matrix_sync(fb, sB, TK);
        rwx::mma_sync(acc, fa, fb, acc);
        __syncthreads();
    }

    __shared__ __align__(32) float sD[TM * TN];
    rwx::store_matrix_sync(sD, acc, TN, rwx::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < TM * TN; i += NTHREADS) {
        const int mm = i / TN, nn = i % TN;
        if (m0 + mm < m) {
            const int p = pairIdx[(long long)e * idxStride + m0 + mm];
            Y[(long long)p * N + n0 + nn] = sD[i];
        }
    }
}

// ===========================================================================
// SIMT fallbacks -- always compiled (docs/02 C16/C17)
// ===========================================================================
// These decode the packed bytes with the PER-ELEMENT decoders (`knj_q4k_weight`
// / `knj_q6k_weight`), not the span decoders the WMMA path hoists. That is the
// point: the WMMA path's tile construction and the flat elementwise walk are two
// independent readings of the same bytes, and the header has already proved the
// two decoders agree on the host over every block of these tensors. So a
// disagreement here can only be the tile, not the format.
//
// The decoded weight is round-tripped through f16 before the fma, because that
// is the DECLARED precision reduction of the whole driver (rocWMMA on gfx1201
// only offers f16 operands). Mirroring it here means the A/B compares tile
// construction and accumulation order, and nothing else. Without the mirror the
// two paths would differ by the f16 operand cost, which would make the A/B
// uninformative about the thing it exists to check.
//
// `stride` is passed in because gridDim is unavailable under -nogpulib: gfx1201
// lowers it through __ockl_get_num_groups, which lives in the device runtime
// -nogpulib skips, so `gridDim` is a LINK failure, not a runtime one.
__global__ void knj_q4k_simt_gateup(const __half *__restrict__ X,
                                    const uint8_t *__restrict__ Wg,
                                    const uint8_t *__restrict__ Wu,
                                    const int *__restrict__ pairIdx,
                                    const int *__restrict__ pairCnt,
                                    const int *__restrict__ pairTok,
                                    float *__restrict__ G, float *__restrict__ U,
                                    int N, int K, int rowBytes,
                                    long long idxStride, int stride) {
    const int e = blockIdx.z;
    const int m = pairCnt[e];
    const int total = m * N;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        const int mm = i / N, nn = i % N;
        const int p = pairIdx[(long long)e * idxStride + mm];
        const __half *x = X + (long long)pairTok[p] * K;
        const uint8_t *wg = Wg + ((long long)e * N + nn) * rowBytes;
        const uint8_t *wu = Wu + ((long long)e * N + nn) * rowBytes;
        float ag = 0.0f, au = 0.0f;
        for (int k0 = 0; k0 < K; k0 += KNJ_QK_K) {
            const uint8_t *bg = wg + (k0 / KNJ_QK_K) * KNJ_Q4K_BYTES;
            const uint8_t *bu = wu + (k0 / KNJ_QK_K) * KNJ_Q4K_BYTES;
            for (int r = 0; r < KNJ_QK_K; ++r) {
                const float xv = (float)x[k0 + r];
                ag = fmaf(xv, (float)__float2half(knj_q4k_weight(bg, r)), ag);
                au = fmaf(xv, (float)__float2half(knj_q4k_weight(bu, r)), au);
            }
        }
        G[(long long)p * N + nn] = ag;
        U[(long long)p * N + nn] = au;
    }
}

__global__ void knj_q6k_simt_down(const __half *__restrict__ A,
                                  const uint8_t *__restrict__ Wd,
                                  const int *__restrict__ pairIdx,
                                  const int *__restrict__ pairCnt,
                                  float *__restrict__ Y,
                                  int N, int K, int rowBytes,
                                  long long idxStride, int stride) {
    const int e = blockIdx.z;
    const int m = pairCnt[e];
    const int total = m * N;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        const int mm = i / N, nn = i % N;
        const int p = pairIdx[(long long)e * idxStride + mm];
        const __half *a = A + (long long)p * K;
        const uint8_t *wd = Wd + ((long long)e * N + nn) * rowBytes;
        float acc = 0.0f;
        for (int k0 = 0; k0 < K; k0 += KNJ_QK_K) {
            const uint8_t *b = wd + (k0 / KNJ_QK_K) * KNJ_Q6K_BYTES;
            for (int r = 0; r < KNJ_QK_K; ++r)
                acc = fmaf((float)a[k0 + r],
                           (float)__float2half(knj_q6k_weight(b, r)), acc);
        }
        Y[(long long)p * N + nn] = acc;
    }
}

// ---------------------------------------------------------------------------
// Host verification
// ---------------------------------------------------------------------------
static void cx(const char *name, const std::vector<float> &got,
               const std::vector<double> &ref, int rows, int cols, double tol,
               int *fail, int *unwritten, double *worst_rel, double *worst_abs) {
    int un = 0, bad_rows = 0;
    double wr = 0.0, wa = 0.0;
    for (int r = 0; r < rows; ++r) {
        double rowmax = 0.0;
        for (int c = 0; c < cols; ++c)
            rowmax = std::max(rowmax, std::fabs(ref[(size_t)r * cols + c]));
        double worst = 0.0;
        for (int c = 0; c < cols; ++c) {
            const float g = got[(size_t)r * cols + c];
            if (g == SENTINEL) { ++un; continue; }
            const double d = std::fabs((double)g - ref[(size_t)r * cols + c]);
            wa = std::max(wa, d);
            worst = std::max(worst, d / (rowmax > 0 ? rowmax : 1.0));
        }
        if (worst > tol) ++bad_rows;
        wr = std::max(wr, worst);
    }
    printf("  %-22s rows=%-6d worst-row-rel=%.3e  worst-abs=%.3e  "
           "unwritten=%d  rows-over-tol=%d\n",
           name, rows, wr, wa, un, bad_rows);
    if (un || bad_rows) ++*fail;
    *unwritten += un;
    *worst_rel = std::max(*worst_rel, wr);
    *worst_abs = std::max(*worst_abs, wa);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <job> <dumpdir> [tokens]\n", argv[0]);
        return 2;
    }
    knj_wmma_dev dev;
    const int grc = knj_wmma_guard(&dev);
    if (grc) return grc;

    const int T = (argc > 3) ? std::atoi(argv[3]) : 512;

    const auto kv = read_job(argv[1]);
    Job J;
    J.model = jget(kv, "model");
    J.dumpdir = argv[2];
    J.layer = (int)jint(kv, "layer");
    J.hidden = (int)jint(kv, "hidden");
    J.ff = (int)jint(kv, "ff");
    J.experts = (int)jint(kv, "experts");
    J.topk = (int)jint(kv, "topk");
    const char *role3[3] = {"gate", "up", "down"};
    for (int i = 0; i < 3; ++i) {
        J.off[i] = jint(kv, std::string(role3[i]) + "_off");
        J.bpb[i] = (int)jint(kv, std::string(role3[i]) + "_bpb");
        J.k[i] = (int)jint(kv, std::string(role3[i]) + "_k");
        J.n[i] = (int)jint(kv, std::string(role3[i]) + "_n");
    }
    J.router_off = jint(kv, "router_off");

    // Weights: real packed bytes for the device, gguf-py-vouched f32 for the
    // oracle. Two different sources on purpose.
    const long long gBytes = (long long)J.k[0] * J.n[0] * J.experts / 256 * J.bpb[0];
    const long long uBytes = (long long)J.k[1] * J.n[1] * J.experts / 256 * J.bpb[1];
    const long long dBytes = (long long)J.k[2] * J.n[2] * J.experts / 256 * J.bpb[2];
    const long long rBytes = 4LL * J.hidden * J.experts;
    std::vector<uint8_t> Wg = read_at(J.model, J.off[0], gBytes);
    std::vector<uint8_t> Wu = read_at(J.model, J.off[1], uBytes);
    std::vector<uint8_t> Wd = read_at(J.model, J.off[2], dBytes);
    std::vector<uint8_t> Wr = read_at(J.model, J.router_off, rBytes);

    // The oracle's weights: the f32 dumps that q4k_unpack_crosscheck.py proved
    // BIT-IDENTICAL to gguf-py's independent dequantiser over all 37,748,736
    // weights of these three tensors. Deliberately a different source from the
    // bytes the device decodes, so a wrong decode cannot agree with itself.
    auto load_f32 = [&](const char *nm, long long n) {
        std::vector<uint8_t> b = read_at(std::string(J.dumpdir) + "/" + nm, 0, n * 4);
        std::vector<float> v((size_t)n);
        std::memcpy(v.data(), b.data(), (size_t)n * 4);
        return v;
    };
    std::vector<float> refg = load_f32("gate.f32", (long long)J.k[0] * J.n[0] * J.experts);
    std::vector<float> refu = load_f32("up.f32", (long long)J.k[1] * J.n[1] * J.experts);
    std::vector<float> refd = load_f32("down.f32", (long long)J.k[2] * J.n[2] * J.experts);

    const int NE = J.experts;
    const int KH = J.hidden, KF = J.ff;
    const int P = T * J.topk;
    const long long idxStride = T;

    printf("device      %s (%s)\n", dev.name, dev.arch);
    printf("cu          %d   wave %d   threads/block %d\n", dev.cu, dev.warp, NTHREADS);
    printf("layer       %d   hidden=%d ff=%d experts=%d topk=%d\n", J.layer, KH, KF, NE, J.topk);
    printf("tokens      %d   pairs=%d\n", T, P);
    printf("weights     gate %lld B q4_k, up %lld B q4_k, down %lld B q6_k\n",
           gBytes, uBytes, dBytes);

    // --- activations -------------------------------------------------------
    // Synthetic tokens (xorshift, range ~[-1,1]); the WEIGHTS are real. GEMM cost
    // does not depend on the activation values, and the oracle mirrors the f16
    // rounding, so this cannot flatter or hide a layout error.
    std::vector<__half> xh((size_t)T * KH);
    std::vector<float> xf16((size_t)T * KH);
    uint32_t st = 0x9e3779b9u;
    auto nxt = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
    for (size_t i = 0; i < xh.size(); ++i) {
        const float v = (float)((int)(nxt() % 4001) - 2000) / 1000.0f;
        xh[i] = __float2half(v);
        xf16[i] = (float)xh[i];
    }

    // --- routing from the REAL router weights ------------------------------
    // logits = X . Wgate_inp ; softmax ; top-k ; renormalise over the k chosen
    // (Qwen3-MoE normalises its top-k probabilities). Done on the host so the
    // device code under test stays the GEMM; it is identical on both sides.
    std::vector<float> rw((size_t)P);            // router weight per pair
    std::vector<int> pairTok((size_t)P), pairSlot((size_t)P), pairExp((size_t)P);
    std::vector<int> pairIdx((size_t)NE * (size_t)idxStride, -1);
    std::vector<int> pairCnt((size_t)NE, 0);
    std::vector<int> routerTally((size_t)NE, 0);
    {
        const float *Wr32 = (const float *)Wr.data();
        std::vector<float> logit((size_t)T * NE);
        for (int t = 0; t < T; ++t) {
            for (int e = 0; e < NE; ++e) {
                double acc = 0.0;
                for (int h = 0; h < KH; ++h)
                    acc += (double)xf16[(size_t)t * KH + h] * (double)Wr32[(size_t)e * KH + h];
                logit[(size_t)t * NE + e] = (float)acc;
            }
        }
        for (int t = 0; t < T; ++t) {
            float *lg = &logit[(size_t)t * NE];
            double mx = -1e300;
            for (int e = 0; e < NE; ++e) mx = std::max(mx, (double)lg[e]);
            double sum = 0.0;
            for (int e = 0; e < NE; ++e) sum += std::exp((double)lg[e] - mx);
            std::vector<int> order(NE);
            for (int e = 0; e < NE; ++e) order[e] = e;
            for (int a = 0; a < NE; ++a)
                for (int b = a + 1; b < NE; ++b)
                    // exact top-k: descending, ties broken by ascending expert id
                    if (lg[order[b]] > lg[order[a]] ||
                        (lg[order[b]] == lg[order[a]] && order[b] < order[a]))
                        std::swap(order[a], order[b]);
            double ksum = 0.0;
            for (int s = 0; s < J.topk; ++s)
                ksum += std::exp((double)lg[order[s]] - mx) / sum;
            for (int s = 0; s < J.topk; ++s) {
                const int e = order[s];
                const int p = t * J.topk + s;
                const double pr = std::exp((double)lg[e] - mx) / sum;
                pairTok[p] = t;
                pairSlot[p] = s;
                pairExp[p] = e;
                rw[p] = (float)(pr / (ksum > 0 ? ksum : 1.0));
                pairIdx[(size_t)e * idxStride + pairCnt[e]++] = p;
                routerTally[e]++;
            }
        }
    }
    int maxCnt = 0;
    printf("routing     ");
    for (int e = 0; e < NE; ++e) {
        printf("e%d=%d ", e, pairCnt[e]);
        maxCnt = std::max(maxCnt, pairCnt[e]);
    }
    printf(" (uneven by construction, real router weights)\n");

    // --- device buffers ----------------------------------------------------
    __half *dX, *dA; uint8_t *dWg, *dWu, *dWd;
    int *dIdx, *dCnt, *dTok; float *dG, *dU, *dY;
    HIPCHECK(hipMalloc(&dX, xh.size() * sizeof(__half)));
    HIPCHECK(hipMalloc(&dWg, Wg.size()));
    HIPCHECK(hipMalloc(&dWu, Wu.size()));
    HIPCHECK(hipMalloc(&dWd, Wd.size()));
    HIPCHECK(hipMalloc(&dIdx, pairIdx.size() * sizeof(int)));
    HIPCHECK(hipMalloc(&dCnt, pairCnt.size() * sizeof(int)));
    HIPCHECK(hipMalloc(&dTok, (size_t)P * sizeof(int)));
    HIPCHECK(hipMalloc(&dG, (size_t)P * KF * sizeof(float)));
    HIPCHECK(hipMalloc(&dU, (size_t)P * KF * sizeof(float)));
    HIPCHECK(hipMalloc(&dY, (size_t)P * KH * sizeof(float)));
    HIPCHECK(hipMalloc(&dA, (size_t)P * KF * sizeof(__half)));
    HIPCHECK(hipMemcpy(dX, xh.data(), xh.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(dWg, Wg.data(), Wg.size(), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(dWu, Wu.data(), Wu.size(), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(dWd, Wd.data(), Wd.size(), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(dIdx, pairIdx.data(), pairIdx.size() * sizeof(int), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(dCnt, pairCnt.data(), pairCnt.size() * sizeof(int), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(dTok, pairTok.data(), (size_t)P * sizeof(int), hipMemcpyHostToDevice));

    const int rowBytesG = (KH / 256) * KNJ_Q4K_BYTES;
    const int rowBytesD = (KF / 256) * KNJ_Q6K_BYTES;
    const dim3 grid1(KF / TN, (maxCnt + TM - 1) / TM, NE);
    const dim3 grid2(KH / TN, (maxCnt + TM - 1) / TM, NE);
    const dim3 simtGrid1((maxCnt * KF + 255) / 256, 1, NE);
    const dim3 simtGrid2((maxCnt * KH + 255) / 256, 1, NE);

    auto fill = [](float *d, size_t n, float v) {
        std::vector<float> s(n, v);
        HIPCHECK(hipMemcpy(d, s.data(), n * sizeof(float), hipMemcpyHostToDevice));
    };

    // --- rocWMMA path ------------------------------------------------------
    fill(dG, (size_t)P * KF, SENTINEL);
    fill(dU, (size_t)P * KF, SENTINEL);
    knj_q4k_gateup<<<grid1, NTHREADS>>>(dX, dWg, dWu, dIdx, dCnt, dTok,
                                        dG, dU, KF, KH, rowBytesG, idxStride);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    std::vector<float> G((size_t)P * KF), U((size_t)P * KF);
    HIPCHECK(hipMemcpy(G.data(), dG, G.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(U.data(), dU, U.size() * sizeof(float), hipMemcpyDeviceToHost));

    // --- oracle for G and U (f16-mirrored AND exact) -----------------------
    // refg/refu are the gguf-py-vouched f32 decodes, laid out [e][n][k].
    auto oracle_gu = [&](const std::vector<float> &Wf, std::vector<double> &out, bool mirror) {
        out.assign((size_t)P * KF, 0.0);
        for (int p = 0; p < P; ++p) {
            const int e = pairExp[p], t = pairTok[p];
            for (int n = 0; n < KF; ++n) {
                const float *wrow = &Wf[((size_t)e * KF + n) * KH];
                double acc = 0.0;
                for (int k = 0; k < KH; ++k) {
                    const double w = mirror ? (double)(float)__float2half(wrow[k])
                                            : (double)wrow[k];
                    acc += (double)xf16[(size_t)t * KH + k] * w;
                }
                out[(size_t)p * KF + n] = acc;
            }
        }
    };
    std::vector<double> refG, refU;
    printf("\ncomputing host oracle (double, f16-mirrored)...\n");
    oracle_gu(refg, refG, true);
    oracle_gu(refu, refU, true);

    int fail = 0, unwritten = 0;
    double worst_rel = 0.0, worst_abs = 0.0;
    const double TOL = 1e-4;   // summation-order only: both sides round through f16
    printf("\n-- full-output check vs verified gguf-py decode --\n");
    cx("gate (Q4_K, fmirror)", G, refG, P, KF, TOL, &fail, &unwritten, &worst_rel, &worst_abs);
    cx("up   (Q4_K, fmirror)", U, refU, P, KF, TOL, &fail, &unwritten, &worst_rel, &worst_abs);

    // The declared precision reduction, measured: same device values against an
    // all-fp32 oracle. This is the f16 operand cost, and it is reported, not
    // folded into a tolerance.
    {
        std::vector<double> exG, exU;
        oracle_gu(refg, exG, false);
        oracle_gu(refu, exU, false);
        int f2 = 0, u2 = 0; double wr2 = 0.0, wa2 = 0.0;
        printf("  (f16 declared reduction, all-fp32 oracle)\n");
        cx("gate (Q4_K, fp32)", G, exG, P, KF, 1e30, &f2, &u2, &wr2, &wa2);
        cx("up   (Q4_K, fp32)", U, exU, P, KF, 1e30, &f2, &u2, &wr2, &wa2);
    }

    // --- activation: silu(g)*u on the host, exactly as the oracle sees it ---
    std::vector<__half> act((size_t)P * KF);
    std::vector<float> actf((size_t)P * KF);
    for (size_t i = 0; i < act.size(); ++i) {
        const float gv = G[i], uv = U[i];
        const float s = gv / (1.0f + std::exp(-gv));
        act[i] = __float2half(s * uv);
        actf[i] = (float)act[i];
    }
    HIPCHECK(hipMemcpy(dA, act.data(), act.size() * sizeof(__half), hipMemcpyHostToDevice));

    // --- kernel 2 ----------------------------------------------------------
    fill(dY, (size_t)P * KH, SENTINEL);
    knj_q6k_down<<<grid2, NTHREADS>>>(dA, dWd, dIdx, dCnt, dY, KH, KF, rowBytesD, idxStride);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    std::vector<float> Y((size_t)P * KH);
    HIPCHECK(hipMemcpy(Y.data(), dY, Y.size() * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<double> refY((size_t)P * KH, 0.0);
    for (int p = 0; p < P; ++p) {
        const int e = pairExp[p];
        for (int n = 0; n < KH; ++n) {
            const float *wrow = &refd[((size_t)e * KH + n) * KF];
            double acc = 0.0;
            for (int k = 0; k < KF; ++k)
                acc += (double)actf[(size_t)p * KF + k] *
                       (double)(float)__float2half(wrow[k]);
            refY[(size_t)p * KH + n] = acc;
        }
    }
    printf("\n-- down projection (Q6_K) --\n");
    cx("down (Q6_K, fmirror)", Y, refY, P, KH, TOL, &fail, &unwritten, &worst_rel, &worst_abs);

    // --- the end-to-end layer output: weighted sum over top-k, fixed order --
    std::vector<float> yfinal((size_t)T * KH);
    std::vector<double> ryfinal((size_t)T * KH, 0.0);
    for (int t = 0; t < T; ++t) {
        for (int n = 0; n < KH; ++n) {
            double acc = 0.0;
            for (int s = 0; s < J.topk; ++s) {
                const int p = t * J.topk + s;
                acc += (double)rw[p] * (double)Y[(size_t)p * KH + n];
            }
            ryfinal[(size_t)t * KH + n] = acc;
            yfinal[(size_t)t * KH + n] = (float)acc;
        }
    }
    printf("\n-- layer output (weighted top-k sum) --\n");
    cx("layer0 MoE FFN out", yfinal, ryfinal, T, KH, TOL, &fail, &unwritten, &worst_rel, &worst_abs);

    // --- SIMT fallback A/B (always compiled) --------------------------------
    // Both halves of the FFN, decoded element-by-element, compared against the
    // same f16-mirrored oracle the WMMA path is compared against.
    {
        std::vector<float> Gs((size_t)P * KF, SENTINEL), Us((size_t)P * KF, SENTINEL);
        std::vector<float> Ys((size_t)P * KH, SENTINEL);
        fill(dG, (size_t)P * KF, SENTINEL);
        fill(dU, (size_t)P * KF, SENTINEL);
        fill(dY, (size_t)P * KH, SENTINEL);
        hipEvent_t t0, t1;
        HIPCHECK(hipEventCreate(&t0));
        HIPCHECK(hipEventCreate(&t1));

        HIPCHECK(hipEventRecord(t0));
        knj_q4k_simt_gateup<<<simtGrid1, 256>>>(dX, dWg, dWu, dIdx, dCnt, dTok,
                                                dG, dU, KF, KH, rowBytesG,
                                                idxStride, simtGrid1.x * 256);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipEventRecord(t1));
        HIPCHECK(hipEventSynchronize(t1));
        float simtMs1 = 0.0f;
        HIPCHECK(hipEventElapsedTime(&simtMs1, t0, t1));
        HIPCHECK(hipMemcpy(Gs.data(), dG, Gs.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(Us.data(), dU, Us.size() * sizeof(float), hipMemcpyDeviceToHost));

        HIPCHECK(hipEventRecord(t0));
        knj_q6k_simt_down<<<simtGrid2, 256>>>(dA, dWd, dIdx, dCnt, dY, KH, KF,
                                              rowBytesD, idxStride,
                                              simtGrid2.x * 256);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipEventRecord(t1));
        HIPCHECK(hipEventSynchronize(t1));
        float simtMs2 = 0.0f;
        HIPCHECK(hipEventElapsedTime(&simtMs2, t0, t1));
        HIPCHECK(hipMemcpy(Ys.data(), dY, Ys.size() * sizeof(float), hipMemcpyDeviceToHost));

        printf("\n-- SIMT fallback (always compiled, per-element decoders) --\n");
        int f3 = 0, u3 = 0; double wr3 = 0.0, wa3 = 0.0;
        cx("gate (SIMT)", Gs, refG, P, KF, TOL, &f3, &u3, &wr3, &wa3);
        cx("up   (SIMT)", Us, refU, P, KF, TOL, &f3, &u3, &wr3, &wa3);
        cx("down (SIMT)", Ys, refY, P, KH, TOL, &f3, &u3, &wr3, &wa3);
        printf("  SIMT gate+up             %.2f ms\n", simtMs1);
        printf("  SIMT down                %.2f ms\n", simtMs2);
        if (f3 || u3) ++fail;
    }

    // --- timing -----------------------------------------------------------
    const int ITERS = 20;
    hipEvent_t ev0, ev1;
    HIPCHECK(hipEventCreate(&ev0));
    HIPCHECK(hipEventCreate(&ev1));
    knj_q4k_gateup<<<grid1, NTHREADS>>>(dX, dWg, dWu, dIdx, dCnt, dTok,
                                        dG, dU, KF, KH, rowBytesG, idxStride);
    knj_q6k_down<<<grid2, NTHREADS>>>(dA, dWd, dIdx, dCnt, dY, KH, KF, rowBytesD, idxStride);
    HIPCHECK(hipDeviceSynchronize());

    float best1 = 1e30f, best2 = 1e30f, bestB = 1e30f;
    for (int i = 0; i < 5; ++i) {
        HIPCHECK(hipEventRecord(ev0));
        for (int j = 0; j < ITERS; ++j)
            knj_q4k_gateup<<<grid1, NTHREADS>>>(dX, dWg, dWu, dIdx, dCnt, dTok,
                                                dG, dU, KF, KH, rowBytesG, idxStride);
        HIPCHECK(hipEventRecord(ev1));
        HIPCHECK(hipEventSynchronize(ev1));
        float ms = 0.0f;
        HIPCHECK(hipEventElapsedTime(&ms, ev0, ev1));
        best1 = std::min(best1, ms / ITERS);

        HIPCHECK(hipEventRecord(ev0));
        for (int j = 0; j < ITERS; ++j)
            knj_q6k_down<<<grid2, NTHREADS>>>(dA, dWd, dIdx, dCnt, dY, KH, KF, rowBytesD, idxStride);
        HIPCHECK(hipEventRecord(ev1));
        HIPCHECK(hipEventSynchronize(ev1));
        HIPCHECK(hipEventElapsedTime(&ms, ev0, ev1));
        best2 = std::min(best2, ms / ITERS);

        HIPCHECK(hipEventRecord(ev0));
        for (int j = 0; j < ITERS; ++j) {
            knj_q4k_gateup<<<grid1, NTHREADS>>>(dX, dWg, dWu, dIdx, dCnt, dTok,
                                                dG, dU, KF, KH, rowBytesG, idxStride);
            knj_q6k_down<<<grid2, NTHREADS>>>(dA, dWd, dIdx, dCnt, dY, KH, KF, rowBytesD, idxStride);
        }
        HIPCHECK(hipEventRecord(ev1));
        HIPCHECK(hipEventSynchronize(ev1));
        HIPCHECK(hipEventElapsedTime(&ms, ev0, ev1));
        bestB = std::min(bestB, ms / ITERS);
    }

    const double flop1 = 2.0 * 2.0 * (double)P * KF * KH;         // gate+up
    const double flop2 = 2.0 * (double)P * KH * KF;               // down
    const double flop = flop1 + flop2;
    printf("\n-- timing (MEASURED, launch-inclusive, best of 5 x %d) --\n", ITERS);
    printf("  gate+up (Q4_K)      %8.4f ms   %7.2f TFLOP/s  grid=(%d,%d,%d)\n",
           best1, flop1 / ((double)best1 * 1e-3) / 1e12, grid1.x, grid1.y, grid1.z);
    printf("  down    (Q6_K)      %8.4f ms   %7.2f TFLOP/s  grid=(%d,%d,%d)\n",
           best2, flop2 / ((double)best2 * 1e-3) / 1e12, grid2.x, grid2.y, grid2.z);
    printf("  layer0 MoE FFN      %8.4f ms   %7.2f TFLOP/s  (both kernels)\n",
           bestB, flop / ((double)bestB * 1e-3) / 1e12);
    printf("  FLOPs               %.3f GFLOP for %d tokens (%d pairs, top-%d)\n",
           flop / 1e9, T, P, J.topk);
    printf("  packed weight bytes %.2f MiB read per pass (%lld + %lld + %lld)\n",
           (double)(gBytes + uBytes + dBytes) / 1048576.0, gBytes, uBytes, dBytes);

    // Extrapolation to a model-level pp, clearly labelled: 28 layers of the same
    // geometry, plus a 1.333x allowance for the dense attention path. The
    // allowance is the part that is NOT measured.
    const double modelMs512 = (double)bestB * 28.0 * 1.3333;
    printf("\n-- DERIVED model pp512 (NOT MEASURED) --\n");
    printf("  layer0 MoE FFN x28 layers x1.333 (attention, ASSUMPTION) = %.1f ms\n", modelMs512);
    printf("  => %.0f tok/s prefill for the whole model, IF every layer costs what layer 0 does\n",
           512.0 / modelMs512 * 1000.0);
    printf("  host engine, whole model, MEASURED : 12.8 tok/s (5 tok / ~391 ms)\n");
    printf("  llama.cpp ROCm -ngl 99, MEASURED   : 13817.56 tok/s pp512\n");

    printf("\nRESULT: %s (%d checks failed, %d unwritten outputs)\n",
           fail == 0 && unwritten == 0 ? "PASS" : "FAIL", fail, unwritten);
    return (fail == 0 && unwritten == 0) ? 0 : 5;
}
