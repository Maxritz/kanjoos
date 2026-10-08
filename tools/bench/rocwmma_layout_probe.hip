// Kanjoos rocWMMA operand-layout probe -- gfx1201.
//
// THE QUESTION THIS ANSWERS
// -------------------------
// `rocwmma::load_matrix_sync` takes a `layout_t` for each operand, and the
// layout argument is a CLAIM ABOUT MEMORY, not a request to transpose. Get it
// wrong and nothing complains: the header compiles, the instruction is legal,
// the kernel launches, the numbers look like numbers. The only symptom is that
// every output is wrong.
//
// Concretely, for S = Q . K^T with Q and K stored [row][head_dim]:
//
//   A = Q   is M x K, element (m, d) at m*hd + d      -> row_major,   ld = hd
//   B = K^T is K x N, element (d, n) at n*hd + d      -> col_major,   ld = hd
//
// B is the trap. The logical contraction index of B is the head dimension, and
// "K^T" makes the memory layout column-major with ld = hd. Declaring B as
// row_major -- which reads like the natural spelling, because K's rows are
// contiguous -- makes element (k, n) resolve to k*hd + n, i.e. it reads the K
// matrix transposed. A half-transposed operand still produces a plausible
// magnitude, which is why this survived a first pass.
//
// So the probe measures all three spellings against a double-precision oracle
// and states which ones are exact. Mode 2 (col_major B) is the one the real
// attention kernel must use; Modes 0 and 1 are the controls that show the wrong
// spelling is detectable rather than harmless.
//
// This matters beyond this file: ai-coder/c2-runtime.md gates the whole
// gfx1201 WMMA path on "the operand-layout question (00 section 7.4a)". The
// hand-reverse-engineering of the _gfx12 builtin is superseded by rocWMMA, but
// the layout question is not -- it just moved from `what VGPR does lane L hold`
// to `which layout_t describes this memory`. This driver is the answer to the
// second form, measured.
//
// BUILD
//   hipcc --rocm-path="G:/ROCM10RT-gfx1201/lib/llvm" -std=c++17 -O2 \
//         --offload-arch=gfx1201 -I tools/bench \
//         tools/bench/rocwmma_layout_probe.hip -o /tmp/rocwmma_layout_probe.exe
// RUN
//   KNJ_BUILD_ARCH=gfx1201 /tmp/rocwmma_layout_probe.exe
// EXIT
//   0 all expected outcomes observed   5 an expectation failed
//   6 arch mismatch (SKIPPED)          7 KNJ_BUILD_ARCH unset

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

#include "knj_wmma_guard.h"

namespace rwx = rocwmma;

#define HC(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            printf("HIP %s\n", hipGetErrorString(e_));                         \
            return 9;                                                          \
        }                                                                      \
    } while (0)

static const int SEQ = 64;   // rows of Q and K
static const int HD  = 128;  // contraction length
static const int T   = 16;   // rocWMMA f16 tile
static const int TM  = 16 * 4;  // absolute tolerance vs the f32-chain oracle

// One S tile, A from global memory, B declared row_major.
__global__ void k_b_row(const __half *Q, const __half *K, float *C, int m, int n) {
    rwx::fragment<rwx::accumulator, T, T, T, float> acc;
    rwx::fill_fragment(acc, 0.0f);
    for (int d0 = 0; d0 < HD; d0 += T) {
        rwx::fragment<rwx::matrix_a, T, T, T, __half, rwx::row_major> fa;
        rwx::fragment<rwx::matrix_b, T, T, T, __half, rwx::row_major> fb;
        rwx::load_matrix_sync(fa, Q + (size_t)m * HD + d0, HD);
        rwx::load_matrix_sync(fb, K + (size_t)n * HD + d0, HD);
        rwx::mma_sync(acc, fa, fb, acc);
    }
    rwx::store_matrix_sync(C, acc, T, rwx::mem_row_major);
}

// Same tile, A and B staged through LDS. The routing cannot be the difference
// between this and k_b_row, so if the two disagree the fault is the layout
// claim, not the memory space.
__global__ void k_b_row_lds(const __half *Q, const __half *K, float *C, int m, int n) {
    __shared__ __align__(64) __half sQ[T * HD], sK[T * HD];
    for (int i = threadIdx.x; i < T * HD; i += blockDim.x) {
        sQ[i] = Q[(size_t)m * HD + i];
        sK[i] = K[(size_t)n * HD + i];
    }
    __syncthreads();
    rwx::fragment<rwx::accumulator, T, T, T, float> acc;
    rwx::fill_fragment(acc, 0.0f);
    for (int d0 = 0; d0 < HD; d0 += T) {
        rwx::fragment<rwx::matrix_a, T, T, T, __half, rwx::row_major> fa;
        rwx::fragment<rwx::matrix_b, T, T, T, __half, rwx::row_major> fb;
        rwx::load_matrix_sync(fa, sQ + d0, HD);
        rwx::load_matrix_sync(fb, sK + d0, HD);
        rwx::mma_sync(acc, fa, fb, acc);
    }
    rwx::store_matrix_sync(C, acc, T, rwx::mem_row_major);
}

// The correct spelling for a stored-[row][hd] K used as K^T.
__global__ void k_b_col(const __half *Q, const __half *K, float *C, int m, int n) {
    rwx::fragment<rwx::accumulator, T, T, T, float> acc;
    rwx::fill_fragment(acc, 0.0f);
    for (int d0 = 0; d0 < HD; d0 += T) {
        rwx::fragment<rwx::matrix_a, T, T, T, __half, rwx::row_major> fa;
        rwx::fragment<rwx::matrix_b, T, T, T, __half, rwx::col_major> fb;
        rwx::load_matrix_sync(fa, Q + (size_t)m * HD + d0, HD);
        rwx::load_matrix_sync(fb, K + (size_t)n * HD + d0, HD);
        rwx::mma_sync(acc, fa, fb, acc);
    }
    rwx::store_matrix_sync(C, acc, T, rwx::mem_row_major);
}

struct Mode {
    const char *name;
    const char *claim;
    bool        expect_exact;
    void (*fn)(const __half *, const __half *, float *, int, int);
};

int main() {
    knj_wmma_dev dev;
    const int g = knj_wmma_guard(&dev);
    if (g) return g;

    printf("device      %s\n", dev.name);
    printf("gcnArchName %s\n", dev.arch);
    printf("cu          %d   wave %d\n\n", dev.cu, dev.warp);

    std::vector<__half> hQ((size_t)SEQ * HD), hK((size_t)SEQ * HD);
    unsigned s = 7u;
    const auto rn = [&s]() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; };
    for (auto &x : hQ) x = __float2half((float)((int)(rn() % 2001) - 1000) / 1000.0f);
    for (auto &x : hK) x = __float2half((float)((int)(rn() % 2001) - 1000) / 1000.0f);

    __half *dQ, *dK;
    float  *dC;
    HC(hipMalloc(&dQ, hQ.size() * sizeof(__half)));
    HC(hipMalloc(&dK, hK.size() * sizeof(__half)));
    HC(hipMalloc(&dC, (size_t)T * T * sizeof(float)));
    HC(hipMemcpy(dQ, hQ.data(), hQ.size() * sizeof(__half), hipMemcpyHostToDevice));
    HC(hipMemcpy(dK, hK.data(), hK.size() * sizeof(__half), hipMemcpyHostToDevice));

    const Mode modes[3] = {
        {"B row_major, A from global", "WRONG -- reads K transposed",   false, k_b_row},
        {"B row_major, A and B in LDS", "WRONG -- memory space is not the cause", false, k_b_row_lds},
        {"B col_major, A from global", "EXACT -- the spelling to use",   true,  k_b_col},
    };

    int  total_bad[3] = {0, 0, 0};
    double total_worst[3] = {0, 0, 0};
    int  tiles = 0;

    std::vector<float> C((size_t)T * T);
    for (int m = 0; m < SEQ; m += T) {
        for (int n = 0; n < SEQ; n += T) {
            tiles++;
            for (int mi = 0; mi < 3; ++mi) {
                modes[mi].fn<<<1, 32>>>(dQ, dK, dC, m, n);
                HC(hipGetLastError());
                HC(hipDeviceSynchronize());
                HC(hipMemcpy(C.data(), dC, C.size() * sizeof(float), hipMemcpyDeviceToHost));
                for (int i = 0; i < T; ++i) {
                    for (int j = 0; j < T; ++j) {
                        double a = 0.0;
                        for (int k = 0; k < HD; ++k)
                            a += (double)__half2float(hQ[(size_t)(m + i) * HD + k]) *
                                 (double)__half2float(hK[(size_t)(n + j) * HD + k]);
                        const double e = std::fabs(a - (double)C[i * T + j]);
                        if (e > 1e-3) total_bad[mi]++;
                        total_worst[mi] = std::max(total_worst[mi], e);
                    }
                }
            }
        }
    }

    printf("%d tiles of %dx%dx%d, %d elements each, %d elements compared per mode\n\n",
           tiles, T, T, HD, T * T, tiles * T * T);

    bool ok = true;
    for (int mi = 0; mi < 3; ++mi) {
        const bool is_exact = (total_bad[mi] == 0);
        const bool agrees   = (is_exact == modes[mi].expect_exact);
        if (!agrees) ok = false;
        printf("  [%s] %-30s %s\n", agrees ? "ok" : "XX", modes[mi].name,
               modes[mi].claim);
        printf("        worst_abs %.3e   wrong %d / %d\n", total_worst[mi],
               total_bad[mi], tiles * T * T);
    }

    printf("\n  worst_abs is measured against a double-precision accumulation of\n");
    printf("  the same f16 inputs, so it carries f16 input rounding (~1e-3 for a\n");
    printf("  128-long dot of unit-scale operands) plus the f32 mma chain. 2e-6 is\n");
    printf("  the expected size for the correct spelling; 1.4e1 for the wrong one is\n");
    printf("  the same order as the answers themselves.\n");

    printf("\n  CONSEQUENCE: tools/bench/rocwmma_attn.hip declares B col_major in\n");
    printf("  its S = Q . K^T loop for exactly this reason, and its PV loop -- where\n");
    printf("  B is V as a real K x N row-major matrix, ld = head_dim -- keeps\n");
    printf("  row_major. Same file, two different layout_t arguments, both required.\n");

    printf("\nRESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 5;
}
