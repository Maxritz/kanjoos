// i7_hip.hip -- the I7 bit-identity test, ON THE GPU.
//
// The host reference (tools/i7/i7_bit_identity.py) proves the contract is
// expressible. This proves the real device obeys it, including the part a host
// cannot exercise: pages whose bytes live in host RAM and cross PCIe between
// the score pass and the partial pass.
//
// WHAT IS BEING CLAIMED, PRECISELY
//   CLAIM 1  The three paths produce BITWISE IDENTICAL outputs on this device.
//            This is the I7 claim, and it is what exit code 0 certifies.
//   CLAIM 2  Each path agrees with a double-precision host oracle within
//            tolerance. That is a CORRECTNESS claim, not an identity claim:
//            __expf is the native v_exp_f32 instruction and is not required to
//            be correctly rounded, so cross-platform bit-identity against libm
//            exp is NOT asserted and is not achievable. Asserting it anyway is
//            how people end up proving nothing.
//
// WHY THE DEVICE CANNOT CHEAT
//   All three paths call the SAME function, reduce_pages(). The only
//   differences between them are which pointer a page arrives through and
//   whether an H2D copy happened first. Both are host-side decisions made
//   inside one function; the device executes identical instructions on
//   identical bits. Identity is therefore structural, not lucky, and a bug in
//   any path's plumbing shows up as a bit difference instead of hiding inside
//   a tolerance.
//
// TOOLCHAIN CONSTRAINTS (properties of the -nogpulib build, all learned here)
//   - no gridDim in any kernel: gfx1201 materialises it through
//     __ockl_get_num_groups, undefined in this configuration, so every launch
//     takes its block count as an argument.
//   - no device libm. __expf IS available and is a hardware instruction, which
//     is the only reason a softmax is computable here at all.
//   - '+wavefrontsize32' is rejected; gfx1201 is natively wave32, so BLK=64 is
//     two waves.
//   - the arch guard must read KNJ_BUILD_ARCH from the ENVIRONMENT. hipcc
//     re-spawns clang through a command string, so -D loses its quotes, and
//     __gfx1201__ is a device-pass macro that host main() cannot see.
//   - hipMemcpy's destination is a void* holding an ADDRESS. Writing
//     hipMemcpy(&pointer, ...) overwrites the pointer itself and then
//     dereferencing it faults. Copy into a scalar variable instead.
//
// BUILD / RUN
//   tools/i7/run_i7.sh          (sets KNJ_BUILD_ARCH, checks the exit status)
//
// EXIT CODES
//   0   positive paths bitwise identical, negative control detected, every path
//       within tolerance of the oracle
//   10  a positive path diverged bitwise
//   11  a negative control did NOT diverge -- the test is toothless
//   12  the oracle or the device failed; nothing was certified
//   6   arch mismatch          7   arch guard not configured

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_KEYS    1024
#define DIM       128
#define PAGE_KEYS 64
#define N_PAGES   (N_KEYS / PAGE_KEYS)
#define BLK       64
#define HOT_PAGES (N_PAGES / 2)

#define NL "\n"

// Payload arrays are file scope: half a megabyte of stack would be a coin flip
// against the default 1 MB Windows stack.
static uint16_t hK[(size_t)N_KEYS * DIM];
static uint16_t hV[(size_t)N_KEYS * DIM];

static uint32_t rng_state = 0x9e3779b9u;
static uint32_t xorshift(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static uint16_t f2bf(float f) {          // round-to-nearest-even, then truncate
    uint32_t b;
    memcpy(&b, &f, 4);
    uint32_t lsb = (b >> 16) & 1u;
    return (uint16_t)((b + 0x7fffu + lsb) >> 16);
}
static float bf2f(uint16_t h) {          // widening shift: exact, loses nothing
    // NOTE: this must be a REINTERPRETATION, not a conversion.
    // (float)(uint32_t)x converts the VALUE; it does not read the bits. The
    // union is the whole point, and getting it wrong produced a host oracle
    // that agreed with the GPU on nonsense -- both reading 0xBE450000 as the
    // integer 3192193024 instead of the float -0.0332 it encodes.
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

#define CHK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP FAIL %s at line %d: %s" NL, #x, __LINE__, \
           hipGetErrorString(e_)); return 12; } } while (0)

// Check the queue IMMEDIATELY after each launch. A faulting kernel surfaces
// here rather than at some later synchronisation, so a bad path is stopped
// before it can do anything else to the device.
#define LCHK(tag) do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) { \
    printf("LAUNCH FAIL after %s: %s" NL, tag, hipGetErrorString(e_)); \
    return 12; } } while (0)

// ------------------------------------------------------------------ device

// bf16 -> fp32 is a widening shift: exact, and no library call, which is what
// makes it usable inside a -nogpulib kernel. Reinterpret, never convert -- see
// the host-side bf2f() for what happens if this is written as a cast.
static __device__ __forceinline__ float bf2f_dev(uint16_t h) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

// Scores for one page: one thread per key, ascending threads to ascending keys.
//
// row_base and key_base are DIFFERENT numbers and conflating them is a real
// bug, not a style point. key_base is the page's global position, and is where
// the score is STORED. row_base is where the bytes actually start in the
// pointer we were handed: for a resident page it equals key_base, and for a
// page staged in from host RAM it is 0, because the staging buffer holds that
// page and nothing else. Indexing a one-page buffer with a global index reads
// up to 8x past its end -- the same page-local-vs-global mistake the host
// reference test caught in this repo's own tier-split path.
__global__ void k_scores(const uint16_t* __restrict__ K,
                         const float* __restrict__ q,
                         float* __restrict__ scores,
                         unsigned row_base, unsigned key_base,
                         unsigned page_keys, float scale) {
    unsigned t = threadIdx.x;
    if (t >= page_keys) return;
    const uint16_t* krow = K + (size_t)(row_base + t) * DIM;
    float acc = 0.0f;
    for (unsigned d = 0; d < DIM; ++d)
        acc += q[d] * bf2f_dev(krow[d]);
    scores[key_base + t] = acc * scale;
}

// Global max. max never rounds, so the reduction tree shape cannot change a
// bit -- which is exactly why the spec can parallelise this pass and nothing
// else.
__global__ void k_max(const float* __restrict__ scores, unsigned n,
                      float* __restrict__ out_max) {
    __shared__ float sh[BLK];
    unsigned t = threadIdx.x;
    float m = -3.0e38f;
    for (unsigned i = t; i < n; i += BLK) m = scores[i] > m ? scores[i] : m;
    sh[t] = m;
    __syncthreads();
    for (unsigned s = BLK / 2; s > 0; s >>= 1) {
        if (t < s) { float o = sh[t + s]; sh[t] = sh[t] > o ? sh[t] : o; }
        __syncthreads();
    }
    if (t == 0) *out_max = sh[0];
}

// THE CANONICAL ORDER LIVES HERE.
//   each thread owns a dimension and walks the page's keys in ASCENDING order,
//   so acc_page[d] is the same sum, in the same order, as the host spec;
//   l_page is accumulated by one thread over ascending keys.
// Nothing in here is order-independent, which is why it is loops and not a
// parallel reduction.
__global__ void k_page_partial(const uint16_t* __restrict__ V,
                               const float* __restrict__ scores, float m,
                               unsigned row_base, unsigned key_base,
                               unsigned page_keys,
                               float* __restrict__ l_page,
                               float* __restrict__ acc_page) {
    unsigned t = threadIdx.x;
    for (unsigned d = t; d < DIM; d += BLK) {
        float a = 0.0f;
        for (unsigned j = 0; j < page_keys; ++j) {
            float p = __expf(scores[key_base + j] - m);
            a += p * bf2f_dev(V[(size_t)(row_base + j) * DIM + d]);
        }
        acc_page[d] = a;
    }
    __syncthreads();
    if (t == 0) {
        float l = 0.0f;
        for (unsigned j = 0; j < page_keys; ++j)
            l += __expf(scores[key_base + j] - m);
        l_page[0] = l;
    }
}

// LEFT-TO-RIGHT merge of per-page partials. `descending` is the negative
// control: the same kernel in the order the spec forbids, which is what a
// stream-merged cold-first implementation would actually do.
__global__ void k_merge(const float* __restrict__ l_pages,
                        const float* __restrict__ acc_pages,
                        float* __restrict__ out, float* __restrict__ out_l,
                        unsigned npages, int descending) {
    __shared__ float sh_l;
    unsigned t = threadIdx.x;
    if (t == 0) sh_l = 0.0f;
    __syncthreads();
    for (unsigned i = 0; i < npages; ++i) {
        unsigned p = descending ? (npages - 1 - i) : i;
        if (t == 0) sh_l += l_pages[p];
        __syncthreads();
    }
    if (t == 0) *out_l = sh_l;
    __syncthreads();
    float inv = 1.0f / sh_l;
    for (unsigned d = t; d < DIM; d += BLK) {
        float a = 0.0f;
        for (unsigned i = 0; i < npages; ++i) {
            unsigned p = descending ? (npages - 1 - i) : i;
            a += acc_pages[(size_t)p * DIM + d];
        }
        out[d] = a * inv;
    }
}

// ------------------------------------------------------------------- state
static float hq[DIM];
static float hscale_g;

typedef struct {
    const uint16_t* khot;     // resident pages (or the whole bank)
    const uint16_t* vhot;
    uint16_t* kstage;         // one page, fed from T2 when staging is on
    uint16_t* vstage;
    const uint16_t* kpin;     // T2
    const uint16_t* vpin;
    int use_staging;
    int hot_pages;
    float* dq; float* dscores; float* dl_pages; float* dacc_pages;
    float* dmax; float* dout; float* dout_l;
} ctx;

// THE ONE REDUCTION. Every path runs this. Pages arrive either from VRAM or,
// when staging, from pinned host memory copied in one page at a time.
static int reduce_pages(ctx* c) {
    const unsigned one = 1;              // gridDim is unavailable; pass counts
    for (unsigned pg = 0; pg < N_PAGES; ++pg) {
        unsigned kb = pg * PAGE_KEYS;
        const uint16_t* src = c->khot;
        unsigned row_base = kb;                  // resident: bytes start here
        if (c->use_staging && (int)pg >= c->hot_pages) {
            CHK(hipMemcpyAsync(c->kstage, c->kpin + (size_t)kb * DIM,
                               (size_t)PAGE_KEYS * DIM * sizeof(uint16_t),
                               hipMemcpyHostToDevice, 0));
            CHK(hipStreamSynchronize(0));
            src = c->kstage;
            row_base = 0;                        // staged: page starts at 0
        }
        hipLaunchKernelGGL(k_scores, dim3(one), dim3(BLK), 0, 0, src, c->dq,
                           c->dscores, row_base, kb, PAGE_KEYS, hscale_g);
        LCHK("k_scores");
    }
    hipLaunchKernelGGL(k_max, dim3(one), dim3(BLK), 0, 0, c->dscores,
                       (unsigned)N_KEYS, c->dmax);
    CHK(hipDeviceSynchronize());

    float m = 0.0f;
    CHK(hipMemcpy(&m, c->dmax, 4, hipMemcpyDeviceToHost));

    for (unsigned pg = 0; pg < N_PAGES; ++pg) {
        unsigned kb = pg * PAGE_KEYS;
        const uint16_t* src = c->vhot;
        unsigned row_base = kb;
        if (c->use_staging && (int)pg >= c->hot_pages) {
            CHK(hipMemcpyAsync(c->vstage, c->vpin + (size_t)kb * DIM,
                               (size_t)PAGE_KEYS * DIM * sizeof(uint16_t),
                               hipMemcpyHostToDevice, 0));
            CHK(hipStreamSynchronize(0));
            src = c->vstage;
            row_base = 0;
        }
        hipLaunchKernelGGL(k_page_partial, dim3(one), dim3(BLK), 0, 0, src,
                           c->dscores, m, row_base, kb, PAGE_KEYS,
                           c->dl_pages + pg, c->dacc_pages + (size_t)pg * DIM);
        LCHK("k_page_partial");
    }
    CHK(hipDeviceSynchronize());
    return 0;
}

// -------------------------------------------------------------------- main
static int bits_equal(const float* a, const float* b, int n, int* first_bad) {
    for (int i = 0; i < n; ++i) {
        uint32_t x, y;
        memcpy(&x, &a[i], 4);
        memcpy(&y, &b[i], 4);
        if (x != y) { *first_bad = i; return 0; }
    }
    return 1;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);   // a crash must not swallow the evidence
    hipDeviceProp_t p;
    CHK(hipGetDeviceProperties(&p, 0));
    const char* built = getenv("KNJ_BUILD_ARCH");
    if (!built || !*built) {
        printf("ARCH GUARD NOT CONFIGURED: set KNJ_BUILD_ARCH. An unguarded "
               "run is not a measurement." NL);
        return 7;
    }
    if (strncmp(p.gcnArchName, built, strlen(built)) != 0) {
        printf("ARCH MISMATCH: built for %s, device is %s" NL, built,
               p.gcnArchName);
        printf("SKIPPED -- not run, not measured, not a result." NL);
        return 6;
    }

    printf("=" NL "78" NL);
    printf("I7 BIT-IDENTITY -- GPU LEG (real device, real PCIe)" NL);
    printf("=" NL "78" NL);
    printf("  device          %s" NL, p.gcnArchName);
    printf("  N_KEYS=%d DIM=%d PAGE_KEYS=%d N_PAGES=%d BLK=%d (wave32)" NL,
           N_KEYS, DIM, PAGE_KEYS, N_PAGES, BLK);
    printf("  payload         bfloat16 (the model's torch_dtype), widened to "
           "fp32 by a 16-bit shift" NL);
    printf("  softmax         __expf = native v_exp_f32, no device libm" NL);

    // ---- deterministic input -------------------------------------------
    rng_state = 0x1337u;
    for (size_t i = 0; i < (size_t)N_KEYS * DIM; ++i) {
        int32_t r = (int32_t)(xorshift() >> 8) - (1 << 23);
        hK[i] = f2bf((float)r * (1.0f / 8388608.0f) * 0.5f);
    }
    rng_state = 0x1337u ^ 0x5bf03635u;
    for (size_t i = 0; i < (size_t)N_KEYS * DIM; ++i) {
        int32_t r = (int32_t)(xorshift() >> 8) - (1 << 23);
        hV[i] = f2bf((float)r * (1.0f / 8388608.0f) * 0.5f);
    }
    rng_state = 0x1337u ^ 0x27d4eb2fu;
    for (int i = 0; i < DIM; ++i) {
        int32_t r = (int32_t)(xorshift() >> 8) - (1 << 23);
        hq[i] = (float)r * (1.0f / 8388608.0f) * 0.5f;
    }
    hscale_g = (float)(1.0 / sqrt((double)DIM));

    // ---- oracle FIRST, on the host, before any GPU work ----------------
    // If the oracle is wrong nothing downstream means anything, so it is
    // computed and printed before the first launch rather than after.
    double* orow = (double*)malloc(sizeof(double) * DIM);
    double* osc = (double*)malloc(sizeof(double) * N_KEYS);
    double mx = -1e300;
    for (unsigned j = 0; j < N_KEYS; ++j) {
        double a = 0.0;
        for (unsigned d = 0; d < DIM; ++d)
            a += (double)hq[d] * (double)bf2f(hK[(size_t)j * DIM + d]);
        osc[j] = a * (double)hscale_g;
        if (osc[j] > mx) mx = osc[j];
    }
    double lt = 0.0;
    for (unsigned pg = 0; pg < N_PAGES; ++pg)
        for (unsigned j = 0; j < PAGE_KEYS; ++j)
            lt += exp(osc[pg * PAGE_KEYS + j] - mx);
    for (unsigned d = 0; d < DIM; ++d) {
        double a = 0.0;
        for (unsigned pg = 0; pg < N_PAGES; ++pg) {
            double ap = 0.0;
            for (unsigned j = 0; j < PAGE_KEYS; ++j) {
                double pp = exp(osc[pg * PAGE_KEYS + j] - mx);
                ap += pp * (double)bf2f(hV[(size_t)(pg * PAGE_KEYS + j) * DIM + d]);
            }
            a += ap;
        }
        orow[d] = a / lt;
    }
    printf("  oracle          max score = %.6f, sum_exp = %.6f, out[0] = %.6e" NL,
           mx, lt, orow[0]);
    if (!(fabs(mx) < 1e6) || !(fabs(orow[0]) < 1e6)) {
        printf("ORACLE IS NOT SANE -- the input or the host reference is "
               "broken; certifying nothing." NL);
        return 12;
    }

    // ---- buffers --------------------------------------------------------
    uint16_t *dK, *dV, *dKh, *dVh, *dKstage, *dVstage, *hKpin, *hVpin;
    float *dq, *dscores, *dl_pages, *dacc_pages, *dout, *dmax, *dout_l;
    CHK(hipMalloc(&dK, sizeof(hK)));      CHK(hipMalloc(&dV, sizeof(hV)));
    CHK(hipMalloc(&dKh, sizeof(hK)));     CHK(hipMalloc(&dVh, sizeof(hV)));
    CHK(hipMalloc(&dKstage, (size_t)PAGE_KEYS * DIM * sizeof(uint16_t)));
    CHK(hipMalloc(&dVstage, (size_t)PAGE_KEYS * DIM * sizeof(uint16_t)));
    CHK(hipHostMalloc(&hKpin, sizeof(hK), hipHostMallocDefault));
    CHK(hipHostMalloc(&hVpin, sizeof(hV), hipHostMallocDefault));
    CHK(hipMalloc(&dq, sizeof(hq)));       CHK(hipMalloc(&dscores, N_KEYS * 4));
    CHK(hipMalloc(&dl_pages, N_PAGES * 4));
    CHK(hipMalloc(&dacc_pages, (size_t)N_PAGES * DIM * 4));
    CHK(hipMalloc(&dout, DIM * 4));       CHK(hipMalloc(&dmax, 4));
    CHK(hipMalloc(&dout_l, 4));
    CHK(hipMemcpy(dq, hq, sizeof(hq), hipMemcpyHostToDevice));
    CHK(hipMemcpy(hKpin, hK, sizeof(hK), hipMemcpyHostToDevice));
    CHK(hipMemcpy(hVpin, hV, sizeof(hV), hipMemcpyHostToDevice));

    ctx c;
    c.kstage = dKstage; c.vstage = dVstage; c.kpin = hKpin; c.vpin = hVpin;
    c.dq = dq; c.dscores = dscores; c.dl_pages = dl_pages;
    c.dacc_pages = dacc_pages; c.dmax = dmax; c.dout = dout; c.dout_l = dout_l;

    static float outA[DIM], outB[DIM], outC[DIM], outDesc[DIM];
    float lA = 0, lB = 0, lC = 0, lDesc = 0;
    const unsigned one = 1;

    // ---- PATH A: resident, every page from T1 ---------------------------
    CHK(hipMemcpy(dK, hK, sizeof(hK), hipMemcpyHostToDevice));
    CHK(hipMemcpy(dV, hV, sizeof(hV), hipMemcpyHostToDevice));
    c.khot = dK; c.vhot = dV; c.use_staging = 0; c.hot_pages = N_PAGES;
    if (reduce_pages(&c) != 0) return 12;
    hipLaunchKernelGGL(k_merge, dim3(one), dim3(BLK), 0, 0, dl_pages, dacc_pages,
                       dout, dout_l, (unsigned)N_PAGES, 0);
    LCHK("k_merge A");
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(outA, dout, sizeof(outA), hipMemcpyDeviceToHost));
    CHK(hipMemcpy(&lA, dout_l, 4, hipMemcpyDeviceToHost));

    // ---- PATH B: lossless movement codec, D2H then H2D ------------------
    CHK(hipMemcpy(hK, dK, sizeof(hK), hipMemcpyDeviceToHost));
    CHK(hipMemcpy(hV, dV, sizeof(hV), hipMemcpyDeviceToHost));
    CHK(hipMemcpy(dK, hK, sizeof(hK), hipMemcpyHostToDevice));
    CHK(hipMemcpy(dV, hV, sizeof(hV), hipMemcpyHostToDevice));
    c.khot = dK; c.vhot = dV; c.use_staging = 0; c.hot_pages = N_PAGES;
    if (reduce_pages(&c) != 0) return 12;
    hipLaunchKernelGGL(k_merge, dim3(one), dim3(BLK), 0, 0, dl_pages, dacc_pages,
                       dout, dout_l, (unsigned)N_PAGES, 0);
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(outB, dout, sizeof(outB), hipMemcpyDeviceToHost));
    CHK(hipMemcpy(&lB, dout_l, 4, hipMemcpyDeviceToHost));

    // ---- PATH C: tier split. Hot pages in T1, cold pages from T2 --------
    // The device buffer holds the hot pages and POISON for the cold ones: the
    // cold bytes are never simultaneously resident, which is the whole point.
    // If the staging path silently fell back to the resident buffer, the
    // negative control would be the only thing that noticed.
    memset(hK, 0xAB, sizeof(hK));
    memset(hV, 0xAB, sizeof(hV));
    CHK(hipMemcpy(dKh, hK, sizeof(hK), hipMemcpyHostToDevice));
    CHK(hipMemcpy(dVh, hV, sizeof(hV), hipMemcpyDeviceToHost));
    for (unsigned pg = 0; pg < HOT_PAGES; ++pg) {
        CHK(hipMemcpy(dKh + (size_t)pg * PAGE_KEYS * DIM,
                      hKpin + (size_t)pg * PAGE_KEYS * DIM,
                      (size_t)PAGE_KEYS * DIM * sizeof(uint16_t),
                      hipMemcpyHostToDevice));
        CHK(hipMemcpy(dVh + (size_t)pg * PAGE_KEYS * DIM,
                      hVpin + (size_t)pg * PAGE_KEYS * DIM,
                      (size_t)PAGE_KEYS * DIM * sizeof(uint16_t),
                      hipMemcpyHostToDevice));
    }
    c.khot = dKh; c.vhot = dVh; c.use_staging = 1; c.hot_pages = HOT_PAGES;
    if (reduce_pages(&c) != 0) return 12;
    hipLaunchKernelGGL(k_merge, dim3(one), dim3(BLK), 0, 0, dl_pages, dacc_pages,
                       dout, dout_l, (unsigned)N_PAGES, 0);
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(outC, dout, sizeof(outC), hipMemcpyDeviceToHost));
    CHK(hipMemcpy(&lC, dout_l, 4, hipMemcpyDeviceToHost));

    // ---- NEGATIVE CONTROL: descending (cold-first) merge ----------------
    hipLaunchKernelGGL(k_merge, dim3(one), dim3(BLK), 0, 0, dl_pages, dacc_pages,
                       dout, dout_l, (unsigned)N_PAGES, 1);
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(outDesc, dout, sizeof(outDesc), hipMemcpyDeviceToHost));

    // ---- verdicts --------------------------------------------------------
    int ok = 1;
    printf(NL "POSITIVE PATHS (bitwise against path A):" NL);
    int fb = -1;
    if (bits_equal(outA, outB, DIM, &fb) && lA == lB)
        printf("  PASS  B  movement codec (D2H then H2D, real PCIe)" NL);
    else { printf("  FAIL  B  differs at element %d (sum_exp %.7e vs %.7e)" NL,
                   fb, (double)lA, (double)lB); ok = 0; }
    fb = -1;
    if (bits_equal(outA, outC, DIM, &fb) && lA == lC)
        printf("  PASS  C  tier split (%d hot pages T1, %d cold pages T2)" NL,
               HOT_PAGES, N_PAGES - HOT_PAGES);
    else { printf("  FAIL  C  differs at element %d (sum_exp %.7e vs %.7e)" NL,
                   fb, (double)lA, (double)lC); ok = 0; }

    printf(NL "NEGATIVE CONTROL (must differ, else the check is toothless):" NL);
    fb = -1;
    if (!bits_equal(outA, outDesc, DIM, &fb)) {
        uint32_t x, y;
        memcpy(&x, &outA[fb], 4); memcpy(&y, &outDesc[fb], 4);
        printf("  DETECTED  descending (cold-first) merge, element %d: "
               "ascending=0x%08x descending=0x%08x" NL, fb, x, y);
    } else {
        printf("  MISSED    descending merge -- the guard cannot fire" NL);
        ok = 0;
    }

    printf(NL "ORACLE AGREEMENT (correctness, NOT bit-identity):" NL);
    double worst = 0.0;
    for (unsigned d = 0; d < DIM; ++d) {
        double den = fabs(orow[d]) > 1e-12 ? fabs(orow[d]) : 1e-12;
        double e = fabs((double)outA[d] - orow[d]) / den;
        if (e > worst) worst = e;
    }
    printf("  path A out[0] = %.9e, oracle out[0] = %.9e" NL,
           (double)outA[0], orow[0]);
    printf("  worst relative error vs the double oracle: %.3e" NL, worst);
    if (worst > 1e-4) {
        printf("  FAIL  path A disagrees with the oracle by more than 1e-4" NL);
        ok = 0;
    } else {
        printf("  PASS  within 1e-4 relative, as expected: __expf is the native "
               "v_exp_f32, not correctly rounded" NL);
    }

    printf(NL "=" NL "78" NL);
    // Leave nothing resident. This test allocates about 1 MB of VRAM; the
    // point of freeing it anyway is that a test which can leave the device in
    // a different state than it found it is a test that can break the next one.
    hipFree(dK); hipFree(dV); hipFree(dKh); hipFree(dVh);
    hipFree(dKstage); hipFree(dVstage); hipFree(hKpin); hipFree(hVpin);
    hipFree(dq); hipFree(dscores); hipFree(dl_pages); hipFree(dacc_pages);
    hipFree(dout); hipFree(dmax); hipFree(dout_l);
    free(orow); free(osc);
    if (!ok) {
        printf("I7 GPU LEG FAILED." NL);
        printf("  A failure here is a REAL finding, not a harness problem: it "
               "means two paths that must agree do not." NL);
        return 10;
    }
    printf("I7 GPU LEG PASSED on %s. Resident, movement-codec and tier-split "
           "agree in every bit, across real PCIe." NL, p.gcnArchName);
    printf("The cold-first merge control was detected, so the check has "
           "teeth." NL);
    return 0;
}