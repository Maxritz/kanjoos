// Kanjoos component trace - sherlock-it tier C instrumentation.
//
// Purpose: produce a per-component timing table in the docs/06-profiling.md
// section 2 output contract (component | ops | %dev | dev us | idle us |
// host us) over every component that ACTUALLY EXECUTES on this machine:
// HIP runtime calls, transfers, dispatch, the three measured chain kernels,
// the C2 arch guard, plus the instrumentation floor measured through the
// same begin/end path.
//
// This is a TRACE of the device layer and its microbenchmarks. It is not an
// inference trace: there is no engine yet (no C4 loader, C16 attention or
// C17 expert GEMM), so no attention/projections rows exist to print. Rows
// here are what the hardware is really asked to do today.
//
// BUILD (the only invocation that links on this machine):
//   hipcc -nogpulib -O2 --offload-arch=gfx1201 -DKNJ_TRACE_ARCH_ID=1201 \
//         -LG:/ROCM10RT-gfx1201/lib -lamdhip64 trace_components.hip -o t.exe
//
// The arch id is passed NUMERICALLY on purpose: -DKNJ_ARCH_NAME="gfx1201"
// loses its quotes through hipcc's command-string re-spelling (docs/00
// section 8.7), so the guard compares integers, never -D strings.
//
// EXIT CODES (repo contract, same as every other tier C driver)
//   0  ran, all correctness traps passed
//   1  HIP call failed or a correctness trap failed
//   6  DECLINED - arch mismatch (built vs requested vs attached device)
//   7  REFUSED - KNJ_BUILD_ARCH not set or not a gfx arch
//
// The compiled-vs-env cross-check and the WMMA row need
// -DKNJ_TRACE_ARCH_ID=<id> on the build line. run_bench.sh does NOT pass -D
// for the arch (hipcc re-spawns clang through a command string and a -D with
// quotes loses them; the env is the canonical channel - docs/00 section 8.7),
// so without the id the env-vs-device guard still runs and the wmma row is
// omitted. Same contract as expert_gemm.hip.

#include <hip/hip_runtime.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef KNJ_TRACE_ARCH_ID
#define KNJ_TRACE_ARCH_ID -1
#endif

// The wmma micro kernel uses the _gfx12 builtin; it is compiled only for the
// gfx1201 build so a gfx1031 build of this file still compiles (its row is
// then omitted rather than reported as a zero).
#if KNJ_TRACE_ARCH_ID == 1201
#define KNJ_HAVE_WMMA_MICRO 1
#else
#define KNJ_HAVE_WMMA_MICRO 0
#endif

// ---------------------------------------------------------------------------
// Kernels. Shapes match tools/bench/wmma_run.hip (the same 64-deep chains the
// compile census measures) so trace numbers and census numbers refer to the
// same code.
// ---------------------------------------------------------------------------

__global__ void knj_tr_empty(void) {}

__global__ void knj_tr_simt_f32(const float *A, const float *B, float *D) {
    float acc = 0.0f;
    for (int k = 0; k < 64; k++) acc += A[k] * B[k];
    *D = acc;
}

typedef _Float16 knj_tr_v2h __attribute__((ext_vector_type(2)));

__device__ static _Float16 knj_tr_bit_to_half(unsigned short h) {
    return __builtin_bit_cast(_Float16, h);
}

__global__ void knj_tr_simt_pk16(const unsigned short *A,
                                 const unsigned short *B, float *D) {
    float acc = 0.0f;
    for (int k = 0; k < 32; k++) {
        knj_tr_v2h a = {(_Float16)knj_tr_bit_to_half(A[2 * k]),
                        (_Float16)knj_tr_bit_to_half(A[2 * k + 1])};
        knj_tr_v2h b = {(_Float16)knj_tr_bit_to_half(B[2 * k]),
                        (_Float16)knj_tr_bit_to_half(B[2 * k + 1])};
        knj_tr_v2h r = {(_Float16)0.0f, (_Float16)0.0f};
        asm volatile("v_pk_fma_f16 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b));
        acc += (float)r.x + (float)r.y;
    }
    *D = acc;
}

#if KNJ_HAVE_WMMA_MICRO
typedef _Float16 knj_tr_v8h __attribute__((ext_vector_type(8)));
typedef float knj_tr_v8f __attribute__((ext_vector_type(8)));

__global__ void knj_tr_wmma_chain(const unsigned short *A,
                                  const unsigned short *B, float *D) {
    knj_tr_v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < 8; k++) {
        knj_tr_v8h a, b;
        __builtin_memcpy(&a, A + k * 8, sizeof(a));
        __builtin_memcpy(&b, B + k * 8, sizeof(b));
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    __builtin_memcpy(D, &acc, sizeof(acc));
}
#endif

// Bulk variant: every thread writes its own slot, so this measures sustained
// math/memory cost instead of a single contested scalar. No gridDim (gfx1201
// materialises it through the device runtime we do not link, docs/00 8.5).
__global__ void knj_tr_grid_f32(const float *A, const float *B, float *D,
                                int K) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    const float *a = A + (unsigned long)i * (unsigned)K;
    const float *b = B + (unsigned long)i * (unsigned)K;
    for (int k = 0; k < K; k++) acc += a[k] * b[k];
    D[i] = acc;
}

// HYPOTHESIS TEST for the bulk row: same grid, one FMA, no K loop. If this
// runs at the VRAM roofline while knj_tr_grid_f32 does not, the dot loop
// (memory-level parallelism / dependency chain) is the culprit; if both are
// equally slow, the problem is the dispatch or memory path, not the loop.
__global__ void knj_tr_grid_mul(const float *A, const float *B, float *D) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    D[i] = A[i] * B[i];
}

// THE FIX for the bulk row. knj_tr_grid_f32 gives each thread a ROW, so at
// each k step consecutive lanes are 256 B apart: every 64 B sector carries
// one useful 4 B dword (599 GB/s x 16/64 = ~37 GB/s, which is exactly what
// that kernel measured). Here the LANE owns k (lane j takes k = j, j+32, ...)
// so consecutive lanes read consecutive floats = one coalesced segment per
// step, and the 32 partial sums are reduced through LDS. No device lib, no
// shfl, no gridDim - s_barrier and LDS are intrinsics.
__global__ void knj_tr_grid_f32_c(const float *A, const float *B, float *D,
                                  int K) {
    __shared__ float s[64];
    unsigned lane = threadIdx.x & 31u;
    unsigned base = threadIdx.x - lane;
    unsigned row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    float acc = 0.0f;
    const float *a = A + (unsigned long)row * (unsigned)K;
    const float *b = B + (unsigned long)row * (unsigned)K;
    for (int k = (int)lane; k < K; k += 32) acc += a[k] * b[k];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < (unsigned)stride) s[base + lane] += s[base + lane + stride];
        __syncthreads();
    }
    if (lane == 0) D[row] = s[base];
}

// ---------------------------------------------------------------------------
// Timing plumbing
// ---------------------------------------------------------------------------

static double knj_now_us(void) {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch())
        .count();
}

#define HIPCHECK(x, what)                                                      \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            printf("  %s FAILED: %s\n", (what), hipGetErrorString(e_));        \
            printf("RESULT: FAIL\n");                                          \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static int knj_parse_gfx(const char *s) {
    if (!s || strncmp(s, "gfx", 3) != 0) return -1;
    return atoi(s + 3);
}

#define KNJ_MAX_ROWS 20
struct KnjTraceRow {
    const char *name;
    long ops;
    double dev_us;   // device elapsed via hipEvent, 0 for host-only rows
    double wall_us;  // wall for the whole batch, includes final sync
    double host_us;  // host time inside the API calls themselves
};
static KnjTraceRow g_rows[KNJ_MAX_ROWS];
static int g_nrows = 0;
static int g_fails = 0;

// NOTE ON UNITS: hipEventElapsedTime returns MILLISECONDS (float). Every
// device-row caller passes that raw value as dev_ms; it is converted to
// microseconds here exactly once. (Passing it through as "us" made an
// impossible 41 TB/s grid row and a 0.00 device floor - caught by the
// physical-plausibility check, not by the harness.)
static void knj_trace_add(const char *name, long ops, double dev_ms,
                          double wall_us, double host_us) {
    if (g_nrows >= KNJ_MAX_ROWS) return;
    g_rows[g_nrows].name = name;
    g_rows[g_nrows].ops = ops;
    g_rows[g_nrows].dev_us = dev_ms * 1000.0;
    g_rows[g_nrows].wall_us = wall_us;
    g_rows[g_nrows].host_us = host_us;
    g_nrows++;
}

static void knj_expect(bool ok, const char *what) {
    if (!ok) {
        printf("  TRAP FAILED: %s\n", what);
        g_fails++;
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(void) {
    // ARCH GUARD, first thing, before any device work.
    const char *env = getenv("KNJ_BUILD_ARCH");
    if (!env || !*env) {
        printf("REFUSED: KNJ_BUILD_ARCH is not set (unset = exit 7).\n");
        return 7;
    }
    const int env_id = knj_parse_gfx(env);
    if (env_id <= 0) {
        printf("REFUSED: KNJ_BUILD_ARCH is not a gfxNNNN arch: '%s'\n", env);
        return 7;
    }
#if KNJ_TRACE_ARCH_ID >= 0
    if (env_id != KNJ_TRACE_ARCH_ID) {
        printf("DECLINED: built for gfx%d, requested gfx%d. Skip, not a "
               "result.\n",
               KNJ_TRACE_ARCH_ID, env_id);
        return 6;
    }
#else
    /* Built without -DKNJ_TRACE_ARCH_ID (the bench runner's contract): the
     * env-vs-device guard below still refuses a mismatched run; only the
     * compiled-vs-env cross-check is unavailable. Never a hard error - a
     * driver that exits 1 here fails the whole tier C gate for everyone. */
#endif

    int dev = 0;
    HIPCHECK(hipGetDevice(&dev), "hipGetDevice");
    hipDeviceProp_t p;
    HIPCHECK(hipGetDeviceProperties(&p, dev), "hipGetDeviceProperties");

    const int dev_id = knj_parse_gfx(p.gcnArchName);
    if (dev_id != env_id) {
        printf("DECLINED: built for gfx%d, attached device is %s. Skip, not "
               "a result.\n",
               KNJ_TRACE_ARCH_ID, p.gcnArchName);
        return 6;
    }

    printf("  device  %s  gfx%d  %d CU  wave%d\n", p.name, dev_id,
           p.multiProcessorCount, p.warpSize);
    printf("  build   --offload-arch=gfx%d -O2 -nogpulib  (HIP 7.16)\n",
           KNJ_TRACE_ARCH_ID);
    printf("\n");

    // ---------------- setup (untraced) ----------------
    const size_t SZ = 256u << 20;      // 256 MB transfer/alloc payload
    const int GRID = 4096, BLK = 64, K = 64;
    const long N_MICRO = 2000, N_DISPATCH = 256, N_PROPS = 16,
               N_GUARD = 4096, N_SYNC = 32, N_XFER = 8, N_D2H = 4,
               N_ALLOC = 8;

    float *dA = nullptr, *dB = nullptr, *dD = nullptr, *dD2 = nullptr;
    HIPCHECK(hipMalloc(&dA, (size_t)GRID * BLK * K * sizeof(float)),
             "hipMalloc A");
    HIPCHECK(hipMalloc(&dB, (size_t)GRID * BLK * K * sizeof(float)),
             "hipMalloc B");
    HIPCHECK(hipMalloc(&dD, (size_t)GRID * BLK * sizeof(float)),
             "hipMalloc D");
    HIPCHECK(hipMalloc(&dD2, (size_t)GRID * BLK * sizeof(float)),
             "hipMalloc D2");
    // Varying inputs on purpose: with constant data every row's result is
    // identical, which would hide a row-mapping bug in the coalesced
    // equivalence trap below.
    {
        const size_t NELEM = (size_t)GRID * BLK * K;
        float *pa = (float *)malloc(NELEM * sizeof(float));
        float *pb = (float *)malloc(NELEM * sizeof(float));
        if (!pa || !pb) {
            printf("ERROR: host fill allocation failed\n");
            return 1;
        }
        for (size_t i = 0; i < NELEM; i++) {
            pa[i] = (float)(i % 17) + 1.0f;
            pb[i] = (float)(i % 13) + 1.0f;
        }
        HIPCHECK(hipMemcpy(dA, pa, NELEM * sizeof(float),
                           hipMemcpyHostToDevice),
                 "fill A");
        HIPCHECK(hipMemcpy(dB, pb, NELEM * sizeof(float),
                           hipMemcpyHostToDevice),
                 "fill B");
        free(pa);
        free(pb);
    }

    unsigned short hA[64], hB[64];
    for (int i = 0; i < 64; i++) { hA[i] = 0x3c00; hB[i] = 0x3c00; }  // 1.0f16
    float *fA = (float *)malloc(64 * sizeof(float));
    float *fB = (float *)malloc(64 * sizeof(float));
    float expect[64];
    for (int i = 0; i < 64; i++) { fA[i] = 1.0f; fB[i] = 1.0f; expect[i] = 64.0f; }

    // dedicated transfer buffer: dA is only 64 MB (the grid payload), the
    // transfer rows move SZ = 256 MB, so they get their own device buffer.
    float *dT = nullptr;
    HIPCHECK(hipMalloc(&dT, SZ), "hipMalloc transfer");

    unsigned short *dH = nullptr;
    float *dM = nullptr;
    float *dF32A = nullptr, *dF32B = nullptr;
    HIPCHECK(hipMalloc(&dH, 64 * sizeof(unsigned short)), "micro A");
    HIPCHECK(hipMalloc(&dF32A, 64 * sizeof(float)), "micro f32 A");
    HIPCHECK(hipMalloc(&dF32B, 64 * sizeof(float)), "micro f32 B");
    HIPCHECK(hipMemcpy(dF32A, fA, 64 * sizeof(float), hipMemcpyHostToDevice),
             "micro upload f32 A");
    HIPCHECK(hipMemcpy(dF32B, fB, 64 * sizeof(float), hipMemcpyHostToDevice),
             "micro upload f32 B");
    HIPCHECK(hipMalloc(&dM, 8 * sizeof(float)), "micro D");
    HIPCHECK(hipMemcpy(dH, hA, 64 * sizeof(unsigned short), hipMemcpyHostToDevice),
             "micro upload A");
    unsigned short *dHb = nullptr;
    HIPCHECK(hipMalloc(&dHb, 64 * sizeof(unsigned short)), "micro B2");
    HIPCHECK(hipMemcpy(dHb, hB, 64 * sizeof(unsigned short), hipMemcpyHostToDevice),
             "micro upload B");

    float *fM = (float *)malloc(64 * sizeof(float));

    hipEvent_t ev0, ev1;
    HIPCHECK(hipEventCreate(&ev0), "event 0");
    HIPCHECK(hipEventCreate(&ev1), "event 1");

    // =======================================================================
    // host-only components
    // =======================================================================

    // arch guard: the exact strcmp work the C2 guard does, scaled up.
    {
        double h0 = knj_now_us();
        volatile int sink = 0;
        for (long i = 0; i < N_GUARD; i++)
            sink += (strcmp(p.gcnArchName, env) == 0);
        double wall = knj_now_us() - h0;
        (void)sink;
        knj_trace_add("arch_guard", N_GUARD, 0.0, wall, wall);
    }

    // device properties query (the doctor/probe path).
    {
        double h0 = knj_now_us(), host = 0.0;
        hipDeviceProp_t q;
        for (long i = 0; i < N_PROPS; i++) {
            double t = knj_now_us();
            HIPCHECK(hipGetDeviceProperties(&q, dev), "props");
            host += knj_now_us() - t;
        }
        double wall = knj_now_us() - h0;
        knj_trace_add("device_props", N_PROPS, 0.0, wall, host);
    }

    // allocation trap: 256 MB device alloc/free.
    {
        double h0 = knj_now_us(), host = 0.0;
        for (long i = 0; i < N_ALLOC; i++) {
            float *t = nullptr;
            double t0 = knj_now_us();
            HIPCHECK(hipMalloc(&t, SZ), "trace hipMalloc");
            HIPCHECK(hipFree(t), "trace hipFree");
            host += knj_now_us() - t0;
        }
        double wall = knj_now_us() - h0;
        knj_trace_add("alloc_256mb", N_ALLOC, 0.0, wall, host);
    }

    // pinned (page-locked) alloc/free.
    {
        double h0 = knj_now_us(), host = 0.0;
        for (long i = 0; i < N_ALLOC; i++) {
            void *t = nullptr;
            double t0 = knj_now_us();
            HIPCHECK(hipHostMalloc(&t, SZ), "hipHostMalloc");
            HIPCHECK(hipHostFree(t), "hipHostFree");
            host += knj_now_us() - t0;
        }
        double wall = knj_now_us() - h0;
        knj_trace_add("alloc_pinned_256mb", N_ALLOC, 0.0, wall, host);
    }

    // =======================================================================
    // transfer trap: 256 MB pageable vs pinned H2D, pinned D2H
    // =======================================================================
    void *page_src = malloc(SZ);
    void *pinned_src = nullptr;
    void *d2h_dst = malloc(SZ);
    void *check_dst = malloc(SZ);
    if (!page_src || !d2h_dst || !check_dst) {
        printf("ERROR: host allocation failed\n");
        return 1;
    }
    memset(page_src, 0x5a, SZ);
    HIPCHECK(hipHostMalloc(&pinned_src, SZ), "hipHostMalloc staging");
    memcpy(pinned_src, page_src, SZ);

    {
        HIPCHECK(hipEventRecord(ev0, 0), "ev0");
        double h0 = knj_now_us(), host = 0.0;
        for (long i = 0; i < N_XFER; i++) {
            double t = knj_now_us();
            HIPCHECK(hipMemcpy(dT, page_src, SZ, hipMemcpyHostToDevice),
                     "H2D pageable");
            host += knj_now_us() - t;
        }
        HIPCHECK(hipEventRecord(ev1, 0), "ev1");
        HIPCHECK(hipDeviceSynchronize(), "sync xfer pageable");
        double wall = knj_now_us() - h0;
        float dev = 0.0f;
        HIPCHECK(hipEventElapsedTime(&dev, ev0, ev1), "elapsed pageable");
        knj_trace_add("h2d_pageable", N_XFER, dev, wall, host);
    }
    {
        HIPCHECK(hipEventRecord(ev0, 0), "ev0");
        double h0 = knj_now_us(), host = 0.0;
        for (long i = 0; i < N_XFER; i++) {
            double t = knj_now_us();
            HIPCHECK(hipMemcpy(dT, pinned_src, SZ, hipMemcpyHostToDevice),
                     "H2D pinned");
            host += knj_now_us() - t;
        }
        HIPCHECK(hipEventRecord(ev1, 0), "ev1");
        HIPCHECK(hipDeviceSynchronize(), "sync xfer pinned");
        double wall = knj_now_us() - h0;
        float dev = 0.0f;
        HIPCHECK(hipEventElapsedTime(&dev, ev0, ev1), "elapsed pinned");
        knj_trace_add("h2d_pinned", N_XFER, dev, wall, host);
    }
    {
        HIPCHECK(hipEventRecord(ev0, 0), "ev0");
        double h0 = knj_now_us(), host = 0.0;
        for (long i = 0; i < N_D2H; i++) {
            double t = knj_now_us();
            HIPCHECK(hipMemcpy(d2h_dst, dT, SZ, hipMemcpyDeviceToHost),
                     "D2H");
            host += knj_now_us() - t;
        }
        HIPCHECK(hipEventRecord(ev1, 0), "ev1");
        HIPCHECK(hipDeviceSynchronize(), "sync d2h");
        double wall = knj_now_us() - h0;
        float dev = 0.0f;
        HIPCHECK(hipEventElapsedTime(&dev, ev0, ev1), "elapsed d2h");
        knj_trace_add("d2h_pinned", N_D2H, dev, wall, host);
    }

    // ACCURACY TRAP (untraced): 1 MB pattern round-trip must be bit-identical.
    {
        const size_t CH = 1u << 20;
        unsigned char *pat = (unsigned char *)malloc(CH);
        for (size_t i = 0; i < CH; i++) pat[i] = (unsigned char)(i % 251);
        void *dst = malloc(CH);
        // dT, not dA: dA holds the grid input payload and must keep its
        // 0x3f fill for the liveness check below.
        HIPCHECK(hipMemcpy(dT, pat, CH, hipMemcpyHostToDevice), "pattern up");
        HIPCHECK(hipMemcpy(dst, dT, CH, hipMemcpyDeviceToHost), "pattern down");
        knj_expect(memcmp(pat, dst, CH) == 0, "1MB H2D/D2H round-trip identical");
        free(pat);
        free(dst);
    }

    // =======================================================================
    // dispatch trap: empty kernel launches, batched
    // =======================================================================
    {
        HIPCHECK(hipEventRecord(ev0, 0), "ev0");
        double h0 = knj_now_us(), host = 0.0;
        for (long i = 0; i < N_DISPATCH; i++) {
            double t = knj_now_us();
            hipLaunchKernelGGL(knj_tr_empty, dim3(1), dim3(64), 0, 0);
            host += knj_now_us() - t;
        }
        HIPCHECK(hipEventRecord(ev1, 0), "ev1");
        HIPCHECK(hipDeviceSynchronize(), "sync dispatch");
        double wall = knj_now_us() - h0;
        float dev = 0.0f;
        HIPCHECK(hipEventElapsedTime(&dev, ev0, ev1), "elapsed dispatch");
        knj_trace_add("dispatch_empty", N_DISPATCH, dev, wall, host);
    }

    // =======================================================================
    // kernel rows
    // =======================================================================
    {
        HIPCHECK(hipEventRecord(ev0, 0), "ev0");
        double h0 = knj_now_us(), host = 0.0;
        for (long i = 0; i < N_MICRO; i++) {
            double t = knj_now_us();
            hipLaunchKernelGGL(knj_tr_simt_f32, dim3(1), dim3(64), 0, 0,
                               dF32A, dF32B, dM);
            host += knj_now_us() - t;
        }
        HIPCHECK(hipEventRecord(ev1, 0), "ev1");
        HIPCHECK(hipDeviceSynchronize(), "sync micro f32");
        double wall = knj_now_us() - h0;
        float dev = 0.0f;
        HIPCHECK(hipEventElapsedTime(&dev, ev0, ev1), "elapsed micro f32");
        knj_trace_add("micro_simt_f32", N_MICRO, dev, wall, host);
    }
    {
        HIPCHECK(hipEventRecord(ev0, 0), "ev0");
        double h0 = knj_now_us(), host = 0.0;
        for (long i = 0; i < N_MICRO; i++) {
            double t = knj_now_us();
            hipLaunchKernelGGL(knj_tr_simt_pk16, dim3(1), dim3(64), 0, 0, dH,
                               dHb, dM);
            host += knj_now_us() - t;
        }
        HIPCHECK(hipEventRecord(ev1, 0), "ev1");
        HIPCHECK(hipDeviceSynchronize(), "sync micro pk16");
        double wall = knj_now_us() - h0;
        float dev = 0.0f;
        HIPCHECK(hipEventElapsedTime(&dev, ev0, ev1), "elapsed micro pk16");
        knj_trace_add("micro_simt_pk16", N_MICRO, dev, wall, host);
    }
#if KNJ_HAVE_WMMA_MICRO
    {
        HIPCHECK(hipEventRecord(ev0, 0), "ev0");
        double h0 = knj_now_us(), host = 0.0;
        for (long i = 0; i < N_MICRO; i++) {
            double t = knj_now_us();
            hipLaunchKernelGGL(knj_tr_wmma_chain, dim3(1), dim3(64), 0, 0, dH,
                               dHb, dM);
            host += knj_now_us() - t;
        }
        HIPCHECK(hipEventRecord(ev1, 0), "ev1");
        HIPCHECK(hipDeviceSynchronize(), "sync micro wmma");
        double wall = knj_now_us() - h0;
        float dev = 0.0f;
        HIPCHECK(hipEventElapsedTime(&dev, ev0, ev1), "elapsed micro wmma");
        knj_trace_add("micro_wmma_chain", N_MICRO, dev, wall, host);
    }
#endif
    {
        HIPCHECK(hipEventRecord(ev0, 0), "ev0");
        double h0 = knj_now_us(), host = 0.0;
        for (long i = 0; i < N_XFER; i++) {
            double t = knj_now_us();
            hipLaunchKernelGGL(knj_tr_grid_f32, dim3(GRID), dim3(BLK), 0, 0,
                               dA, dB, dD, K);
            host += knj_now_us() - t;
        }
        HIPCHECK(hipEventRecord(ev1, 0), "ev1");
        HIPCHECK(hipDeviceSynchronize(), "sync grid");
        double wall = knj_now_us() - h0;
        float dev = 0.0f;
        HIPCHECK(hipEventElapsedTime(&dev, ev0, ev1), "elapsed grid");
        knj_trace_add("grid_f32_bulk", N_XFER, dev, wall, host);
    }
    // SCALING ROW for the bulk kernel: same kernel, 16x fewer blocks. If the
    // per-launch cost is fixed overhead the two rows will be close; if it
    // tracks block count it is throughput (bandwidth or compute) bound.
    {
        HIPCHECK(hipEventRecord(ev0, 0), "ev0");
        double h0 = knj_now_us(), host = 0.0;
        for (long i = 0; i < N_XFER; i++) {
            double t = knj_now_us();
            hipLaunchKernelGGL(knj_tr_grid_f32, dim3(256), dim3(BLK), 0, 0,
                               dA, dB, dD, K);
            host += knj_now_us() - t;
        }
        HIPCHECK(hipEventRecord(ev1, 0), "ev1");
        HIPCHECK(hipDeviceSynchronize(), "sync grid256");
        double wall = knj_now_us() - h0;
        float dev = 0.0f;
        HIPCHECK(hipEventElapsedTime(&dev, ev0, ev1), "elapsed grid256");
        knj_trace_add("grid_f32_256blk", N_XFER, dev, wall, host);
    }
    {
        HIPCHECK(hipEventRecord(ev0, 0), "ev0");
        double h0 = knj_now_us(), host = 0.0;
        for (long i = 0; i < N_XFER; i++) {
            double t = knj_now_us();
            hipLaunchKernelGGL(knj_tr_grid_mul, dim3(GRID), dim3(BLK), 0, 0,
                               dA, dB, dD);
            host += knj_now_us() - t;
        }
        HIPCHECK(hipEventRecord(ev1, 0), "ev1");
        HIPCHECK(hipDeviceSynchronize(), "sync mul");
        double wall = knj_now_us() - h0;
        float dev = 0.0f;
        HIPCHECK(hipEventElapsedTime(&dev, ev0, ev1), "elapsed mul");
        knj_trace_add("grid_mul_4096blk", N_XFER, dev, wall, host);
    }
    {
        HIPCHECK(hipEventRecord(ev0, 0), "ev0");
        double h0 = knj_now_us(), host = 0.0;
        // 262144 rows (one per serial-thread) / 2 waves per block = 131072
        // blocks. An earlier GRID/2 covered only 4096 rows - the equivalence
        // trap caught it as exactly 262144-4096 mismatches.
        const int COAL_BLOCKS = GRID * BLK / 2;
        for (long i = 0; i < N_XFER; i++) {
            double t = knj_now_us();
            hipLaunchKernelGGL(knj_tr_grid_f32_c, dim3(COAL_BLOCKS),
                               dim3(64), 0, 0, dA, dB, dD2, K);
            host += knj_now_us() - t;
        }
        HIPCHECK(hipEventRecord(ev1, 0), "ev1");
        HIPCHECK(hipDeviceSynchronize(), "sync coalesced");
        double wall = knj_now_us() - h0;
        float dev = 0.0f;
        HIPCHECK(hipEventElapsedTime(&dev, ev0, ev1), "elapsed coalesced");
        knj_trace_add("grid_f32_coalesced", N_XFER, dev, wall, host);
    }

    // sync trap: cost of an explicit device sync on a drained queue.
    {
        HIPCHECK(hipEventRecord(ev0, 0), "ev0");
        double h0 = knj_now_us(), host = 0.0;
        for (long i = 0; i < N_SYNC; i++) {
            double t = knj_now_us();
            HIPCHECK(hipDeviceSynchronize(), "sync loop");
            host += knj_now_us() - t;
        }
        double wall = knj_now_us() - h0;
        knj_trace_add("sync_drained", N_SYNC, 0.0, wall, host);
    }

    // =======================================================================
    // accuracy traps: dead-kernel and value checks
    // =======================================================================
    hipLaunchKernelGGL(knj_tr_simt_f32, dim3(1), dim3(64), 0, 0, dF32A,
                       dF32B, dM);
    HIPCHECK(hipDeviceSynchronize(), "sync for check");
    HIPCHECK(hipMemcpy(fM, dM, sizeof(float), hipMemcpyDeviceToHost),
             "read f32 D");
    knj_expect(fM[0] == 64.0f, "micro simt f32 gives 64.0");

    hipLaunchKernelGGL(knj_tr_simt_pk16, dim3(1), dim3(64), 0, 0, dH, dHb, dM);
    HIPCHECK(hipDeviceSynchronize(), "sync for pk16 check");
    HIPCHECK(hipMemcpy(fM, dM, sizeof(float), hipMemcpyDeviceToHost),
             "read pk16 D");
    knj_expect(fM[0] == 64.0f, "micro simt pk16 gives 64.0");

#if KNJ_HAVE_WMMA_MICRO
    hipLaunchKernelGGL(knj_tr_wmma_chain, dim3(1), dim3(64), 0, 0, dH, dHb, dM);
    HIPCHECK(hipDeviceSynchronize(), "sync for wmma check");
    HIPCHECK(hipMemcpy(fM, dM, sizeof(float), hipMemcpyDeviceToHost),
             "read wmma D");
    knj_expect(fM[0] > 0.0f, "micro wmma chain is alive (D > 0)");
#endif

    {
        float *g = (float *)malloc((size_t)GRID * BLK * sizeof(float));
        hipLaunchKernelGGL(knj_tr_grid_f32, dim3(GRID), dim3(BLK), 0, 0, dA,
                           dB, dD, K);
        HIPCHECK(hipDeviceSynchronize(), "sync for grid check");
        HIPCHECK(hipMemcpy(g, dD, (size_t)GRID * BLK * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "read grid D");
        // A and B were memset to 0x3f3f3f3f; product is finite and > 0.
        knj_expect(g[0] > 0.0f, "grid f32 slot 0 alive");
        knj_expect(g[GRID * BLK - 1] > 0.0f, "grid f32 last slot alive");
        free(g);
    }
    {
        float *g = (float *)malloc((size_t)GRID * BLK * sizeof(float));
        hipLaunchKernelGGL(knj_tr_grid_mul, dim3(GRID), dim3(BLK), 0, 0, dA,
                           dB, dD);
        HIPCHECK(hipDeviceSynchronize(), "sync for mul check");
        HIPCHECK(hipMemcpy(g, dD, (size_t)GRID * BLK * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "read mul D");
        knj_expect(g[0] > 0.0f, "grid mul slot 0 alive");
        free(g);
    }
    // EQUIVALENCE TRAP: the coalesced kernel must reproduce the serial one.
    // rtol 1e-4 (float reassociation): a row-mapping bug moves results by
    // hundreds, so the tolerance cannot mask it. Inputs vary per element, so
    // rows differ and a permuted-row implementation cannot pass by luck.
    {
        const size_t NROW = (size_t)GRID * BLK;
        float *gr = (float *)malloc(NROW * sizeof(float));
        float *gn = (float *)malloc(NROW * sizeof(float));
        if (!gr || !gn) {
            printf("ERROR: equivalence buffers failed\n");
            return 1;
        }
        hipLaunchKernelGGL(knj_tr_grid_f32, dim3(GRID), dim3(BLK), 0, 0, dA,
                           dB, dD, K);
        HIPCHECK(hipDeviceSynchronize(), "sync serial");
        HIPCHECK(hipMemcpy(gr, dD, NROW * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "read serial");
        hipLaunchKernelGGL(knj_tr_grid_f32_c, dim3(GRID * BLK / 2),
                           dim3(64), 0, 0, dA, dB, dD2, K);
        HIPCHECK(hipDeviceSynchronize(), "sync coalesced check");
        HIPCHECK(hipMemcpy(gn, dD2, NROW * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "read coalesced");
        long bad = 0, differing_rows = 0;
        for (size_t i = 0; i < NROW; i++) {
            double ref = (double)gr[i];
            double dif = (double)gn[i] - ref;
            if (dif < 0) dif = -dif;
            double tol = 1e-4 * (ref < 0 ? -ref : ref) + 1e-3;
            if (dif > tol) bad++;
            if (gr[i] != gr[0]) differing_rows++;
        }
        knj_expect(bad == 0, "coalesced == serial within rtol 1e-4");
        // non-vacuity: the trap is only meaningful if rows actually differ
        knj_expect(differing_rows > 0, "input rows are distinguishable");
        printf("  equivalence: %ld mismatches, %ld/%zu rows differ from row 0\n",
               bad, differing_rows, NROW);
        free(gr);
        free(gn);
    }

    // =======================================================================
    // instrumentation floor: 256 empty ops through the same begin/end path
    // =======================================================================
    double host_floor = 0.0, dev_floor = 0.0;
    {
        double h0 = knj_now_us();
        for (int i = 0; i < 256; i++) {
            double a = knj_now_us();
            double b = knj_now_us();
            host_floor += (b - a);
        }
        host_floor /= 256.0;
        (void)h0;

        // device floor: one empty launch per op, each wrapped in its own
        // begin/end event pair, exactly the path real rows are timed through.
        double dsum = 0.0;
        for (int i = 0; i < 256; i++) {
            hipEventRecord(ev0, 0);
            hipLaunchKernelGGL(knj_tr_empty, dim3(1), dim3(64), 0, 0);
            hipEventRecord(ev1, 0);
            hipEventSynchronize(ev1);
            float e = 0.0f;
            hipEventElapsedTime(&e, ev0, ev1);  // milliseconds
            dsum += e;
        }
        dev_floor = dsum / 256.0 * 1000.0;  // ms -> us per op
    }

    // =======================================================================
    // output: the docs/06 section 2 contract
    // =======================================================================
    // sort by dev desc, host-only rows (dev 0) sink, matching the contract
    for (int i = 1; i < g_nrows; i++) {
        KnjTraceRow key = g_rows[i];
        int j = i - 1;
        while (j >= 0 && g_rows[j].dev_us < key.dev_us) {
            g_rows[j + 1] = g_rows[j];
            j--;
        }
        g_rows[j + 1] = key;
    }
    double dev_total = 0.0;
    for (int i = 0; i < g_nrows; i++) dev_total += g_rows[i].dev_us;

    printf("     component           ops   %%dev      dev us     idle us     host us\n");
    for (int i = 0; i < g_nrows; i++) {
        const KnjTraceRow &r = g_rows[i];
        double pct = dev_total > 0.0 ? 100.0 * r.dev_us / dev_total : 0.0;
        double idle = r.wall_us - r.dev_us - r.host_us;
        if (idle < 0.0) idle = 0.0;
        printf("     %-19s %5ld  %5.1f%%  %10.2f  %10.2f  %10.2f\n", r.name,
               r.ops, pct, r.dev_us, idle, r.host_us);
    }
    printf("   instrumentation floor, 256 empty ops timed through the same "
           "begin/end path: %.2f us host, %.2f us device each.\n",
           host_floor, dev_floor);
    printf("\n");
    if (g_fails > 0) {
        printf("RESULT: FAIL (%d traps)\n", g_fails);
        return 1;
    }
    printf("RESULT: PASS\n");
    return 0;
}
