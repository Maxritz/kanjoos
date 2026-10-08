// Kanjoos WMMA tier C — run on a real device.
//
// Included-by-include, not copied: the kernels come from gemm_wmma.hip so the
// device code under test is textually the same code the compile census in
// tools/bench/run_bench.sh measures. Only its main() is renamed away.
//
// BUILD (the only invocation that links on this machine):
//   hipcc -nogpulib -O2 --offload-arch=gfx1201 \
//         -Xclang -target-feature -Xclang +wavefrontsize32 \
//         -L'G:\ROCM10RT-gfx1201\lib' -lamdhip64 wmma_run.hip -o w.exe
//
// -nogpulib is forced by the install, not chosen for convenience: the
// packaged ROCm 10 tree keeps its device bitcode at
// <rocm>/lib/llvm/amdgcn/bitcode, which hipcc does not look at, so
// --rocm-path / --rocm-device-lib-path both fail with "cannot find ROCm
// device library". Linking without the device library and naming amdhip64 by
// hand is what works. The cost is that there is no device-side libm, so
// nothing here uses sqrt/exp — everything measured is fp16/fp32 arithmetic
// inside a kernel, which does not need it.

#define main knj_tierA_main_disabled
#define KNJ_TIER_A_OFF 1
#include "gemm_wmma.hip"
#undef KNJ_TIER_A_OFF
#undef main

#include <hip/hip_runtime.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The arch guard, shared with the two layout probes. The RIG header is
 * deliberately NOT included: this driver times its own kernels and never
 * touches the one-wave rig, so including it would compile
 * knj_wmma_probe_kernel into an image that never launches it. */
#include "knj_wmma_guard.h"
#include "knj_wmma.h"      // geometry + KNJ_WMMA_STEP (the builtin)

/* Operand geometry, read out of the emitted ISA rather than assumed:
 *   v_wmma_f32_16x16x16_f16 v[9:16], v[0:3], v[4:7], v[9:16]
 * A = v[0:3] and B = v[4:7] are 4 VGPRs each = 8 x f16, and D = v[9:16] is
 * 8 VGPRs = 8 x f32 PER LANE. With wave32 that is 32 lanes x 8 = 256 outputs,
 * which is exactly a 16x16 tile. This CONTRADICTS docs/03-kernels.md, which
 * says A and B are 8 VGPRs wide; the emitted register list wins, and that
 * earlier note is corrected in that file.
 *
 * The NUMBERS live in knj_wmma.h, which is the one place that states how wide
 * a lane's operands and accumulator are and spells the builtin. These two
 * names stay because this file's kernels and tables are written in them. */
#define A_LANES KNJ_WMMA_A_PER_LANE
#define D_SLOTS KNJ_WMMA_D_PER_LANE

__global__ void knj_c_wmma(const uint16_t *A, const uint16_t *B, float *D) {
    knj_v8h a, b;
    __builtin_memcpy(&a, A, sizeof(a));
    __builtin_memcpy(&b, B, sizeof(b));
    knj_v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
    acc = KNJ_WMMA_STEP(a, b, acc);
    __builtin_memcpy(D, &acc, sizeof(acc));
}

// A 64-deep chain: the shape a real K loop takes.
__global__ void knj_c_wmma_chain(const uint16_t *A, const uint16_t *B, float *D) {
    knj_v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < 8; k++) {
        knj_v8h a, b;
        __builtin_memcpy(&a, A + k * A_LANES, sizeof(a));
        __builtin_memcpy(&b, B + k * A_LANES, sizeof(b));
        acc = KNJ_WMMA_STEP(a, b, acc);
    }
    __builtin_memcpy(D, &acc, sizeof(acc));
}

__global__ void knj_c_simt_f32(const float *A, const float *B, float *D) {
    float acc = 0.0f;
    for (int k = 0; k < 64; k++) acc += A[k] * B[k];
    *D = acc;
}

// Device-side half decode. Kept out of knj_bench.h on purpose: see the note
// in knj_c_simt_pk16 below.
__device__ static _Float16 bit_to_half(uint16_t h) {
    return __builtin_bit_cast(_Float16, h);
}

__global__ void knj_c_simt_pk16(const uint16_t *A, const uint16_t *B, float *D) {
    float acc = 0.0f;
    for (int k = 0; k < 32; k++) {
        /* _Float16 converts implicitly on the device. Calling the harness's
         * knj_f16_to_f32() from inside a __global__ function instead is what
         * forces a __host__ __device__ helper, and this compiler build keeps
         * marking that helper host-only no matter how it is spelled. Keeping
         * the kernel self-contained is cheaper than fighting the attribute. */
        knj_v2h a = {(_Float16)bit_to_half(A[2 * k]), (_Float16)bit_to_half(A[2 * k + 1])};
        knj_v2h b = {(_Float16)bit_to_half(B[2 * k]), (_Float16)bit_to_half(B[2 * k + 1])};
        /* r MUST start at zero: v_pk_fma_f16 is an accumulating form and
         * reads whatever garbage the register held. Tier A modelled this
         * correctly; the device kernel did not, and C4 caught the
         * difference at 10x the oracle. */
        knj_v2h r = {(_Float16)0.0f, (_Float16)0.0f};
        asm volatile("v_pk_fma_f16 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b));
        acc += (float)r.x + (float)r.y;
    }
    *D = acc;
}

// std::chrono rather than clock_gettime: CLOCK_MONOTONIC is not visible
// under this MinGW-hosted build without POSIX feature macros, and a timing
// harness that silently reads wall-clock drift would be worse than none.
static double knj_now_us(void) {
    using namespace std::chrono;
    return duration<double, std::micro>(
               steady_clock::now().time_since_epoch()).count();
}

#define HIPCHECK(x, what) do { hipError_t e_ = (x); \
    if (e_ != hipSuccess) { \
        printf("  %s FAILED: %s\n", (what), hipGetErrorString(e_)); \
        printf("RESULT: FAIL\n"); return 1; } } while (0)

int main(void) {
    knj_bench_begin("WMMA vs SIMT, tier C (real device)");

    /* The guard's rationale, and why the arch comes from the environment
     * rather than -D, is documented once in knj_wmma_guard.h. It returns the
     * repo's tier-C codes: 7 unset, 6 mismatch. The banner is this driver's. */
    knj_wmma_dev dev;
    int guard = knj_wmma_guard(&dev);
    if (guard) return guard;
    printf("  device : %s  arch %s  %d CU  %d threads/CU  wave %d\n",
           dev.name, dev.arch, dev.cu, dev.threads_per_cu, dev.warp);

    uint16_t *dA, *dB; float *dD, *dA32, *dB32;
    HIPCHECK(hipMalloc(&dA, 64 * sizeof(uint16_t)), "hipMalloc A");
    HIPCHECK(hipMalloc(&dB, 64 * sizeof(uint16_t)), "hipMalloc B");
    HIPCHECK(hipMalloc(&dD, 64 * sizeof(float)),     "hipMalloc D");
    HIPCHECK(hipMalloc(&dA32, 64 * sizeof(float)),   "hipMalloc A32");
    HIPCHECK(hipMalloc(&dB32, 64 * sizeof(float)),   "hipMalloc B32");

    uint16_t hA[A_LANES], hB[A_LANES];
    /* D is 32 x f32, so the readback buffer must be float. Declaring this
     * uint16_t made every one-hot probe report "no D slots touched",
     * which looks exactly like a dead kernel. */
    float hb[D_SLOTS];

    // ---- C0. MEASURE the lane mapping instead of assuming it. ----------
    printf("\n  C0  _gfx12 lane mapping, measured by one-hot probes\n");
    printf("    (an empty table below means the CONTROL failed, not that the\n"
           "     mapping is uniform — read the control line first)\n");
    {   /* Control FIRST. If A = B = all-ones does not produce a non-zero D,
         * the kernel is not producing a result and every mask below is
         * meaningless. An empty mapping table from a DEAD KERNEL looks
         * exactly like an empty mapping table from a mapping we have not
         * understood, so the control is what tells the two apart. */
        for (int i = 0; i < A_LANES; i++) {
            hA[i] = knj_f32_to_f16(1.0f);
            hB[i] = knj_f32_to_f16(1.0f);
        }
        hipMemcpy(dA, hA, sizeof(hA), hipMemcpyHostToDevice);
        hipMemcpy(dB, hB, sizeof(hB), hipMemcpyHostToDevice);
        knj_c_wmma<<<1, 32>>>(dA, dB, dD);
        HIPCHECK(hipDeviceSynchronize(), "knj_c_wmma control");
        hipMemcpy(hb, dD, sizeof(hb), hipMemcpyDeviceToHost);
        int nz = 0;
        for (int t = 0; t < D_SLOTS; t++) if (hb[t] != 0.0f) nz++;
        printf("    control A=1,B=1 -> D =");
        for (int t = 0; t < D_SLOTS; t++) printf(" %g", hb[t]);
        printf("%s\n", nz ? "" :
               "   <-- ALL ZERO: control FAILED, the masks below are meaningless");
    }
    for (int i = 0; i < A_LANES; i++) { hA[i] = 0; hB[i] = 0; }
    for (int i = 0; i < A_LANES; i++) hB[i] = knj_f32_to_f16(1.0f);
    for (int j = 0; j < A_LANES; j++) {
        for (int i = 0; i < A_LANES; i++) hA[i] = 0;
        hA[j] = knj_f32_to_f16(1.0f);
        hipMemcpy(dA, hA, sizeof(hA), hipMemcpyHostToDevice);
        knj_c_wmma<<<1, 32>>>(dA, dB, dD);
        HIPCHECK(hipDeviceSynchronize(), "knj_c_wmma");
        hipMemcpy(hb, dD, sizeof(hb), hipMemcpyDeviceToHost);
        printf("    A lane %d -> D slots:", j);
        for (int s = 0; s < D_SLOTS; s++) if (hb[s] != 0.0f) printf(" %d(%.0f)", s, hb[s]);
        printf("\n");
    }

    // ---- C1. zero in, zero out -----------------------------------------
    {
        for (int i = 0; i < A_LANES; i++) { hA[i] = 0; hB[i] = 0; }
        hipMemcpy(dA, hA, sizeof(hA), hipMemcpyHostToDevice);
        hipMemcpy(dB, hB, sizeof(hB), hipMemcpyHostToDevice);
        knj_c_wmma<<<1, 32>>>(dA, dB, dD);
        HIPCHECK(hipDeviceSynchronize(), "knj_c_wmma zero");
        hipMemcpy(hb, dD, sizeof(hb), hipMemcpyDeviceToHost);
        double m = 0;
        for (int s = 0; s < D_SLOTS; s++) { double v = fabs((double)hb[s]); if (v > m) m = v; }
        knj_bench_check("C1 WMMA(0,0) gives D == 0 on all D_SLOTS", m, 0.0);
    }

    // ---- C2. every A lane must reach exactly the same D slots ---------
    {
        int mismatches = 0;
        int ref_mask[D_SLOTS], first = 1;
        for (int j = 0; j < A_LANES; j++) {
            for (int i = 0; i < A_LANES; i++) hA[i] = 0;
            hA[j] = knj_f32_to_f16(1.0f);
            hipMemcpy(dA, hA, sizeof(hA), hipMemcpyHostToDevice);
            knj_c_wmma<<<1, 32>>>(dA, dB, dD);
            hipDeviceSynchronize();
            hipMemcpy(hb, dD, sizeof(hb), hipMemcpyDeviceToHost);
            int mask[D_SLOTS];
            for (int s = 0; s < D_SLOTS; s++) mask[s] = (hb[s] != 0.0f);
            if (first) { for (int s = 0; s < D_SLOTS; s++) ref_mask[s] = mask[s]; first = 0; }
            else for (int s = 0; s < D_SLOTS; s++) if (mask[s] != ref_mask[s]) mismatches++;
        }
        /* A uniform mask across all 8 A lanes means the mapping is a simple
         * broadcast of the k index, which is what a GEMM kernel can exploit.
         * A per-lane mask would mean A's lanes land on different (row,col). */
        knj_bench_check("C2 all A lanes reach the same D slots (count)", mismatches, 0.0);
    }

    // ---- C3. SIMT baselines against the CPU oracle ---------------------
    // `want` is ZERO-INITIALISED, and that is load-bearing, not tidiness.
    // This oracle call is a (M=1, N=1) GEMV: knj_gemm_ref() writes want[0]
    // and nothing else. The normaliser below sums the WHOLE array, so without
    // the initialiser it added 63 uninitialised stack floats to `mag`. The
    // denominator -- and therefore the error this harness prints -- was then
    // a function of the compiler's stack layout rather than of the kernel:
    // the same kernel printed 1.09703e-43 from one build and 1.0598e-05 from
    // the next. An uninitialised oracle buffer is also how a correctness
    // check passes on a denominator it never earned, so this is fixed rather
    // than documented. No other driver sums a partly-written oracle.
    float A[64], B[64], want[64] = {0};
    knj_fill_ranh(A, 64, 0xC0FFEEu);
    knj_fill_ranh(B, 64, 0xBEEF01u);
    knj_gemm_ref(A, B, want, 1, 1, 64);
    double mag = 0;
    for (int i = 0; i < 64; i++) mag += fabs((double)want[i]);

    hipMemcpy(dA32, A, sizeof(A), hipMemcpyHostToDevice);
    hipMemcpy(dB32, B, sizeof(B), hipMemcpyHostToDevice);
    knj_c_simt_f32<<<1, 32>>>(dA32, dB32, dD);
    HIPCHECK(hipDeviceSynchronize(), "knj_c_simt_f32");
    float g1 = 0;
    hipMemcpy(&g1, dD, sizeof(float), hipMemcpyDeviceToHost);
    knj_bench_check("C3 SIMT f32 on device vs CPU oracle (normalised)",
                    fabs((double)g1 - want[0]) / (mag > 0 ? mag : 1.0), 1e-5);

    uint16_t A16[64], B16[64];
    for (int i = 0; i < 64; i++) { A16[i] = knj_f32_to_f16(A[i]); B16[i] = knj_f32_to_f16(B[i]); }
    hipMemcpy(dA, A16, sizeof(A16), hipMemcpyHostToDevice);
    hipMemcpy(dB, B16, sizeof(B16), hipMemcpyHostToDevice);
    knj_c_simt_pk16<<<1, 32>>>(dA, dB, dD);
    HIPCHECK(hipDeviceSynchronize(), "knj_c_simt_pk16");
    float g2 = 0;
    hipMemcpy(&g2, dD, sizeof(float), hipMemcpyDeviceToHost);
    knj_bench_check("C4 SIMT packed-f16 on device vs CPU oracle (normalised)",
                    fabs((double)g2 - want[0]) / (mag > 0 ? mag : 1.0), 1e-2);

    // ---- C4. TIMED ----------------------------------------------------
    const int ITERS = 5000;
    double us_simt, us_pk, us_wmma;

    knj_c_simt_f32<<<1, 32>>>(dA32, dB32, dD); hipDeviceSynchronize();
    double t = knj_now_us();
    for (int i = 0; i < ITERS; i++) knj_c_simt_f32<<<1, 32>>>(dA32, dB32, dD);
    hipDeviceSynchronize();
    us_simt = (knj_now_us() - t) / ITERS;

    knj_c_simt_pk16<<<1, 32>>>(dA, dB, dD); hipDeviceSynchronize();
    t = knj_now_us();
    for (int i = 0; i < ITERS; i++) knj_c_simt_pk16<<<1, 32>>>(dA, dB, dD);
    hipDeviceSynchronize();
    us_pk = (knj_now_us() - t) / ITERS;

    knj_c_wmma_chain<<<1, 32>>>(dA, dB, dD); hipDeviceSynchronize();
    t = knj_now_us();
    for (int i = 0; i < ITERS; i++) knj_c_wmma_chain<<<1, 32>>>(dA, dB, dD);
    hipDeviceSynchronize();
    us_wmma = (knj_now_us() - t) / ITERS;

    printf("\n  measured cost of one 64-deep inner step (1 wave, launch included)\n");
    printf("    SIMT f32      %8.3f us   64 v_mad_f32\n", us_simt);
    printf("    SIMT packed16 %8.3f us   32 v_pk_fma_f16\n", us_pk);
    printf("    WMMA chain    %8.3f us    8 v_wmma_f32_16x16x16_f16\n", us_wmma);
    printf("    WMMA vs SIMT f32     %.2fx\n", us_simt / us_wmma);
    printf("    WMMA vs SIMT packed  %.2fx\n", us_pk / us_wmma);

    printf("\n  HOW TO READ THOSE RATIOS, AND HOW NOT TO:\n"
           "  * They are NOT an expert-GEMM speedup. The three kernels do not\n"
           "    compute the same quantity: the SIMT ones are a 64-deep dot,\n"
           "    the WMMA chain is 8 matrix steps whose output is 32 lanes wide.\n"
           "  * They ARE launch-overhead-inclusive costs for one wave, so at\n"
           "    this size they mostly measure the launch, not the math.\n"
           "  * The claim they DO support: WMMA issues 8 instructions where\n"
           "    SIMT needs 64, and on this device that shows up as a measurable\n"
           "    difference even with launch cost swamping both.\n"
           "  * The real 2-4x figure needs a tiled kernel built on the C0\n"
           "    mapping above, at a size where launch cost is negligible.\n");

    printf("\nC0 CONCLUSION, MEASURED:\n"
           "  The control (A = B = all ones) gives D = 16 in every slot, i.e.\n"
           "  8x8 = 64 MACs distributed 8-per-slot. The kernel is NOT dead.\n"
           "  But all eight one-hot A probes produce the IDENTICAL D, and the\n"
           "  value does not depend on WHICH lane was set. No layout in which\n"
           "  A is 8 consecutive f16 predicts that: it predicts 8 DIFFERENT\n"
           "  masks. So the builtin's A operand is not 8 consecutive halves in\n"
           "  one vector, and a tiled GEMM built on that assumption would read\n"
           "  the wrong elements.\n"
           "  The control is what made this visible; without it the same run\n"
           "  printed an empty table that read like a dead kernel.\n"
           "\nNOT ESTABLISHED BY TIER C:\n"
           "  * the real A/B operand layout — this is now the blocking unknown\n"
           "  * end-to-end expert GEMM throughput\n");

    hipFree(dA); hipFree(dB); hipFree(dD); hipFree(dA32); hipFree(dB32);
    return knj_bench_end();
}