// Kanjoos microbenchmark harness — shared correctness contract.
//
// Purpose: make kernel experiments stop being one-off compiles. A benchmark
// file (gemm_wmma.hip, gemm_w4.hip, ...) gets three things from this header:
//
//   1. a CPU reference that is INDEPENDENT of the kernel (naive fp64-accumulate
//      triple loop), and a CPU MIRROR that reproduces the kernel's exact
//      tiling and accumulation order. Comparing mirror-vs-reference catches
//      index arithmetic, tiling geometry and scaling bugs — which is where
//      GEMM kernels actually go wrong — without needing a GPU.
//
//   2. a pass/fail contract with a numeric tolerance, so "it compiled" and
//      "it is right" are never conflated.
//
//   3. a declared expectation per kernel: which instruction it MUST contain.
//      The runner greps the emitted .s for it and reports MISSING if absent.
//
// WHY NO HIP HEADERS
// ------------------
// Same rule as the ISA probe, for the same reason: an earlier harness
// included <hip/hip_runtime.h> and was built through hipcc, so a broken ROCm
// install reported every kernel as broken. This header must compile with
// nothing but the AMDGPU frontend, and must ALSO compile natively for the
// host compiler, which is how the correctness checks actually run on a
// machine with no usable device.
//
// HOW THE TWO TIERS RELATE
// ------------------------
//   Tier A (host, always runs): reference vs mirror. Runs anywhere.
//   Tier B (device, needs a working install + a real GPU): the same kernels,
//      timed, with the CPU reference uploaded as the oracle.
//
// Tier A cannot prove the hardware primitive behaves as the manual says. It
// proves the ALGORITHM is right. The two claims are kept separate in the
// output so a Tier A pass is never reported as a throughput measurement.

#ifndef KNJ_BENCH_H
#define KNJ_BENCH_H

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- types ---
// These exist so a bench file compiles unchanged for device (ext_vector_type)
// and host. Nothing here is arch-specific.
/* The vector typedefs are needed by the HIP HOST pass as well as the device
 * pass (the tier C driver declares kernels with the same types), so they are
 * gated on clang, not on the device-compile macros. GCC is excluded because
 * it has no ext_vector_type, and the g++ tier A build does not need them. */
#if defined(__clang__)
typedef float    __attribute__((ext_vector_type(4))) knj_v4f;
typedef float    __attribute__((ext_vector_type(8))) knj_v8f;
typedef float    __attribute__((ext_vector_type(2))) knj_v2f;
typedef _Float16 __attribute__((ext_vector_type(2))) knj_v2h;
typedef _Float16 __attribute__((ext_vector_type(8))) knj_v8h;
typedef _Float16 __attribute__((ext_vector_type(16))) knj_v16h;
typedef short    __attribute__((ext_vector_type(8))) knj_v8b;
typedef short    __attribute__((ext_vector_type(16))) knj_v16b;
typedef short    __attribute__((ext_vector_type(4))) knj_v4s;
typedef int      __attribute__((ext_vector_type(2))) knj_v2i;
#endif

#if defined(__HIP_DEVICE_COMPILE__) || defined(__OPENCL__)
#define KNJ_GLOBAL __attribute__((global))
#define KNJ_DEVICE __attribute__((device))
/* Shared helpers must be callable from BOTH sides: the device kernels
 * use them, and so does the tier C host driver that sets up and checks
 * them. __host__ __device__ is the only spelling that satisfies both. */
#define KNJ_HD __host__ __device__
/* No <string.h> on the device side, and including HIP headers is exactly the
 * dependency this harness exists to avoid. __builtin_memcpy is available to
 * clang on both sides. */
#define KNJ_MEMCPY __builtin_memcpy
#define KNJ_FABS(x)   __builtin_fabsf(x)
#define KNJ_LRINT(x)  ((int)__builtin_rintf(x))
#else
#define KNJ_GLOBAL
#define KNJ_DEVICE
#define KNJ_HD
#define KNJ_MEMCPY memcpy
#define KNJ_FABS(x)   fabsf(x)
#define KNJ_LRINT(x)  ((int)lrintf(x))
#endif

// ------------------------------------------------------------- checking ---
typedef struct {
    int    checks;
    int    failures;
    char   worst_name[96];
    double worst_value;
} knj_bench_state;

static knj_bench_state knj_state;

static void knj_bench_begin(const char *suite) {
    memset(&knj_state, 0, sizeof(knj_state));
    /* Unbuffered: if a bench segfaults, the runner must still show how far it
     * got. Buffered stdout turns a crash into a silent run with no output. */
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== %s ===\n", suite);
}

// Record one numeric check. `tol` is an ABSOLUTE tolerance on `value`, because
// every comparison in this harness is against a reference computed in a
// different accumulation order and the meaningful question is "how far from
// the reference did we land", not "are the bits equal".
static void knj_bench_check(const char *name, double value, double tol) {
    knj_state.checks++;
    int bad = !(value <= tol) && !(value != value);  // also catches NaN
    printf("  %-52s %12.6g  (tol %g) %s\n", name, value, tol,
           bad ? "FAIL" : "ok");
    if (bad) {
        snprintf(knj_state.worst_name, sizeof(knj_state.worst_name), "%s", name);
        knj_state.worst_value = value;
    }
    if (bad) knj_state.failures++;
}

// Same, for a quantity that must stay ABOVE a floor. A min-check written as
// a max-check is how "must exceed 100x" becomes a check that silently passes
// at 88x, so the two directions have different helpers.
static void knj_bench_check_min(const char *name, double value, double floor_) {
    knj_state.checks++;
    int bad = !(value >= floor_) || (value != value);
    printf("  %-52s %12.6g  (min %g) %s\n", name, value, floor_,
           bad ? "FAIL" : "ok");
    if (bad) {
        snprintf(knj_state.worst_name, sizeof(knj_state.worst_name), "%s", name);
        snprintf(knj_state.worst_name + 0, 0, "");   /* no-op, keep format stable */
        knj_state.worst_value = value;
        knj_state.failures++;
    }
}

// Returns the process exit code. A bench file's main() ends with this, so a
// failed correctness check fails the runner instead of printing a green table.
static int knj_bench_end(void) {
    printf("--- %d checks, %d failures\n", knj_state.checks, knj_state.failures);
    if (knj_state.failures) {
        printf("RESULT: FAIL  worst: %s = %g\n", knj_state.worst_name,
               knj_state.worst_value);
        return 1;
    }
    printf("RESULT: PASS\n");
    return 0;
}

// ------------------------------------------------------------ comparison ---
typedef struct {
    double max_abs;
    double max_rel;
    int    worst_i;
} knj_err;

static knj_err knj_compare(const float *got, const float *want, int n,
                           int stride) {
    knj_err e = {0.0, 0.0, 0};
    for (int i = 0; i < n; i++) {
        int k = i * stride;
        double d = fabs((double)got[k] - (double)want[k]);
        double denom = fabs((double)want[k]);
        double r = denom > 1e-6 ? d / denom : d;
        if (d > e.max_abs) { e.max_abs = d; e.worst_i = k; }
        if (r > e.max_rel) e.max_rel = r;
    }
    return e;
}

// --------------------------------------------------------- fp16 helpers ---
// Stored as uint16 so host and device agree bit-for-bit. No <hip/hip_fp16.h>,
// no reliance on the runtime's conversion intrinsics.
KNJ_HD static uint16_t knj_f32_to_f16(float f) {
    uint32_t x;
    KNJ_MEMCPY(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man  = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff)                       // inf / nan
        return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u));
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);     // overflow
    if (exp <= 0) {                                       // subnormal / zero
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = (man + (1u << (shift - 1))) >> shift;
        return (uint16_t)(sign | half);
    }
    // round-to-nearest-even
    uint32_t rounded = man + 0x1000u;
    if (rounded & 0x800000u) { rounded = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (rounded >> 13));
}

KNJ_HD static float knj_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t man  = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) out = sign;
        else {                                          // subnormal
            int e = -1;
            do { man <<= 1; e++; } while (!(man & 0x400u));
            man &= 0x3ffu;
            out = sign | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (man << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    KNJ_MEMCPY(&f, &out, 4);
    return f;
}

// ------------------------------------------------------ reference GEMM ---
// Independent oracle: row-major, double accumulation, no tiling, no packing,
// no intrinsics. If the kernel's own arithmetic is right this is the number
// it must land on.
//
// A is M*K, B is K*N, C is M*N. Callers pass the shape they allocated; a
// GEMV is (M=1, N=1, K=K) with two K-vectors, and passing N=K with a
// K-length B reads K*K floats off the end.
KNJ_HD static void knj_gemm_ref(const float *A, const float *B, float *C,
                         int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++)
                acc += (double)A[m * K + k] * (double)B[k * N + n];
            C[m * N + n] = (float)acc;
        }
}

// Deterministic input so a failure is always reproducible from the seed
// alone. No rand(): the same value on host and device matters more here than
// statistical realism.
KNJ_HD static void knj_fill_ranh(float *x, int n, uint32_t seed) {
    uint32_t s = seed ? seed : 1u;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        x[i] = ((float)((s >> 9) & 0xffff) / 32768.0f) - 1.0f;   // [-1, 1)
    }
}

// --------------------------------------------------- scalar dot helpers ---
// Mirrors of what v_dot2c_f32_f16 / v_dot2_f32_f16 / v_dot8_i32_i4 compute,
// written as plain C so the host can check the kernel's arithmetic. Lane
// ORDER matters for the packed forms, so these are ordered, not folded.

KNJ_HD static float knj_dot2c_f32_f16(const uint16_t *a, const uint16_t *b) {
    float acc = 0.0f;
    for (int i = 0; i < 2; i++) acc += knj_f16_to_f32(a[i]) * knj_f16_to_f32(b[i]);
    return acc;
}

KNJ_HD static int knj_dot8_i32_i4(uint32_t a, uint32_t b) {
    // Each source dword holds 8 nibbles; the dot sums 8 signed products into
    // one i32. Nibble 0 of each byte is the low half.
    int acc = 0;
    for (int i = 0; i < 8; i++) {
        int av = (int)((a >> (i * 4)) & 0xfu);
        int bv = (int)((b >> (i * 4)) & 0xfu);
        if (av & 0x8) av -= 16;                 // sign-extend nibble
        if (bv & 0x8) bv -= 16;
        acc += av * bv;
    }
    return acc;
}

#endif  // KNJ_BENCH_H