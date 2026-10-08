// kernels/knj_kernels.hip -- device kernels for the expert path and the
// capability probes that gate them.
//
// ---------------------------------------------------------------------------
// TOOLCHAIN CONSTRAINTS THIS FILE RESPECTS (all measured, all learned the hard
// way; see docs/00-verified-facts.md sections 7.0, 8.5 and 8.7)
// ---------------------------------------------------------------------------
//
// 1. NO gridDim IN ANY KERNEL. Under the link line this project must use
//    (-nogpulib, because hipcc cannot resolve the device bitcode directory on
//    this machine), gfx1201 materialises gridDim through __ockl_get_num_groups,
//    whose inline definition lives in the device runtime we are not linking. The
//    link then fails with `undefined hidden symbol`.
//
//    Therefore EVERY launch takes its block count and grid stride as explicit
//    kernel arguments, and every loop is written as an explicit grid-stride
//    loop. There is no `gridDim` and no `blockDim` anywhere below. If a future
//    ROCm makes the runtime resolvable this can change, but it must not change
//    silently: the launch helper in src/device/ is where the decision lives.
//
// 2. NO DEVICE-SIDE LIBM. No sqrt/exp/pow in any kernel. This file uses only
//    integer ops, fp32 multiply-add, and hand-rolled fp16<->fp32 bit
//    conversions. __expf would be acceptable (it is the native v_exp_f32
//    instruction, not a libm call) but nothing here needs it.
//
// 3. NO HIP HEADERS REQUIRED. This file compiles both with <hip/hip_runtime.h>
//    present (real build through hipcc) and with no HIP include path at all
//    (compile-only capability checks via -nogpuinc), which is how
//    tools/isa_probe stays immune to a broken install.
//
// 4. ACCUMULATORS ARE ALWAYS INITIALISED. Every `v_dot*` is the ACCUMULATING
//    form, so an uninitialised accumulator register is silent-wrong-answer, not
//    a compile error. It shipped once in this repo at 10x the oracle error
//    (docs/00-verified-facts.md section 7.5 item 6).
//
// ---------------------------------------------------------------------------
// WHAT IS DELIBERATELY NOT HERE YET
// ---------------------------------------------------------------------------
// The tiling here is the SIMPLE version: one thread per (row, column) output,
// two dots per eight weights. It is correct and it is verifiable, which is the
// right thing to land first. The measured tuning (2 columns per thread, hoisted
// row bound, split-K 1->64) took the benchmark kernel from 0.8% to 5.2% of
// peak -- see docs/00-verified-facts.md section 8.9 -- and that tuning is C3's
// job at runtime, not something to hard-code before there is a correctness
// baseline to tune against. Any variant added later must be A/B'd against this
// one on identical inputs.

// Include HIP's runtime when it is available, and stand in for it when it is not.
//
// The two builds this file must satisfy:
//   * full build through hipcc  -- <hip/hip_runtime.h> is present, and hipcc
//     additionally wants to EMIT LAUNCHER STUBS for every __global__ it sees.
//     Without the header it cannot find `hipLaunchKernel` and fails with
//     "Can't find declaration for hipLaunchKernel".
//   * compile-only check through clang -nogpuinc -- no HIP headers at all, so
//     the qualifiers are supplied here. This is the mode that keeps the
//     capability checks running on a machine with a broken ROCm install.
#if defined(__has_include)
#  if __has_include(<hip/hip_runtime.h>)
#    include <hip/hip_runtime.h>
#    define KNJ_HAVE_HIP_HEADERS 1
#  endif
#endif

#ifndef KNJ_HAVE_HIP_HEADERS
#  define KNJ_HAVE_HIP_HEADERS 0
#  ifndef __global__
#    define __global__ __attribute__((global))
#  endif
#endif

#include "knj_pack.h"
#include "knj_dot.h"
#include "knj_index.h"

// ---------------------------------------------------------------------------
// fp16 <-> fp32, by bit manipulation (no libm, no __half, no HIP headers)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float knj_bits_to_f32(uint32_t bits) {
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

__device__ __forceinline__ uint32_t knj_f32_to_bits(float f) {
    uint32_t b;
    __builtin_memcpy(&b, &f, sizeof(b));
    return b;
}

// fp16 -> fp32. Correct including subnormals; the scale of a W4 group is never
// subnormal in practice but a silently wrong subnormal path is the kind of bug
// that only appears on a model with a badly conditioned group.
__device__ __forceinline__ float knj_h2f(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (uint32_t)(h >> 10) & 0x1fu;
    const uint32_t man  = (uint32_t)(h & 0x3ffu);
    uint32_t bits;
    if (exp == 0u) {
        if (man == 0u) {
            bits = sign;
        } else {
            uint32_t e = 0u, m = man;
            while ((m & 0x400u) == 0u) { m <<= 1; ++e; }
            m &= 0x3ffu;
            bits = sign | ((113u - e) << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    return knj_bits_to_f32(bits);
}

// ---------------------------------------------------------------------------
// knj_control -- the control kernel
// ---------------------------------------------------------------------------
// Purpose, and it is not optional: a kernel that computes the wrong thing and a
// kernel that was never dispatched look identical from the host if you only
// check the output. On this machine a gfx1031 code object executed on the
// gfx1201 card without error, dispatched nothing, and printed a plausible
// timing table at 2327% of peak (docs/00-verified-facts.md section 8.7).
//
// So the FIRST thing every driver does is launch this. It writes
// `out[0] = seed + blockIdx_linear*0 + tid_from_args`, i.e. a value the host
// can predict from the launch parameters alone. If that does not appear, nothing
// else in the run means anything.
//
// Note it takes its indices as ARGUMENTS for the same reason as everything else:
// there is no blockIdx/threadIdx available under -nogpulib either.
extern "C" __global__
void knj_control(uint32_t *out, uint32_t seed, uint32_t n_threads,
                 uint32_t n_blocks) {
    // knj_tid/knj_bid are compiler builtins wrapped in knj_index.h, so this
    // compiles with or without HIP headers. gridDim is NOT usable here -- see
    // constraint 1 at the top -- which is why the launch configuration arrives
    // as an argument.
    const uint32_t gid = knj_gid(n_threads, n_blocks);
    if (gid == 0u) {
        out[0] = seed + 1u;              /* host expects exactly this */
        out[1] = n_blocks;               /* echo: proves the launch config */
    }
}

// ===========================================================================
// The W4 expert GEMM
// ===========================================================================
//
// Computes, for one expert, one projection:
//
//     C[m][n] = sum_k  dequant(Bq[n][k]) * Act[m][k]      for all m<M, n<N
//
// Layouts (all documented in kernels/knj_pack.h, and all mirrored by the host
// oracle in tests/host_oracle.h -- if these three descriptions disagree, one of
// them is wrong and the oracle catches it):
//
//   Bq    packed weights, COLUMN-major over n, group-major over k.
//         Per (n, group): 64 payload bytes (128 nibbles) + fp16 scale +
//         uint8 zero-point + 1 pad byte = 68 bytes.
//   Act   de-interleaved int8 activations, [M][K/8][2] uint32 words.
//         word 0 of each pair = even k offsets, word 1 = odd k offsets.
//   Asum  int32 [M][groups] -- sum of activations in each group, precomputed.
//         This is what removes the +/-8 centring term from the inner loop.
//   C     fp32 [M][N].
//
// Why Asum is precomputed rather than accumulated per column:
//     weight_k = (u_k - 8) * scale
//     sum_k weight_k * a_k = scale * (sum_k u_k a_k - 8 * sum_k a_k)
// The second term depends only on (m, group), not on n. Computing it inside the
// kernel would add two extra dots per eight weights and roughly DOUBLE the
// instruction count of the inner loop, which would make the pack look more
// expensive than it is. Hoisting it costs one O(M*K) pass. The inner loop is
// then exactly: 1 weight load + 2 unpack + 2 dots per 8 MACs, which is the shape
// docs/00-verified-facts.md section 7.6 settled.
extern "C" __global__
void knj_gemm_w4_i8(const uint8_t *Bq,
                    const uint32_t *Act,
                    const int32_t *Asum,
                    float *C,
                    uint32_t M, uint32_t N, uint32_t K,
                    uint32_t n_threads, uint32_t n_blocks) {
    const uint32_t stride = n_threads * n_blocks;
    const uint32_t groups = K / KNJ_GROUP_EXPERT;
    const uint32_t iters_per_group = KNJ_GROUP_EXPERT / 8;   /* 16 */

    for (uint32_t idx = knj_gid(n_threads, n_blocks);
         idx < M * N; idx += stride) {
        const uint32_t m = idx / N;
        const uint32_t n = idx - m * N;

        const uint8_t *col = Bq + (size_t)n * groups * KNJ_GROUP_STRIDE_BYTES(4, KNJ_GROUP_EXPERT);
        const uint32_t *arow = Act + (size_t)m * (K / 8) * 2;
        const int32_t *asum_row = Asum + (size_t)m * groups;

        float acc = 0.0f;

        for (uint32_t g = 0; g < groups; ++g) {
            const uint8_t *grp = col + (size_t)g * KNJ_GROUP_STRIDE_BYTES(4, KNJ_GROUP_EXPERT);
            const float scale = knj_h2f((uint16_t)(grp[64] | (uint16_t)(grp[65] << 8)));
            const int32_t asum_g = asum_row[g];

            int32_t raw = 0;
            const uint32_t kbase = g * KNJ_GROUP_EXPERT;
            for (uint32_t j = 0; j < iters_per_group; ++j) {
                uint32_t packed;
                __builtin_memcpy(&packed, grp + (size_t)j * 4, 4);
                int lo, hi;
                knj_unpack8_u4(packed, &lo, &hi);
                const uint32_t a2 = kbase / 8 + j;
                raw = knj_dot4_i32_i8(lo, (int)arow[a2 * 2 + 0], raw);
                raw = knj_dot4_i32_i8(hi, (int)arow[a2 * 2 + 1], raw);
            }
            acc += scale * ((float)raw - (float)(KNJ_NIBBLE_BIAS * asum_g));
        }
        C[idx] = acc;
    }
}

// Asum is indexed by (m, g) but the kernel above reads one value per group from
// a per-m row; the two must agree, so the helper that fills it is here so the
// layout is written down once.
__device__ __forceinline__
int32_t knj_asum_at(const int32_t *Asum, uint32_t m, uint32_t g, uint32_t groups) {
    return Asum[(size_t)m * groups + g];
}

// ===========================================================================
// A standalone Asum builder, so the (m, g) layout above is produced by code in
// this tree rather than by whoever calls the GEMM.
// ===========================================================================
extern "C" __global__
void knj_build_asum(const uint32_t *Act, int32_t *Asum,
                    uint32_t M, uint32_t K,
                    uint32_t n_threads, uint32_t n_blocks) {
    const uint32_t stride = n_threads * n_blocks;
    const uint32_t groups = K / KNJ_GROUP_EXPERT;

    for (uint32_t idx = knj_gid(n_threads, n_blocks);
         idx < M * groups; idx += stride) {
        const uint32_t m = idx / groups;
        const uint32_t g = idx - m * groups;
        const uint32_t *arow = Act + (size_t)m * (K / 8) * 2;
        const uint32_t kbase = g * KNJ_GROUP_EXPERT;
        int32_t s = 0;
        for (uint32_t j = 0; j < KNJ_GROUP_EXPERT / 8; ++j) {
            const uint32_t a2 = kbase / 8 + j;
            // Assign, do NOT `+=`. knj_dot4_i32_i8 returns acc + dot, so writing
            // `s += knj_dot4_i32_i8(..., s)` yields 2*s + dot and the sum doubles
            // every iteration. That is exactly the bug that was here: a true sum
            // of 7 came back as 47,298,495, which is 2^32-ish growth over 32
            // iterations. It compiled cleanly and the kernel demonstrably wrote
            // to every entry, so neither the control kernel nor a poisoned buffer
            // would have flagged it -- only comparing the values did.
            s = knj_dot4_i32_i8(0x01010101, (int)arow[a2 * 2 + 0], s);
            s = knj_dot4_i32_i8(0x01010101, (int)arow[a2 * 2 + 1], s);
        }
        Asum[idx] = s;
    }
}

// ===========================================================================
// Capability probe -- runs on the device, reports what the device actually did
// ===========================================================================
//
// docs/02-components.md C2 requires that caps come from PROBING and that "a
// kernel that needs a cap the device does not advertise is never instantiated".
// A runtime probe that just reads back what the compiler was told is not a
// probe, it is an echo. What is actually verifiable at runtime is narrower and
// more useful:
//
//   cap[0] = 1 iff the integer dot path produced a self-consistent result
//            (the kernel computes a known identity and compares it in-kernel);
//   cap[1] = 1 iff packed fp16 multiply-add behaves as fp16 (no silent fp32
//            promotion that would change the numerics of the SIMT path), checked
//            via v_pk_fma_f16 because no f16 dot links on both targets.
//
// Anything that cannot be checked this way -- WMMA operand layout, for instance
// -- is reported as UNKNOWN and must be established by the offline probe plus a
// dedicated correctness test, NOT asserted here. Guessing a capability from the
// arch string is how this project shipped a false ISA matrix once.
extern "C" __global__
void knj_probe_caps(uint32_t *cap, const int32_t *in) {
    const uint32_t gid = knj_gid(32u, 1u);
    if (gid != 0u) {
        return;
    }
    // Integer dot identity: dot4(1,1) == 4. This is the instruction the expert
    // path actually ships on, verified by execution rather than by assertion.
    int32_t d = 0;
    d = knj_dot4_i32_i8(0x01010101, 0x01010101, d);
    cap[0] = (d == 4) ? 1u : 0u;

    // Packed fp16 FMA accumulation, from an explicitly zeroed accumulator.
    // This is a v_pk_fma_f16 check, NOT an f16 dot: there is no f16 dot in this
    // toolchain that links on both targets (see kernels/knj_dot.h, "The f16
    // float dot"). Probing the instruction that actually exists is the point.
    //
    // Semantics: v_pk_fma_f16 D, S0, S1, S2 means D = S0*S1 + S2, so S2 is the
    // ACCUMULATOR. Starting from 1.0h and chaining two FMAs therefore gives 3.0h
    // (1 -> 2 -> 3), not 4.0h. An earlier version of this check expected 0x4400
    // and reported FAILED with 0x4200 observed -- the probe was right and the
    // expectation was wrong. It now starts from zero, which both pins the
    // accumulate-into-initialised-register behaviour (docs/00 section 7.5 item 6)
    // and gives a legible expected value: two passes over 1.0h from zero is 2.0h.
    {
        const int one = (int)0x3c003c00;          /* two fp16 1.0 in one VGPR */
        int r = 0;                                /* MUST be initialised */
        asm volatile("v_pk_fma_f16 %0, %1, %2, %0" : "+v"(r) : "v"(one), "v"(one));
        asm volatile("v_pk_fma_f16 %0, %1, %2, %0" : "+v"(r) : "v"(one), "v"(one));
        const uint16_t lo = (uint16_t)(r & 0xffff);
        cap[1] = (lo == 0x4000u) ? 1u : 0u;       /* 2.0h == 0x4000 */
        cap[4] = (uint32_t)lo;                    /* report the raw pattern too */
    }

    cap[2] = 1u;   /* reached this line at all: the kernel really dispatched */
    cap[3] = in ? in[0] : 0u;
}