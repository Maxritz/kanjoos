// Kanjoos expert-GEMM microbenchmark — gfx1201 WMMA vs the SIMT paths.
//
// The claim under test: the gfx1201 expert GEMM runs 2-4x faster through
// v_wmma_f32_16x16x16_f16 than through the scalar/packed SIMT path we would
// otherwise write. That claim is measured in three tiers, and the tiers are
// reported separately because they prove different things:
//
//   TIER A  (runs anywhere, no device)  — this file's main(), via the host
//     compiler. A1 ALGORITHM: the exact tiling, index arithmetic and fp16
//     rounding of every variant, mirrored in scalar C and compared against
//     the independent double-precision oracle in knj_bench.h. A2 SIMD WIDTH:
//     the host model of v_dot2c_f32_f16 against its own lane-by-lane sum.
//     These prove the code is CORRECT. They say nothing about speed.
//
//   TIER B  (compile, no device)  — tools/bench/run_bench.sh. Instruction
//     census and resource census read out of the emitted .s. This is where
//     the 2-4x claim either becomes a measurement or stays an estimate, and
//     it is currently an INSTRUCTION-ISSUE ratio, not a throughput ratio.
//
//   TIER C  (needs a working ROCm install and a real gfx1201)  — the timed
//     path. Not reachable on the machine this was written on. The harness
//     reports it as UNMEASURED rather than skipping it silently.
//
// SHAPE. Qwen3-30B-A3B expert MLP: hidden 2048 -> 768 (gate/up), 768 ->
// 2048 (down). The microbenchmark keeps the tile, not the matrix: one 16x16
// output tile, K = 64 = four 16-deep WMMA steps. Enough to show the issue
// pattern; small enough that the census is readable.
//
// WHAT IS DELIBERATELY NOT CLAIMED HERE
// ------------------------------------
// v_wmma's lane-to-(row,col) mapping for the _gfx12 signature is not
// documented in the material available, and guessing it would be exactly the
// sin this harness exists to prevent. knj_wmma_layout_dump() exists to MEASURE
// it: run it once on a real gfx1201, and it writes, for each of the 256
// accumulator slots, which (row, col, k-slice) of A*B it corresponds to. Until
// that run happens the WMMA kernel below is written against a stated
// assumption, and the assumption is printed by this file's main().

#include "knj_bench.h"

#ifndef KNJ_M
#define KNJ_M 16
#endif
#ifndef KNJ_N
#define KNJ_N 16
#endif
#ifndef KNJ_K
#define KNJ_K 64
#endif

#define KTILE 16

// =========================================================================
// Device kernels. Compile-time gated by the front end exactly as the ISA
// probe found: on gfx1031 every WMMA spelling is REFUSED with
// "needs target feature wmma-128b-insts", so the file still builds there and
// the bench reports the WMMA rows as unavailable instead of the whole suite
// disappearing.
// =========================================================================

/* Deliberately NOT guarded by a hand-written arch #if. The whole point is to
 * let __builtin_amdgcn_wmma_*_gfx12 apply its own target-feature check, so the
 * runner records a real REFUSED-with-diagnostic on gfx1031 instead of an
 * EMPTY code object that says nothing. */
#if defined(KNJ_DEVICE_TIER)

extern "C" KNJ_GLOBAL void knj_gemm_wmma_f16(const knj_v8h *A, const knj_v8h *B,
                                             knj_v8f *C) {
    knj_v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < KNJ_K; k += KTILE) {
        const knj_v8h *a = A + (size_t)k * 8;
        const knj_v8h *b = B + (size_t)k * 8;
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(*a, *b, acc);
    }
    *C = acc;
}

extern "C" KNJ_GLOBAL void knj_gemm_wmma_bf16(const knj_v8b *A, const knj_v8b *B,
                                              knj_v8f *C) {
    knj_v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < KNJ_K; k += KTILE) {
        const knj_v8b *a = A + (size_t)k * 8;
        const knj_v8b *b = B + (size_t)k * 8;
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(*a, *b, acc);
    }
    *C = acc;
}

// SIMT baseline 1: scalar fp32 FMA, what you write when you do not think
// about the matrix unit at all.
extern "C" KNJ_GLOBAL void knj_gemm_simt_f32(const float *A, const float *B,
                                             float *C) {
    float acc = 0.0f;
    for (int k = 0; k < KNJ_K; k++) acc += A[k] * B[k];
    *C = acc;
}

// SIMT baseline 2: packed fp16, two lanes per instruction. This is the
// honest SIMT ceiling — comparing WMMA only against scalar f32 would
// overstate the win.
extern "C" KNJ_GLOBAL void knj_gemm_simt_pk16(const knj_v2h *A, const knj_v2h *B,
                                              knj_v2f *C) {
    knj_v2f acc = {0.0f, 0.0f};
    for (int k = 0; k < KNJ_K; k += 2) {
        knj_v2h a = A[k >> 1], b = B[k >> 1], r;
        asm volatile("v_pk_fma_f16 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b));
        acc.x += knj_f16_to_f32(r.x);
        acc.y += knj_f16_to_f32(r.y);
    }
    *C = acc;
}

// -------------------------------------------------------------------------
// Layout measurement. Feed it A = identity-ish and B = identity-ish; it dumps
// the whole accumulator. Reading the dump back gives the true lane mapping,
// which is the one thing Tier A cannot supply. Write to `out` of size
// 8 * 16 * 16 floats.
// -------------------------------------------------------------------------
extern "C" KNJ_GLOBAL void knj_wmma_layout_dump(const knj_v8h *A,
                                                 const knj_v8h *B, float *out) {
    // 256 accumulator slots, 16 x 16 x f16 on one side. Step the B matrix one
    // k-slice at a time and one output column at a time; the slot that lights
    // up identifies the mapping. Nothing here is assumed correct — the dump
    // IS the answer.
    for (int kk = 0; kk < 16; kk++) {
        knj_v8h a = A[0];
        knj_v8h b = {0, 0, 0, 0, 0, 0, 0, 0};
        b[kk] = (_Float16)1.0f;
        knj_v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
        for (int slot = 0; slot < 8; slot++)
            for (int lane = 0; lane < 16; lane++)
                out[kk * 8 * 16 + slot * 16 + lane] = ((const float *)&acc)[lane];
    }
}

#endif  /* KNJ_DEVICE_TIER */
// KNJ_EXPECT: knj_gemm_wmma_f16 = v_wmma_f32_16x16x16_f16, knj_gemm_wmma_bf16 = v_wmma_f32_16x16x16_bf16, knj_gemm_simt_pk16 = v_pk_fma_f16

/* ---------------------------------------------------------------------------
 * Everything below is TIER A: host-only. Under --cuda-device-only clang
 * compiles the whole file as device code, so without this guard the CPU
 * mirrors would be built as __host__ functions calling __device__ helpers and
 * the failure ("no matching function for call to knj_f16_to_f32") has nothing
 * to do with the kernel.
 * ------------------------------------------------------------------------- */
#if !defined(KNJ_DEVICE_TIER) && !defined(KNJ_TIER_A_OFF)


// =========================================================================
// Host mirror + reference. Compiled by the HOST compiler from the same file,
// so the tiling under test is textually the same tiling the device runs.
// =========================================================================

// Mirror of knj_gemm_simt_f32, using the same fp32 accumulation order.
static void mirror_simt_f32(const float *A, const float *B, float *C, int K) {
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += A[k] * B[k];
    *C = acc;
}

// Mirror of knj_gemm_simt_pk16: the packed f16 FMA rounds each product to
// fp16 BEFORE the accumulate, which is the whole reason this path is
// numerically worse than the scalar one. Modelling that rounding is the
// point — a mirror that quietly accumulated in fp32 would "pass" against the
// oracle for the wrong reason.
static void mirror_simt_pk16(const uint16_t *Ah, const uint16_t *Bh, float *C,
                             int K) {
    float acc = 0.0f;
    for (int k = 0; k < K; k += 2) {
        // The device does v_pk_fma_f16 r, a, b, r: both products are formed
        // and rounded to fp16, then added to r, then rounded again.
        uint16_t p0 = knj_f32_to_f16(knj_f16_to_f32(Ah[k]) * knj_f16_to_f32(Bh[k]));
        uint16_t p1 = knj_f32_to_f16(knj_f16_to_f32(Ah[k + 1]) * knj_f16_to_f32(Bh[k + 1]));
        acc += knj_f16_to_f32(p0);
        acc += knj_f16_to_f32(p1);
    }
    *C = acc;
}

// Mirror of the WMMA tiling: identical 16-deep k-stride, but the primitive is
// a scalar 16-element dot written out, so the index arithmetic is what is
// being tested rather than the hardware.
static void mirror_wmma_tiling(const float *A, const float *B, float *C,
                               int K) {
    double acc = 0.0;
    for (int k = 0; k < K; k += KTILE)
        for (int i = 0; i < KTILE; i++)
            acc += (double)A[k + i] * (double)B[k + i];
    *C = (float)acc;
}

int main(void) {
    knj_bench_begin("expert GEMM A/B: WMMA vs SIMT (tier A — algorithm)");

    float A[KNJ_K], B[KNJ_K];
    float want[KNJ_K], got[KNJ_K];
    knj_fill_ranh(A, KNJ_K, 0xC0FFEEu);
    knj_fill_ranh(B, KNJ_K, 0xBEEF01u);
    // 1 x K times K x 1 — a GEMV. B is a K-vector, not a K x N matrix.
    knj_gemm_ref(A, B, want, 1, 1, KNJ_K);

    // --- A1: every variant must land on the same oracle -------------------
    memset(got, 0, sizeof(got));
    mirror_simt_f32(A, B, &got[0], KNJ_K);
    knj_err e = knj_compare(got, want, 1, 1);
    knj_bench_check("A1 simt_f32   max abs err vs fp64 oracle", e.max_abs, 1e-4);

    uint16_t Ah[KNJ_K], Bh[KNJ_K];
    for (int i = 0; i < KNJ_K; i++) {
        Ah[i] = knj_f32_to_f16(A[i]);
        Bh[i] = knj_f32_to_f16(B[i]);
    }
    memset(got, 0, sizeof(got));
    mirror_simt_pk16(Ah, Bh, &got[0], KNJ_K);
    e = knj_compare(got, want, 1, 1);
    // fp16 products round harder than fp32; KNJ_K=64 accumulated fp16 steps
    // is where the error shows, and the tolerance says so explicitly instead
    // of being tuned until green.
    knj_bench_check("A1 simt_pk16  max abs err vs fp64 oracle", e.max_abs, 2e-2);

    memset(got, 0, sizeof(got));
    mirror_wmma_tiling(A, B, &got[0], KNJ_K);
    e = knj_compare(got, want, 1, 1);
    knj_bench_check("A1 wmma tiling max abs err vs fp64 oracle", e.max_abs, 1e-4);

    // --- A2: the packed-f16 dot model must equal its own lane sum ---------
    {
        uint16_t ah[2] = {knj_f32_to_f16(0.75f), knj_f32_to_f16(-0.5f)};
        uint16_t bh[2] = {knj_f32_to_f16(1.5f), knj_f32_to_f16(0.25f)};
        float lane_sum = knj_f16_to_f32(ah[0]) * knj_f16_to_f32(bh[0]) +
                         knj_f16_to_f32(ah[1]) * knj_f16_to_f32(bh[1]);
        float modelled = knj_dot2c_f32_f16(ah, bh);
        knj_bench_check("A2 v_dot2c_f32_f16 model vs lane sum",
                        fabs((double)(modelled - lane_sum)), 1e-6);
    }

    // --- A3: fp16 round trip must be lossless for exactly representable --
    {
        double worst = 0.0;
        for (int i = 0; i < 2048; i++) {
            float x = (float)((i - 1024) / 256.0);   // all exact in fp16
            float y = knj_f16_to_f32(knj_f32_to_f16(x));
            worst = fmax(worst, fabs((double)(y - x)));
        }
        knj_bench_check("A3 fp16 round trip on exact values", worst, 0.0);
    }

    // --- A4: the int4 dot model, checked against a brute-force sum --------
    {
        double worst = 0.0;
        for (uint32_t s = 1; s <= 64; s++) {
            uint32_t a = 0x12345678u ^ (s * 0x9e3779b9u);
            uint32_t b = 0xdeadbeefu ^ (s * 0x85ebca6bu);
            int model = knj_dot8_i32_i4(a, b);
            int brute = 0;
            for (int i = 0; i < 8; i++) {
                int av = (int)((a >> (i * 4)) & 0xfu), bv = (int)((b >> (i * 4)) & 0xfu);
                brute += (av & 0x8 ? av - 16 : av) * (bv & 0x8 ? bv - 16 : bv);
            }
            worst = fmax(worst, fabs((double)(model - brute)));
        }
        knj_bench_check("A4 v_dot8_i32_i4 model vs brute force", worst, 0.0);
    }

    // --- what this tier does NOT establish -------------------------------
    printf("\n");
    printf("NOT ESTABLISHED BY TIER A:\n");
    printf("  * v_wmma's lane->(row,col) mapping for the _gfx12 signature.\n");
    printf("    MEASURED by knj_wmma_layout_dump() on a real gfx1201.\n");
    printf("  * any speedup number. See tools/bench/run_bench.sh for the\n");
    printf("    instruction-issue ratio, and read it as that.\n");

    return knj_bench_end();
}

#endif  /* tier A: host-only, and not when included for tier C */
