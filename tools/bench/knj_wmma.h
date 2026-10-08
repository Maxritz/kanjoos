// Kanjoos gfx12 WMMA tile contract — geometry, the one-lane step, and the
// mapping under test, each stated once.
//
// WHY THIS FILE EXISTS
// --------------------
// Before this header the same three things were restated in every probe:
// how wide a lane's operands and accumulator are (three copies of A_LANES /
// D_SLOTS), which builtin to call, and what the published lane->(row,col)
// mapping predicts. The first two are HARDWARE facts — a probe that disagrees
// with another probe about the tile size is a measurement bug that reads like
// a hardware surprise — and the third is the HYPOTHESIS, which docs/ also
// states. Geometry and prediction live together on purpose: a prediction that
// disagrees with the step about the tile is how a probe quietly stops testing
// what it claims to test.
//
// NO RUNTIME DEPENDENCY. Nothing here needs HIP: the geometry is arithmetic,
// the step is a device intrinsic, and the maps are host math. That is what
// lets a driver's host side score a mapping, and it is why this header can be
// included by a bench the tier A/B build compiles without a device.
//
// WHAT IS MEASURED, NOT ASSUMED (emitted ISA, docs/00-verified-facts.md 7.3):
//   v_wmma_f32_16x16x16_f16   A = 4 VGPRs = 8 x f16 PER LANE
//                             B = 4 VGPRs = 8 x f16 PER LANE
//                             D = 8 VGPRs = 8 x f32 PER LANE
//   32 lanes x 8 = 256 outputs = exactly one 16x16 tile, one 16-deep K step.

#ifndef KNJ_WMMA_H
#define KNJ_WMMA_H

#include "knj_bench.h"

// --------------------------------------------------------------- geometry ---
#define KNJ_WMMA_TILE       16   // M and N of one step
#define KNJ_WMMA_WAVE       32   // lanes in a wave (RDNA4 is wave32)
#define KNJ_WMMA_K_STEP     16   // K consumed by one step
#define KNJ_WMMA_A_PER_LANE 8    // f16 halves of A a lane supplies
#define KNJ_WMMA_B_PER_LANE 8    // f16 halves of B a lane supplies
#define KNJ_WMMA_D_PER_LANE 8    // f32 accumulators a lane receives

// ------------------------------------------------------------- the step ---
// The single owner of the builtin spelling, as a MACRO.
//
// A function is not available here. This compiler build marks a helper
// host-only no matter how its attribute is spelled -- `static inline` is
// __host__ by default, and a __device__-only spelling is rejected as well --
// so calling either from inside a __global__ body fails with "no matching
// function" / "call to __host__ function from __global__ function". That is
// the same wall expert_gemm.hip names at its operand-decode macros, and why
// every existing probe inlines the builtin in the kernel body. A macro is the
// only spelling that has one name for the builtin and still compiles on both
// passes; it expands only where a kernel writes it, so nothing host-only ever
// references the builtin accidentally.
#define KNJ_WMMA_STEP(a, b, acc) \
    __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12((a), (b), (acc))

// ------------------------------------------------------- the mapping model ---
// Two candidate answers to "which (row,col) does lane L, output slot g carry?"
// Both are pure index arithmetic, so either can be scored without a device.

/* H_PUBLISHED (ROCm optimization guide, RDNA4 dense WMMA): lane L, slot g ->
 * row (L/16)*8+g, col L%16. Its inverse: (row,col) -> lane (row/8)*16+col. */
static inline int knj_wmma_pub_row(int lane, int g) { return (lane / 16) * 8 + g; }
static inline int knj_wmma_pub_col(int lane)        { return lane % 16; }
static inline int knj_wmma_pub_lane(int row, int col) { return (row / 8) * 16 + col; }

/* H_FLAT: lane L carries A row L%16 and B column L%16, over K slots [0,8).
 * This is the "8 consecutive halves" reading the first probes assumed. */
static inline int knj_wmma_flat_row(int lane) { return lane % 16; }
static inline int knj_wmma_flat_col(int lane) { return lane % 16; }

/* THE MAC FACTOR. Every probe that fed a known constant to both operands came
 * out at exactly TWICE what the published maps predict, across one step of one
 * wave on the attached RX 9070 XT: ramp x ramp 408 vs 204, one-hot A 2(k+1) vs
 * (k+1), one-hot B symmetric, the all-ones control 16 vs 8. It is a property
 * of this device + compiler pair, not of any one probe, so it is stated once
 * and applied by knj_wmma_pub_pred_slice below. */
#define KNJ_WMMA_MAC_FACTOR 2.0

/* The un-factored per-lane dot: what ONE lane's 8 A halves and 8 B halves
 * contribute if they are the K-aligned operand pair (a[j] against b[j]).
 * This is the "published" prediction with no correction applied, so the
 * factor knj_wmma_pub_pred_slice() divides by it IS the measurement, not an
 * assumption baked into the predictor. */
static inline double knj_wmma_dot_slice(const uint16_t *a, const uint16_t *b) {
    double s = 0.0;
    for (int j = 0; j < KNJ_WMMA_A_PER_LANE; j++)
        s += (double)knj_f16_to_f32(a[j]) * (double)knj_f16_to_f32(b[j]);
    return s;
}

/* What the published mapping predicts a lane sees, with the measured factor
 * applied. One function because every probe's prediction is this same sum on
 * different inputs:
 *   ramp x ramp      -> 2*(1^2+..+8^2)      = 408
 *   one-hot a_k, ramp-> 2*(k+1)
 *   ones x ones      -> 2*8                 =  16
 *   lane-scaled ramp -> 72 (the P3 discriminant) */
static inline double knj_wmma_pub_pred_slice(const uint16_t *a,
                                             const uint16_t *b) {
    return KNJ_WMMA_MAC_FACTOR * knj_wmma_dot_slice(a, b);
}

/* Ground truth for a full tile: D = A*B in double, row-major 16x16. This is
 * the oracle the known-multiply settlement compares against; it is
 * independent of both candidate maps and of any kernel.
 *
 * A and B are f16 BIT PATTERNS (uint16), not floats, because that is what a
 * driver actually hands the kernel: the lane slices are raw halves. Decoding
 * here keeps the oracle and the operand in the same domain -- an oracle that
 * took floats would silently reinterpret subnormal patterns as integers and
 * agree with a kernel fed a different matrix. */
static inline void knj_wmma_ref_tile(const uint16_t A[KNJ_WMMA_TILE][KNJ_WMMA_TILE],
                                     const uint16_t B[KNJ_WMMA_TILE][KNJ_WMMA_TILE],
                                     float D[KNJ_WMMA_TILE][KNJ_WMMA_TILE]) {
    for (int r = 0; r < KNJ_WMMA_TILE; r++)
        for (int c = 0; c < KNJ_WMMA_TILE; c++) {
            double s = 0.0;
            for (int k = 0; k < KNJ_WMMA_K_STEP; k++)
                s += (double)knj_f16_to_f32(A[r][k]) * (double)knj_f16_to_f32(B[k][c]);
            D[r][c] = (float)s;
        }
}

#endif  // KNJ_WMMA_H
