// Kanjoos WMMA device-probe rig — the one-wave A/B/D state and the scenario
// fills, each stated once, for every driver that measures a single gfx12 WMMA
// step. The arch guard is knj_wmma_guard.h.
//
// WHY THIS FILE EXISTS
// --------------------
// Two probes (wmma_layout.hip, wmma_tile_probe.hip) each carried a private
// copy of the same rig: the same `NWAVE 32` / `D_SLOTS 8`, the same four
// globals, the same upload/launch/readback, and the same `lanes_differ()`.
// Only the SCENARIOS differed. Worse, the score table in wmma_tile_probe.hip
// scored numbers it had not measured -- it quoted wmma_layout.hip's P1..P4
// output as literals, so a change in one probe could leave the other scoring a
// world that no longer existed.
//
// One wave, one 16x16x16 WMMA step, A/B/D in global memory -- the rig both
// layout probes measure through. A driver owns its scenario ORDER and its
// printing, nothing else. Because both drivers call knj_wmma_scenario() to
// produce the operands, the tile probe's score is computed from the same
// measurements the layout probe prints -- there is no second copy to drift.
//
// Requires HIP: this is the DEVICE-side rig, unlike knj_bench.h and
// knj_wmma.h, which are deliberately HIP-free so tier A can compile them.

#ifndef KNJ_WMMA_PROBE_H
#define KNJ_WMMA_PROBE_H

#include "knj_wmma_guard.h"   // knj_wmma_dev + knj_wmma_guard()
#include "knj_wmma.h"         // geometry + the builtin, as KNJ_WMMA_STEP

// ---- scenario ----
// The fills the probes are made of. Naming them once is what lets the tile
// probe score the SAME operands the layout probe prints.
//
//   P1  ramp x ramp        A[j] = B[j] = j+1        -> the 8-point dot
//   P2  one-hot A, ramp B  A[j] = (j==k), B[j]=j+1  -> which slots A lane k feeds
//   P3  lane-scaled ramp   A[j] = (lane+1)(j+1), B=1 -> is the LANE an input?
//   P4  one-hot B, ramp A  A[j] = j+1, B[j] = (j==k) -> which slots B lane k feeds
//   CTRL all ones          A = B = 1                 -> is the kernel alive at all?
enum {
    KNJ_WMMA_P1_RAMP    = 0,
    KNJ_WMMA_P2_AONEHOT = 1,
    KNJ_WMMA_P3_LANERAMP = 2,
    KNJ_WMMA_P4_BONEHOT = 3,
    KNJ_WMMA_CTRL_ONES  = 4
};

// ---------------------------------------------------------------- rig ---

typedef struct {
    uint16_t a[KNJ_WMMA_WAVE * KNJ_WMMA_A_PER_LANE];
    uint16_t b[KNJ_WMMA_WAVE * KNJ_WMMA_B_PER_LANE];
    float    d[KNJ_WMMA_WAVE * KNJ_WMMA_D_PER_LANE];
    uint16_t *da, *db;
    float    *dd;
} knj_wmma_rig;

__global__ void knj_wmma_probe_kernel(const uint16_t *A, const uint16_t *B,
                                      float *D) {
    int lane = threadIdx.x;
    knj_v8h a, b;
    /* Each lane reads ITS OWN slice, so lane independence is testable:
     * layout A[lane][j], B[lane][j]. D is written per lane as well
     * (lane L writes D[L*8 .. L*8+7]) so no two lanes race. */
    KNJ_MEMCPY(&a, A + lane * KNJ_WMMA_A_PER_LANE, sizeof(a));
    KNJ_MEMCPY(&b, B + lane * KNJ_WMMA_B_PER_LANE, sizeof(b));
    knj_v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
    acc = KNJ_WMMA_STEP(a, b, acc);
    KNJ_MEMCPY(D + lane * KNJ_WMMA_D_PER_LANE, &acc, sizeof(acc));
}

static int knj_wmma_rig_open(knj_wmma_rig *r) {
    if (hipMalloc(&r->da, sizeof(r->a)) != hipSuccess ||
        hipMalloc(&r->db, sizeof(r->b)) != hipSuccess ||
        hipMalloc(&r->dd, sizeof(r->d)) != hipSuccess) {
        printf("malloc failed\n");
        return 1;
    }
    return 0;
}

static void knj_wmma_rig_close(knj_wmma_rig *r) {
    hipFree(r->da); hipFree(r->db); hipFree(r->dd);
}

/* Fill r->a / r->b for one scenario. `k` is the one-hot index for P2/P4 and
 * is ignored otherwise. */
static void knj_wmma_scenario(knj_wmma_rig *r, int scenario, int k) {
    for (int l = 0; l < KNJ_WMMA_WAVE; l++)
        for (int j = 0; j < KNJ_WMMA_A_PER_LANE; j++) {
            int i = l * KNJ_WMMA_A_PER_LANE + j;
            float av = 0.0f, bv = 0.0f;
            switch (scenario) {
            case KNJ_WMMA_P1_RAMP:     av = (float)(j + 1);          bv = (float)(j + 1); break;
            case KNJ_WMMA_P2_AONEHOT:  av = (j == k) ? 1.0f : 0.0f;  bv = (float)(j + 1); break;
            case KNJ_WMMA_P3_LANERAMP: av = (float)((l + 1) * (j + 1)); bv = 1.0f;        break;
            case KNJ_WMMA_P4_BONEHOT:  av = (float)(j + 1);          bv = (j == k) ? 1.0f : 0.0f; break;
            default:                   av = 1.0f;                    bv = 1.0f;           break;
            }
            r->a[i] = knj_f32_to_f16(av);
            r->b[i] = knj_f32_to_f16(bv);
        }
}

/* Upload, launch one wave, read back. Exits 1 on a launch error, as the
 * probes did: a launch that did not happen must not look like a measurement. */
static void knj_wmma_rig_run(knj_wmma_rig *r) {
    hipMemcpy(r->da, r->a, sizeof(r->a), hipMemcpyHostToDevice);
    hipMemcpy(r->db, r->b, sizeof(r->b), hipMemcpyHostToDevice);
    knj_wmma_probe_kernel<<<1, KNJ_WMMA_WAVE>>>(r->da, r->db, r->dd);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        printf("launch failed: %s\n", hipGetErrorString(e));
        exit(1);
    }
    hipMemcpy(r->d, r->dd, sizeof(r->d), hipMemcpyDeviceToHost);
}

/* Fill, run, return in one call -- the shape every probe actually wants. */
static void knj_wmma_measure(knj_wmma_rig *r, int scenario, int k) {
    knj_wmma_scenario(r, scenario, k);
    knj_wmma_rig_run(r);
}

static float knj_wmma_at(const knj_wmma_rig *r, int lane, int g) {
    return r->d[lane * KNJ_WMMA_D_PER_LANE + g];
}

/* Does D depend on the lane? Compare each lane's 8 slots against lane 0's.
 * Note this is a comparison against LANE 0, so it answers "is lane 0
 * representative", not "are all lanes equal" -- the difference is exactly the
 * P3 finding (lanes 0-3 identical, lanes 4-31 not). */
static int knj_wmma_lanes_differ(const knj_wmma_rig *r) {
    for (int l = 1; l < KNJ_WMMA_WAVE; l++)
        for (int g = 0; g < KNJ_WMMA_D_PER_LANE; g++)
            if (knj_wmma_at(r, l, g) != knj_wmma_at(r, 0, g)) return 1;
    return 0;
}

/* Are all 8 accumulator slots of `lane` equal? The published output map says
 * lane L carries one column of 8 rows, so under it they must be. */
static int knj_wmma_slots_equal(const knj_wmma_rig *r, int lane) {
    for (int g = 1; g < KNJ_WMMA_D_PER_LANE; g++)
        if (knj_wmma_at(r, lane, g) != knj_wmma_at(r, lane, 0)) return 0;
    return 1;
}

#endif  // KNJ_WMMA_PROBE_H
