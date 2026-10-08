// Kanjoos WMMA operand-layout probe — the blocking unknown for gfx1201.
//
// WHAT THIS FILE IS NOW
// ---------------------
// It is the SCENARIO and the PRESENTATION, nothing else. The tile geometry and
// the builtin live in knj_wmma.h; the arch guard, the one-wave A/B/D state,
// the scenario fills and the launch live in knj_wmma_probe.h. This file says
// which scenarios to run, in what order, and how to print them.
//
// WHY THAT SPLIT
// --------------
// wmma_tile_probe.hip scores these same probes. When each file carried its own
// copy of `A_LANES`/`D_SLOTS`, its own four globals and its own `run()`, the
// tile probe ended up scoring literals copied from this file's output rather
// than the measurements themselves. Now both call knj_wmma_scenario(), so the
// score cannot outlive the number it scores.
//
// WHY THE FIRST PROBE FAILED
// --------------------------
// The earlier one-hot probe set A = e_k and B = ALL ONES, then compared D
// across k. Every D slot was identical, and the wrong conclusion drawn was
// "the A lanes are not individually addressable". The probe was broken, not
// the hardware: with B uniform, ANY mapping of A onto D produces the same D.
// A one-hot probe needs a VARYING B to have a signal at all.
//
// This probe therefore varies B (B[j] = j+1) on every probe, so a
// non-uniform A-to-D mapping shows up as non-uniform D.
//
// WHAT IS BEING MEASURED
// ----------------------
//   P1  A = [1..8], B = [1..8]      -> D per lane per slot
//   P2  A = e_k,  B = [1..8]        -> which D slots A's lane k feeds
//   P3  A = [1..8] per lane scaled by (lane+1), B = [1..8]
//                                    -> does D depend on the LANE at all?
//   P4  B = e_k,  A = [1..8]        -> which D slots B's lane k feeds
//
// Build (the only line that links on this machine):
//   hipcc -nogpulib -O2 --offload-arch=gfx1201 \
//     -Xclang -target-feature -Xclang +wavefrontsize32 \
//     -I tools/bench -I kernels -L'G:\ROCM10RT-gfx1201\lib' -lamdhip64 \
//     wmma_layout.hip -o lay.exe
//   KNJ_BUILD_ARCH=gfx1201 ./lay.exe

#include <hip/hip_runtime.h>
#include <stdio.h>

#include "knj_wmma_probe.h"

int main(void) {
    knj_wmma_dev dev;
    int guard = knj_wmma_guard(&dev);
    if (guard) return guard;
    printf("device %s  arch %s  wave %d\n", dev.name, dev.arch, dev.warp);

    knj_wmma_rig rig;
    if (knj_wmma_rig_open(&rig)) return 1;

    // ---- P1: ramp x ramp -------------------------------------------------
    knj_wmma_measure(&rig, KNJ_WMMA_P1_RAMP, 0);
    printf("\nP1  A=[1..8] B=[1..8]\n");
    printf("  lane 0 D =");
    for (int i = 0; i < KNJ_WMMA_D_PER_LANE; i++) printf(" %6g", knj_wmma_at(&rig, 0, i));
    printf("\n  D identical across all 32 lanes? %s\n",
           knj_wmma_lanes_differ(&rig) ? "NO" : "yes");

    // ---- P2: one-hot A, ramp B ------------------------------------------
    printf("\nP2  A = e_k, B = [1..8]   (which D slots does A lane k feed?)\n");
    int masks[KNJ_WMMA_A_PER_LANE][KNJ_WMMA_D_PER_LANE];
    for (int k = 0; k < KNJ_WMMA_A_PER_LANE; k++) {
        knj_wmma_measure(&rig, KNJ_WMMA_P2_AONEHOT, k);
        for (int i = 0; i < KNJ_WMMA_D_PER_LANE; i++) masks[k][i] = (int)knj_wmma_at(&rig, 0, i);
        printf("  A=e%d ->", k);
        for (int i = 0; i < KNJ_WMMA_D_PER_LANE; i++) printf(" %6g", knj_wmma_at(&rig, 0, i));
        printf("%s\n", knj_wmma_lanes_differ(&rig) ? "   (D varies with lane)" : "");
    }
    int distinct = 0;
    for (int k = 1; k < KNJ_WMMA_A_PER_LANE; k++) {
        int same = 1;
        for (int i = 0; i < KNJ_WMMA_D_PER_LANE; i++)
            if (masks[k][i] != masks[0][i]) same = 0;
        if (!same) distinct++;
    }
    printf("  -> %d of %d A lanes produce a DISTINCT D pattern\n",
           distinct, KNJ_WMMA_A_PER_LANE - 1);

    // ---- P3: is the lane index even an input? ---------------------------
    knj_wmma_measure(&rig, KNJ_WMMA_P3_LANERAMP, 0);
    printf("\nP3  A[lane][j] = (lane+1)*(j+1), B = 1\n");
    for (int l = 0; l < 4; l++) {
        printf("  lane %2d D =", l);
        for (int i = 0; i < KNJ_WMMA_D_PER_LANE; i++) printf(" %8g", knj_wmma_at(&rig, l, i));
        printf("\n");
    }
    printf("  -> D depends on the lane? %s\n", knj_wmma_lanes_differ(&rig) ? "YES" : "NO");

    // ---- P4: one-hot B, ramp A ------------------------------------------
    printf("\nP4  A = [1..8], B = e_k   (which D slots does B lane k feed?)\n");
    for (int k = 0; k < KNJ_WMMA_A_PER_LANE; k++) {
        knj_wmma_measure(&rig, KNJ_WMMA_P4_BONEHOT, k);
        printf("  B=e%d ->", k);
        for (int i = 0; i < KNJ_WMMA_D_PER_LANE; i++) printf(" %6g", knj_wmma_at(&rig, 0, i));
        printf("\n");
    }

    printf("\nREAD THIS AS: a per-lane 8-point dot if every slot in a row is\n"
           "equal; a 16x16 tile assignment if they differ. Both are possible\n"
           "for v_wmma and they need completely different GEMM kernels.\n");
    knj_wmma_rig_close(&rig);
    return 0;
}
