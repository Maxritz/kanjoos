// Kanjoos WMMA arch guard — refuse to report numbers from a binary that is not
// running on the arch it was built for. One implementation for every WMMA
// driver that talks to a device.
//
// WHY THIS IS ITS OWN FILE, APART FROM knj_wmma_probe.h
// -----------------------------------------------------
// Two different things were bundled under "the probe rig": the GUARD (do not
// measure on the wrong arch) and the RIG (one wave, 32 lanes, A/B/D in global
// memory, scenario fills). Only the two layout probes need the rig.
// wmma_run.hip needs the guard and nothing else -- it times its own kernels and
// never touches the rig. Including the rig header there would compile
// knj_wmma_probe_kernel into an image that never launches it, which is dead
// device code, not a shared dependency. So the guard gets its own header and
// wmma_run includes only that.
//
// Requires HIP: this is device-runtime HOST code, unlike knj_bench.h and
// knj_wmma.h, which are deliberately HIP-free so tier A can compile them.

#ifndef KNJ_WMMA_GUARD_H
#define KNJ_WMMA_GUARD_H

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char name[256];
    char arch[64];
    int  warp;
    int  cu;
    int  threads_per_cu;
} knj_wmma_dev;

/* ARCH GUARD.
 *
 * A binary built --offload-arch=gfx1031 RUNS on a gfx1201 card without an
 * error. The kernel is never dispatched, every output keeps its hipMemset
 * value, and a timing table comes out as "2327% of peak" in 0.1 us. A clean
 * build, a plausible number and a well-formatted table are all consistent
 * with the kernel never having run at all. So a probe refuses to report
 * numbers unless the arch it was built for is the arch it is running on.
 *
 * The arch comes from the ENVIRONMENT, not from -D. hipcc re-spawns clang
 * through a command STRING, so a -D carrying quotes loses them to the inner
 * shell and the compiler sees a bare identifier: "use of undeclared
 * identifier gfx1201". It also cannot come from __gfx1201__ and friends,
 * because those are DEVICE-pass macros and this is HOST code -- a guard
 * written that way compiles to nothing and silently guards nothing.
 *
 * Returns 0 to proceed, 7 (KNJ_BUILD_ARCH unset) or 6 (arch mismatch) to
 * stop. Those are the repo's tier-C exit codes and run_bench.sh reads them.
 * On success *out carries the device facts; the banner stays the caller's,
 * because the drivers print different ones. */
static int knj_wmma_guard(knj_wmma_dev *out) {
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, 0) != hipSuccess) {
        printf("no HIP device visible\n");
        return 7;
    }
    const char *built = getenv("KNJ_BUILD_ARCH");
    if (!built || !*built) {
        printf("ARCH GUARD NOT CONFIGURED: set KNJ_BUILD_ARCH to the arch you"
               " built for. An unguarded run is not a measurement.\n");
        return 7;
    }
    if (strncmp(p.gcnArchName, built, strlen(built)) != 0) {
        printf("ARCH MISMATCH: built for %s, device is %s\n",
               built, p.gcnArchName);
        printf("SKIPPED -- not run, not measured, not a result.\n");
        return 6;
    }
    snprintf(out->name, sizeof(out->name), "%s", p.name);
    snprintf(out->arch, sizeof(out->arch), "%s", p.gcnArchName);
    out->warp           = p.warpSize;
    out->cu             = p.multiProcessorCount;
    out->threads_per_cu = p.maxThreadsPerMultiProcessor;
    return 0;
}

#endif  // KNJ_WMMA_GUARD_H
