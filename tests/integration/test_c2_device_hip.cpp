// tests/integration/test_c2_device_hip.cpp -- HIP device integration test for
// the C2 Device abstraction layer.
//
// This is the test that exercises the C2 table emitters on a REAL device,
// cross-checking the compile-time KNJ_HAS_* facts (probed by
// tools/isa_probe/run_isa_probe.sh and baked into the binary by cmake) against
// hipDeviceProp_t. It is the end-to-end check that the host-only C2 library
// and the HIP device path agree on what the attached card can do.
//
// EXIT CODES (same contract as tests/test_gemm_w4.hip, docs/06 §8 / 07 §8):
//   0  PASS -- compile-time caps match device properties, doctor header printed
//   5  FAIL -- a capability claim is contradicted by the device
//   6  DECLINED -- built for a different arch than the attached device (skip)
//   7  REFUSED -- KNJ_BUILD_ARCH is not set (must not run)
//
// Build: hipcc -nogpulib -O2 --offload-arch=<arch> -L'<rocm>/lib' -lamdhip64
//        knj_device.a test_c2_device_hip.cpp -o test_c2_device_hip
//
// The arch guard uses getenv("KNJ_BUILD_ARCH"), not a -D, for the same reason
// the GEMM driver does (docs/00-verified-facts.md section 8.7): hipcc
// re-spawns clang through a command string and a -D carrying quotes does not
// survive that layer.

#define _CRT_SECURE_NO_WARNINGS 1

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "knj_arch_config.h"
#include "device_caps.h"
#include "backend.h"

// --- helpers ---------------------------------------------------------------

#define CHECK_HIP(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), \
                 __FILE__, __LINE__); return 5; } } while (0)

// Human-readable yes/no for a bool.
static const char* yn(bool v) { return v ? "yes" : "no"; }

// --- main -----------------------------------------------------------------

int main() {
    // ---- 1. arch guard (first, before anything else) --------------------
    const char* build_arch = std::getenv("KNJ_BUILD_ARCH");
    if (!build_arch || !*build_arch) {
        std::fprintf(stderr,
            "KNJ_BUILD_ARCH is not set. Refusing to run: a driver that cannot "
            "prove which arch it was built for must not touch the device.\n");
        return 7;
    }

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));

    std::printf("build arch   : %s\n", build_arch);
    std::printf("device arch  : %s\n", prop.gcnArchName);
    std::printf("device       : %s\n", prop.name);

    if (std::strcmp(build_arch, prop.gcnArchName) != 0) {
        std::fprintf(stderr,
            "DECLINED: built for %s, attached device is %s. This is a skip, "
            "not a result.\n", build_arch, prop.gcnArchName);
        return 6;
    }

    // Compiled-arch check. KNJ_BUILD_ARCH must also agree with the arch this
    // binary was compiled for (KNJ_ARCH_NAME, from knj_arch_config.h). A
    // disagreement means the binary targets a different arch than it claims --
    // the same defect as the device mismatch above, so it is the same SKIP
    // (exit 6), not a capability failure (exit 5). Measured failure mode:
    // docs/00-verified-facts.md section 8.7 (a gfx1031 code object on the
    // gfx1201 card dispatches nothing and prints confident nonsense).
    if (std::strcmp(build_arch, KNJ_ARCH_NAME) != 0) {
        std::fprintf(stderr,
            "DECLINED: binary compiled for %s, KNJ_BUILD_ARCH claims %s. "
            "Built-for-a-different-arch is a skip, not a result.\n",
            KNJ_ARCH_NAME, build_arch);
        return 6;
    }

    // ---- 2. host-side C2 probe (knj_probe_host_caps) --------------------
    // This exercises the SAME code path the unit tests exercise, but now with
    // KNJ_BUILD_ARCH set to the attached device's arch, so the arch guard
    // passes and the host probe returns the compiled-in capability set.
    DeviceCaps host_caps = knj_probe_host_caps();

    if (!host_caps.arch_ok) {
        std::fprintf(stderr, "FAIL: host arch guard refused: %s\n",
                     host_caps.arch_reason ? host_caps.arch_reason : "(unknown)");
        return 5;
    }

    std::printf("\n=== C2 host probe (compile-time facts, env-driven) ===\n");
    std::printf("  gcn_arch        %s\n", host_caps.gcn_arch);
    std::printf("  has_wmma        %s\n", yn(host_caps.has_wmma));
    std::printf("  has_fp8         %s\n", yn(host_caps.has_fp8));
    std::printf("  has_dot_builtin %s\n", yn(host_caps.has_dot_builtin));
    std::printf("  wave_size       %d\n", host_caps.wave_size);
    std::printf("  arch_ok         %s\n", yn(host_caps.arch_ok));

    // ---- 3. device-side properties (hipDeviceProp_t) ---------------------
    // These are the GROUND TRUTH from the driver. The compile-time facts must
    // not contradict them.

    // CU count.
    const unsigned cu_count = (unsigned)prop.multiProcessorCount;
    // Wave size. This ROCm version's hipDeviceProp_t exposes warpSize (CUDA-
    // compatible naming). On RDNA4/gfx1201 HIP reports warpSize=32 (native
    // wave32); on RDNA2/gfx1031 the same. Defensive default if unexpected.
    int wave = (int)prop.warpSize;
    if (wave != 32 && wave != 64) { wave = 64; }

    // VRAM: HIP reports totalGlobalMem in bytes.
    const double vram_total_gib =
        (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);

    // ReBAR: this ROCm version's hipDeviceProp_t does not expose flags /
    // devPropRebbar / devPropZovHostSwappable. We report ReBAR as unknown
    // (matching the host probe's default of "no full aperture" until a
    // platform-level probe runs). A definitive ReBAR probe is a platform step
    // in tools/doctor/kanjoos_doctor.sh, not a compile-time check.
    const bool rebar_reported = false;  // unknown on this ROCm version

    std::printf("\n=== Device properties (hipDeviceProp_t) ===\n");
    std::printf("  CU count        %u\n", cu_count);
    std::printf("  wave size       %d\n", wave);
    std::printf("  VRAM total      %.2f GiB\n", vram_total_gib);
    std::printf("  ReBAR reported  %s\n", yn(rebar_reported));
    std::printf("  PCI bus/device  %04x:%04x\n", prop.pciBusID, prop.pciDeviceID);

    // ---- 4. cross-check: compile-time caps vs device reality -------------
    // The compile-time fact is the ISA probe result baked in by cmake. The
    // device property is what the driver says. They must agree. Where the
    // device cannot express a fact (e.g. "does this card have the sdot4
    // builtin?"), we trust the ISA probe and document why.

    int failures = 0;

    // WMMA: gfx1201 has it, gfx1031 does not. The device prop does not expose
    // "has WMMA" directly, so the ISA probe is authoritative here. We verify
    // the arch is the one we expect for the claimed WMMA state.
    {
        const bool want_wmma = (std::strcmp(build_arch, "gfx1201") == 0);
        if (host_caps.has_wmma != want_wmma) {
            std::fprintf(stderr,
                "FAIL: has_wmma=%d but arch=%s expects %d\n",
                host_caps.has_wmma, build_arch, want_wmma);
            ++failures;
        } else {
            std::printf("\n[check] has_wmma = %d  (arch %s -> expected %d)  OK\n",
                        host_caps.has_wmma, build_arch, want_wmma);
        }
    }

    // FP8: gfx1201 has it, gfx1031 does not.
    {
        const bool want_fp8 = (std::strcmp(build_arch, "gfx1201") == 0);
        if (host_caps.has_fp8 != want_fp8) {
            std::fprintf(stderr,
                "FAIL: has_fp8=%d but arch=%s expects %d\n",
                host_caps.has_fp8, build_arch, want_fp8);
            ++failures;
        } else {
            std::printf("[check] has_fp8 = %d  (arch %s -> expected %d)  OK\n",
                        host_caps.has_fp8, build_arch, want_fp8);
        }
    }

    // Dot: both arches have the inline-asm v_dot4_i32_i8 path (KNJ_HAS_DOT_ASM
    // is 1 for both). The dot path is selected at runtime by has_dot_builtin
    // (sdot4 builtin present on gfx1031, refused on gfx1201).
    {
        const bool is1031 = (std::strcmp(build_arch, "gfx1031") == 0);
        const bool want_dot_builtin = is1031;  // sdot4 builtin compiles on gfx1031
        if (host_caps.has_dot_builtin != want_dot_builtin) {
            std::fprintf(stderr,
                "FAIL: has_dot_builtin=%d but arch=%s expects %d\n",
                host_caps.has_dot_builtin, build_arch, want_dot_builtin);
            ++failures;
        } else {
            std::printf("[check] has_dot_builtin = %d  (arch %s -> expected %d)  OK\n",
                        host_caps.has_dot_builtin, build_arch, want_dot_builtin);
        }
    }

    // Wave size: both targets are wave32.
    {
        if (wave != 32) {
            std::fprintf(stderr,
                "FAIL: device reports wave size %d, expected 32 for %s\n",
                wave, build_arch);
            ++failures;
        } else {
            std::printf("[check] wave_size = %d  (expected 32)  OK\n", wave);
        }
        // The host probe hard-codes wave_size=32; the device must agree.
        if (host_caps.wave_size != 32) {
            std::fprintf(stderr,
                "FAIL: host probe wave_size=%d, expected 32\n",
                host_caps.wave_size);
            ++failures;
        }
    }

    // CU count sanity: RX 9070 XT is 32 CU. gfx1031 in a laptop/desktop config
    // varies; we report it but do not hard-fail on an unexpected CU count —
    // the CU count is a device fact, not a compile-time fact, and the device
    // may be a different card than the one the ISA probe was run on.
    std::printf("[info ] CU count = %u (device fact, not a compile-time claim)\n", cu_count);

    // ---- 5. doctor header (docs/06-profiling.md section 8 format) --------
    // This is the EXACT output shape the C2 emitters must produce. The unit
    // tests verify the emitters in isolation; this integration test verifies
    // them against a real device so the same strings appear on device.

    // Build a DeviceCaps that combines the host probe's compile-time facts with
    // the device's runtime facts (CU count, VRAM, ReBAR). This is the shape the
    // real kanjoos doctor produces (see C21 in docs/02-components.md).
    DeviceCaps combined = host_caps;
    combined.sms = cu_count;
    combined.vram_total = (uint64_t)(vram_total_gib * 1024.0 * 1024.0 * 1024.0);
    combined.rebar_full_aperture = rebar_reported;

    char doc_header[512];
    knj_print_doctor_header(combined, prop.name, true, doc_header, sizeof(doc_header));
    std::printf("\n=== kanjoos doctor (device) ===\n%s\n", doc_header);

    // Also print the host-only path (no device name) for the record.
    char doc_host[512];
    knj_print_doctor_header(combined, nullptr, true, doc_host, sizeof(doc_host));
    std::printf("=== kanjoos doctor (host-only label) ===\n%s\n", doc_host);

    // Capability line alone, for grep-ability.
    char cap_line[256];
    knj_print_cap_line(combined, true, cap_line, sizeof(cap_line));
    std::printf("=== capability line ===\n%s\n", cap_line);

    // ---- 6. Backend::create() on device ----------------------------------
    // This proves the arch-guarded factory works in the device binary: with
    // KNJ_BUILD_ARCH set to the attached device's arch, create() should hand
    // out a Backend.
    auto backend = Backend::create();
    if (!backend) {
        std::fprintf(stderr, "FAIL: Backend::create() returned nullptr on device\n");
        ++failures;
    } else {
        const DeviceCaps& bcaps = backend->caps();
        std::printf("\n[check] Backend::create() OK\n");
        std::printf("  backend gcn_arch  %s\n", bcaps.gcn_arch);
        std::printf("  backend has_wmma  %s\n", yn(bcaps.has_wmma));
        std::printf("  backend arch_ok   %s\n", yn(bcaps.arch_ok));

        if (std::strcmp(bcaps.gcn_arch, build_arch) != 0) {
            std::fprintf(stderr, "FAIL: backend gcn_arch=%s != build_arch=%s\n",
                         bcaps.gcn_arch, build_arch);
            ++failures;
        }
        if (!bcaps.arch_ok) {
            std::fprintf(stderr, "FAIL: backend arch_ok=false after create() succeeded\n");
            ++failures;
        }

        // would_instantiate checks.
        std::printf("\n[check] would_instantiate:\n");
        std::printf("  WmmaF16  %s\n", yn(backend->would_instantiate(KnjCap::WmmaF16)));
        std::printf("  Fp8      %s\n", yn(backend->would_instantiate(KnjCap::Fp8)));
        std::printf("  Dot4I32I8 %s\n", yn(backend->would_instantiate(KnjCap::Dot4I32I8)));

        if (backend->would_instantiate(KnjCap::WmmaF16) != host_caps.has_wmma) {
            std::fprintf(stderr, "FAIL: would_instantiate(WmmaF16) disagrees with has_wmma\n");
            ++failures;
        }
        if (backend->would_instantiate(KnjCap::Fp8) != host_caps.has_fp8) {
            std::fprintf(stderr, "FAIL: would_instantiate(Fp8) disagrees with has_fp8\n");
            ++failures;
        }
    }

    // ---- 7. verdict -------------------------------------------------------
    std::printf("\n=== Verdict ===\n");
    if (failures == 0) {
        std::printf("RESULT: PASS (exit 0)\n");
        return 0;
    } else {
        std::printf("RESULT: FAIL (%d check(s) contradicted)  (exit 5)\n", failures);
        return 5;
    }
}
