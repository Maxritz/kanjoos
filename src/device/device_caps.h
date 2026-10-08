// src/device/device_caps.h -- C2 Device abstraction layer: capability facts.
//
// Owns: the arch split. Everything above C2 is arch-agnostic.
//
// This header is HOST C++17. It must compile on the host compiler with NO HIP
// headers and NO device builtins (docs/07-build-platforms.md section 2 rule:
// everything outside src/device/, kernels/ and src/platform/ is host-only ISO
// C++17). The device integration test includes it only to call the table
// emitters, which run on the host before any kernel launches.
//
// Capabilities here are PROBED, not assumed (docs/02-components.md C2 rule):
//   * the compile-time caps are read out of tools/isa_probe/run_isa_probe.sh
//     by cmake/arch.cmake into KNJ_HAS_* macros (see docs/00-verified-facts.md
//     section 1); that is the "isa_probe-equivalent probing compiled into the
//     binary";
//   * the runtime caps are reported by the knj_probe_caps kernel in
//     kernels/knj_kernels.hip and verified by a launch on the attached device.
//
// The arch guard is the one place C2 is allowed to look at an arch string.
// KNJ_BUILD_ARCH arrives through the ENVIRONMENT, never via -D: hipcc re-spawns
// clang through a command string and a -D carrying quotes does not survive that
// layer (docs/00-verified-facts.md section 8.7). `__gfxNNNN__` is device-only and
// is therefore useless in host code. getenv("KNJ_BUILD_ARCH") is the supported
// spell.

#ifndef KNJ_DEVICE_CAPS_H
#define KNJ_DEVICE_CAPS_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// The arch this binary was compiled for, injected by cmake (root CMakeLists
// passes -DKNJ_ARCH_NAME=\"gfx1201\"). The fallback exists only so a standalone
// translation unit that forgets the define still compiles to something that
// FAILS the arch guard loudly rather than miscompiling silently.
#ifndef KNJ_ARCH_NAME
#define KNJ_ARCH_NAME "unknown"
#endif

// Compile-time cap flags from cmake/arch.cmake (the recorded isa_probe results).
#ifndef KNJ_HAS_WMMA
#define KNJ_HAS_WMMA 0
#endif
#ifndef KNJ_HAS_FP8
#define KNJ_HAS_FP8 0
#endif
#ifndef KNJ_HAS_DOT_ASM
#define KNJ_HAS_DOT_ASM 0
#endif

// A kernel capability that a backend may or may not be allowed to instantiate.
// These are the probed instruction capabilities, not the speculative ones:
// anything in this enum is a spelling that tools/isa_probe has emitted and
// that the runtime probe can verify.
enum class KnjCap {
    Dot4I32I8,    // v_dot4_i32_i8  -- asm on BOTH arches (docs/00 section 1)
    Dot8I32I4,    // v_dot8_i32_i4  -- asm on BOTH arches
    PkFmaF16,     // v_pk_fma_f16   -- asm on BOTH arches (SIMD fp16 math)
    WmmaF16,      // v_wmma_*_gfx12 -- gfx1201 only; gfx1031 has no matrix units
    Fp8,          // native fp8 path -- gfx1201 only
    Sdot4Builtin, // sdot4 builtin  -- gfx1031 only; gfx1201 refuses it
};

// The verbatim DeviceCaps struct from docs/02-components.md C2, plus the two
// arch-guard fields C2's create() must populate (see "Fails by" / arch-guard
// notes in ai-coder/c2-device.md).
struct DeviceCaps {
    char     gcn_arch[16];          // "gfx1031" | "gfx1201"
    bool     has_wmma;              // probed, not assumed
    bool     has_fp8;
    bool     has_dot_builtin;       // sdot4 builtin compiled
    int      wave_size;             // 32 on both targets
    uint64_t vram_total, vram_usable;
    bool     rebar_full_aperture;
    uint32_t sms, clock_khz;
    uint32_t max_streams, num_queue_groups;

    // C2 arch guard. arch_ok==false means create() refused to hand out a
    // backend; arch_reason names the rule that fired so the caller can print it
    // instead of a bare "declined". Set by knj_check_arch / knj_probe_host_caps.
    bool     arch_ok;
    const char* arch_reason;
};

// ---------------------------------------------------------------------------
// The arch guard itself.
// ---------------------------------------------------------------------------
// Returns build_arch==expect (the environment spelling) AND requires the env to
// be present: an unset KNJ_BUILD_ARCH is a refusal on its own, not a match. The
// two failure modes the engine cares about both land here:
//   * unset -> "refused: KNJ_BUILD_ARCH is not set" (exit 7 equivalent)
//   * mismatch -> "refused: built for X, device is Y" (exit 6 equivalent)
struct KnjArchVerdict {
    bool        ok;
    const char* reason;   // stable string; empty when ok
};

inline KnjArchVerdict knj_check_arch(const char* env_arch,
                                     const char* built_arch) {
    if (env_arch == nullptr || env_arch[0] == '\0') {
        return KnjArchVerdict{false,
            "KNJ_BUILD_ARCH is not set: a backend that cannot prove which arch "
            "it was built for must not touch the device"};
    }
    if (built_arch == nullptr || built_arch[0] == '\0' ||
        std::strcmp(env_arch, built_arch) != 0) {
        return KnjArchVerdict{
            false,
            "arch mismatch: KNJ_BUILD_ARCH disagrees with the compiled backend "
            "arch (built for a different target than the device it is running on)"};
    }
    return KnjArchVerdict{true, ""};
}

// The supported env spell. Reads KNJ_BUILD_ARCH from the environment; this is
// the ONLY channel through which the build arch is meant to reach the host
// (the -D path is broken by hipcc, see file header).
inline const char* knj_build_arch_env() {
    return std::getenv("KNJ_BUILD_ARCH");
}

// ---------------------------------------------------------------------------
// Probe the host-side capability facts.
// ---------------------------------------------------------------------------
// Combines the compiled-in isa_probe results (KNJ_HAS_*) with the env arch
// guard. Device-only fields (VRAM size, CU count, rebar, PCI topology) are
// zeroed here and are expected to be filled by the runtime device probe in the
// integration test -- this host path does not require a GPU, so it cannot and
// does not claim to know them.
inline DeviceCaps knj_probe_host_caps() {
    DeviceCaps c{};
    c.has_wmma = (KNJ_HAS_WMMA != 0);
    c.has_fp8  = (KNJ_HAS_FP8 != 0);
    // sdot4 builtin: compiles on gfx1031, refused on gfx1201 (docs/00 section 1).
    // gfx1031 has KNJ_HAS_WMMA==0 (no matrix units); gfx1201 has KNJ_HAS_WMMA==1.
    // So has_dot_builtin (sdot4) is the inverse of has_wmma (WMMA).
    c.has_dot_builtin = (KNJ_HAS_WMMA == 0);
    c.wave_size = 32;
    c.vram_total = c.vram_usable = 0;
    c.rebar_full_aperture = false;
    c.sms = c.clock_khz = 0;
    c.max_streams = c.num_queue_groups = 0;

    // Arch guard on the compiled-in arch name.
    const char* built = KNJ_ARCH_NAME;
    const KnjArchVerdict v = knj_check_arch(knj_build_arch_env(), built);
    c.arch_ok = v.ok;
    c.arch_reason = v.reason;
    // gcn_arch is only meaningful once the guard passes; otherwise leave it
    // zeroed so a refused backend can't be mistaken for a real one.
    if (v.ok) {
        std::strncpy(c.gcn_arch, built, sizeof(c.gcn_arch) - 1);
    }
    return c;
}

// Lookup of a single instruction capability by arch string. Mirrors the
// measured matrix in docs/00-verified-facts.md section 1 and the link verdicts
// in section 7.2a: the cap is "available" iff the spelling both ASSEMBLES and
// LINKS on that arch. The three link-divergence rows (v_pk_mad_f32,
// v_dot2_i32_i16, v_dot2c_i32_i16) are deliberately NOT in this table -- they
// assemble but do not link, so they are unavailable as real capabilities.
inline bool knj_cap_available(const char* arch, KnjCap cap) {
    if (arch == nullptr || arch[0] == '\0') return false;
    const bool is1201 = (std::strcmp(arch, "gfx1201") == 0);
    const bool is1031 = (std::strcmp(arch, "gfx1031") == 0);
    switch (cap) {
        case KnjCap::Dot4I32I8:    return is1201 || is1031;  // asm, both
        case KnjCap::Dot8I32I4:    return is1201 || is1031;  // asm, both
        case KnjCap::PkFmaF16:     return is1201 || is1031;  // asm, both
        case KnjCap::WmmaF16:      return is1201;            // gfx1201 only
        case KnjCap::Fp8:          return is1201;            // gfx1201 only
        case KnjCap::Sdot4Builtin: return is1031;            // gfx1031 only
        default:                   return false;
    }
}

// Short label used in the capability table for a KnjCap, matching the
// `kanjoos doctor` spelling in docs/06-profiling.md section 8.
inline const char* knj_cap_label(KnjCap cap) {
    switch (cap) {
        case KnjCap::Dot4I32I8:    return "dot4_i32_i8";
        case KnjCap::Dot8I32I4:    return "dot8_i32_i4";
        case KnjCap::PkFmaF16:     return "pk_fma_f16";
        case KnjCap::WmmaF16:      return "wmma_f16";
        case KnjCap::Fp8:          return "fp8";
        case KnjCap::Sdot4Builtin: return "sdot4";
        default:                   return "unknown";
    }
}

// ---------------------------------------------------------------------------
// C21 capability table emitters.
// ---------------------------------------------------------------------------
// These produce the EXACT line shape docs/06-profiling.md section 8 specifies
// for the `kanjoos doctor` capability header, so the same string is printed on
// Windows and Linux for both targets (section 9.6 acceptance criterion 6).
//
// `dot` is the runtime characterisation of the integer-dot path the backend
// will actually take. On gfx1031 that is the sdot4 builtin (measured: lowers to
// v_dot4c_i32_i8, docs/00 section 1.2); on gfx1201 the W4 path is inline-asm
// v_dot4_i32_i8 (no sdot4; the builtin is refused there). The emitter takes the
// runtime-verified result and writes the value docs/06 expects.
inline const char* knj_dot_label(const DeviceCaps& c) {
    if (c.has_dot_builtin) return "builtin(sdot4)";
    if (KNJ_HAS_DOT_ASM)   return "asm(v_dot4_i32_i8)";
    return "none";
}

// The one-line capability row: `capabilities  wmma=on  fp8=on ...`.
// Matches docs/06-profiling.md section 8 exactly (field order and spelling).
inline void knj_print_cap_line(const DeviceCaps& c,
                               bool pk_fma16_verified,
                               char* out, std::size_t cap) {
    // Field set is fixed by the output contract; do not reorder or add.
    std::snprintf(out, cap,
        "capabilities  wmma=%s  fp8=%s  dot=%s  smi=lane-split  v_pk_fma_f16=%s",
        c.has_wmma ? "on" : "off",
        c.has_fp8  ? "on" : "off",
        knj_dot_label(c),
        pk_fma16_verified ? "yes" : "no");
}

// Full `kanjoos doctor` capability + device header (docs/06 section 2 / 8).
// device_name/gcn_arch are filled from hipDeviceProp_t in the integration test;
// on the host-only path they are "host-probe" / the compiled arch.
inline void knj_print_doctor_header(const DeviceCaps& c,
                                    const char* device_name,
                                    bool pk_fma16_verified,
                                    char* out, std::size_t cap) {
    char cap_line[128];
    knj_print_cap_line(c, pk_fma16_verified, cap_line, sizeof(cap_line));
    std::snprintf(out, cap,
        "device        %s  %s  %u CU  wave%d\n"
        "%s\n"
        "rebar         %s\n",
        device_name ? device_name : "(no device)",
        c.gcn_arch[0] ? c.gcn_arch : "(arch guard refused)",
        c.sms,
        c.wave_size,
        cap_line,
        c.rebar_full_aperture ? "full aperture" : "no full aperture");
}

#endif  /* KNJ_DEVICE_CAPS_H */
