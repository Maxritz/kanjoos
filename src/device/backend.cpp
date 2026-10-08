// src/device/backend.cpp -- implementation of the C2 Backend.

#include "backend.h"

#include <cstdio>
#include <cstring>

// Thread-local refusal reason surfaced by create(). Set when create() returns
// nullptr and consumed by the driver (or the unit test) to print a message
// instead of a bare "declined".
namespace {
thread_local char g_refusal[128] = "";
}

const char* Backend::knj_last_refusal() {
    return g_refusal;
}

// The concrete backend. One impl per arch is the intent (docs/07-build-platforms
// section 2: src/device/backend_gfx1201, backend_gfx1031), but C2 in this phase
// owns only the GATE: the cap set is probed and the same HostBackend serves
// both, selecting simt vs wmma through cap checks rather than a second
// translation unit. Separating the two backends is a C3 refactor, not a C2
// one, and the gating tests here do not depend on it.
class HostBackend final : public Backend {
public:
    using Backend::Backend;

    // WMMA path: refused when the device did not advertise has_wmma. This is the
    // "kernel that needs a cap the device does not advertise is never
    // instantiated" rule made literal -- a false return means "not built for
    // this device".
    bool launch_expert_gemm_wmma(const ExpertPlan&) override {
        return caps_.has_wmma;
    }

    // SIMT path: integer dots via inline asm (v_dot4_i32_i8), which assembles and
    // links on BOTH arches (docs/00 section 1 / 7.2a). Always available once the
    // arch guard has passed; the gate is the arch guard, not a per-cap check.
    bool launch_expert_gemm_simt(const ExpertPlan&) override {
        return caps_.arch_ok;
    }

    bool launch_attention(const AttentionPlan&) override {
        return caps_.arch_ok;
    }

    void launch_kv_write(const KvWritePlan&) override {
        // No-op stub: KV write lives in C15/C16; C2 only asserts it is permitted
        // to run (the arch guard). Returning void matches docs/02's signature.
    }
};

std::unique_ptr<Backend> Backend::create() {
    const DeviceCaps caps = knj_probe_host_caps();
    if (!caps.arch_ok) {
        std::snprintf(g_refusal, sizeof(g_refusal), "%s",
                      caps.arch_reason ? caps.arch_reason : "arch guard refused");
        return nullptr;
    }
    return std::unique_ptr<Backend>(new HostBackend(caps));
}

std::unique_ptr<Backend> Backend::from_caps(const DeviceCaps& caps) {
    g_refusal[0] = '\0';
    return std::unique_ptr<Backend>(new HostBackend(caps));
}

bool Backend::would_instantiate(KnjCap cap) const noexcept {
    // The gating rule: a capability not advertised by the device is never
    // instantiated. The arch string is the single source of truth for the
    // instruction set (a probed fact, not an assumption).
    const bool inst = knj_cap_available(caps_.gcn_arch, cap);
    if (!inst) return false;
    // The WMMA and FP8 paths are additionally gated on the probed device caps,
    // because a probed cap can be absent even when the arch nominally supports
    // it (a future card, a stripped ROCm build, etc. -- ai-coder/c2-device.md
    // "coder notes / pitfalls").
    switch (cap) {
        case KnjCap::WmmaF16: return caps_.has_wmma;
        case KnjCap::Fp8:     return caps_.has_fp8;
        default:              return true;
    }
}
