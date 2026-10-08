// src/device/backend.h -- C2 Device abstraction layer: Backend.
//
// Owns: the arch split. Everything above C2 is arch-agnostic, so the backend is
// the only place that may branch on arch. `Backend::create()` performs the
// mandatory arch guard (docs/02-components.md C2 / ai-coder/c2-device.md): it
// refuses to hand out a backend unless KNJ_BUILD_ARCH is set AND matches the
// arch the binary was compiled for. A backend that cannot prove its target must
// not touch the device -- a binary built for one arch that runs on another
// dispatches nothing and lies (docs/00-verified-facts.md section 8.7).
//
// The cap-gating rule from docs/02-components.md is enforced here too: a kernel
// that needs a cap the device does not advertise is never instantiated. That is
// `would_instantiate(KnjCap)` and the `launch_*` methods, both of which read the
// caps off the injected DeviceCaps.
//
// This is HOST C++17. It compiles on the host compiler with no HIP headers.

#ifndef KNJ_BACKEND_H
#define KNJ_BACKEND_H

#include "device_caps.h"

#include <cstddef>
#include <memory>

// Minimal plan types. The real ExpertPlan / AttentionPlan / KvWritePlan live in
// C17 / C16 / C14/C15 and are not built yet; C2 owns the GATE, not the kernels.
// These stand-ins exist only so Backend can keep the spec's virtual interface
// shape and so the unit test can exercise the gating without dragging the op
// graph in. They are intentionally empty.
struct AttentionPlan {};
struct ExpertPlan {};
struct KvWritePlan {};

// `Backend` is the abstract device handle. One concrete impl is selected by
// `create()`; the host unit test injects caps through `from_caps()` instead.
class Backend {
public:
    // Arch-guarded factory. Refuses (returns nullptr) when:
    //   * KNJ_BUILD_ARCH is unset/empty         -> knj_last_refusal() set
    //   * KNJ_BUILD_ARCH != KNJ_ARCH_NAME         -> knj_last_refusal() set
    // The refusal strings are the human text the driver prints (docs/06-
    // profiling.md section 8 / 07-build-platforms.md section 8).
    static std::unique_ptr<Backend> create();

    // Test hook: build a backend from an INJECTED DeviceCaps, bypassing the
    // environment arch guard. Used by tests/unit/test_c2_device.cpp to feed a
    // fake device that advertises a cap subset.
    static std::unique_ptr<Backend> from_caps(const DeviceCaps& caps);

    const DeviceCaps& caps() const noexcept { return caps_; }

    // The cap-gating question the worked example in ai-coder/c2-device.md §1.3
    // checks: would a kernel that needs `cap` be instantiated on this backend?
    bool would_instantiate(KnjCap cap) const noexcept;

    // launch_expert_gemm_wmma needs the WMMA path (gfx1201 only). Refused
    // outright when has_wmma is false -- the kernel is never instantiated.
    virtual bool launch_expert_gemm_wmma(const ExpertPlan&) = 0;
    // launch_expert_gemm_simt uses the inline-asm integer dots, which are
    // available on BOTH arches. No WMMA, no FP8; this is the fallback path.
    virtual bool launch_expert_gemm_simt(const ExpertPlan&) = 0;
    virtual bool launch_attention(const AttentionPlan&) = 0;
    virtual void launch_kv_write(const KvWritePlan&) = 0;

    // Why create() refused, if it did. Empty string when create() succeeded.
    // Thread-local so the message survives the move-out of the unique_ptr.
    static const char* knj_last_refusal();

    virtual ~Backend() = default;

protected:
    explicit Backend(const DeviceCaps& caps) : caps_(caps) {}

    DeviceCaps caps_;
};

#endif  /* KNJ_BACKEND_H */
