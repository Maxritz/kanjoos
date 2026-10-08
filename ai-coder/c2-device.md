# Worksheet — C2 Device abstraction layer

**What it owns:** the arch split. Everything above C2 is arch-agnostic.

**Why it exists (from `01` §7):** the engine ships one architecture and two kernel
backends behind C2. Policy parameters differ per arch; the component graph does not.
C2 is the gate that keeps arch-specific code from leaking above it.

**Spec lines that must hold:**
* `02-components.md` — C2 section (verbatim DeviceCaps, Backend API, rule, done-when).
* `00-verified-facts.md` §1 — measured ISA matrix: what each arch actually has.
* `01-architecture.md` §7 — what is different between the two targets.

**DeviceCaps (verbatim):**
```cpp
struct DeviceCaps {
    char     gcn_arch[16];      // "gfx1031" | "gfx1201"
    bool     has_wmma;          // probed, not assumed
    bool     has_fp8;
    bool     has_dot_builtin;   // sdot4 compiled
    int      wave_size;         // 32 on both targets
    uint64_t vram_total, vram_usable;
    bool     rebar_full_aperture;
    uint32_t sms, clock_khz;
    uint32_t max_streams, num_queue_groups;
};
```

**Backend (verbatim):**
```cpp
class Backend {                      // virtual, one impl per arch
public:
    static std::unique_ptr<Backend> create();   // probes, never guesses
    const DeviceCaps& caps() const;
    virtual void launch_attention(const AttentionPlan&) = 0;
    virtual void launch_expert_gemm(const ExpertPlan&) = 0;
    virtual void launch_kv_write(...) = 0;
};
```

**Rule (verbatim):** caps come from a `tools/isa_probe`-equivalent runtime probing
compiled into the binary, cross-checked against `hipDeviceProp_t`. A kernel that
needs a cap the device does not advertise is never instantiated.

**Measured facts this component must reflect (not assume):**
* gfx1201 **has** WMMA — `v_wmma_f32_16x16x16_f16` via the `_gfx12` builtin +
  `+wavefrontsize32` (`00` §1.1, §7.3). A and B are 4 VGPRs each (8×f16); D is
  8×f32 per lane. gfx1031 has **no** WMMA on either spelling (`00` §1.4).
* gfx1031 `sdot4` builtin lowers to native `v_dot4c_i32_i8` (`00` §1.2) — a single
  hardware dot-4, not a shift/add expansion.
* Inline-asm dot variants (`v_dot4_i32_i8`, `v_dot8_i32_i4`, `v_dot2c_f32_f16`)
  assemble cleanly on **both** arches (`00` §1.3), but note `v_dot2c_f32_f16` does **not** link on either — L-FAIL on both with `invalid operand`/`instruction not supported`; see `00` §7.2a. Use `v_dot4_i32_i8` / `v_dot8_i32_i4` as the shareable asm dots. Dot products can be written once
  in asm and shared; the builtin is strictly worse.
* gfx1201 has native FP8 (2× FP16 throughput); gfx1031 has **no** FP8 (`00` §1.4,
  `01` §7).
* wave_size = 32 on both targets. `+wavefrontsize32` is inert on gfx1201.

**Acceptance test (verbatim "done when"):**
* The same binary, on either box, selects the right backend and prints the capability
  table that C21 shows in the profile header.

**Gates:**
* None for the interface — the cap set is measured, not assumed. The "done when" is a
  real integration test on both boxes.
* **Do not** implement a backend that uses WMMA on gfx1031 (no matrix units).
* **Do not** implement a backend that uses FP8 on gfx1031 (unavailable).
* The gfx1201 backend **may** use WMMA, but see the gating notes below.

**Coder notes / pitfalls:**
* `has_wmma` is probed, not hardcoded. A gfx1201 binary may run on a future card
  that lacks WMMA; the backend must degrade, not assert.
* Cross-check probed caps against `hipDeviceProp_t.gcnArchName` — and enforce the
  arch guard: a binary built for gfx1031 that runs on gfx1201 dispatches nothing and
  lies (see `00` §8.7). C2's `create()` must verify `KNJ_BUILD_ARCH` matches
  `gcnArchName` before instantiating a device backend, or the rest of the engine is
  running on a lie.
* Pass `KNJ_BUILD_ARCH` through the **environment**, not `-D` (hipcc eats quotes;
  `-DKNJ_BUILD_ARCH="gfx1201"` fails with "use of undeclared identifier").
* The gfx1201 WMMA path is gated behind the `_gfx12` operand-layout question
  (`00` §7.4a): the builtin does **not** consume A as 8 consecutive f16. A tiled
  expert GEMM written on that assumption reads the wrong elements. Until P1-3 settles
  the layout (A/B against the Tensile oracle), do not build a WMMA expert GEMM that
  assumes the wrong layout — it would be silently wrong. The SIMT path stays compiled
  in as the fallback (`08` risk register: "WMMA numerics diverge from SIMT — high
  impact, mitigation: SIMT path stays a compiled-in fallback; P1-3 A/Bs them against
  the Tensile oracle").

**Worked micro-example (host, sanity-check before device):**
Probe a fake device that advertises a cap subset, instantiate Backend::create(), and
assert: (1) the printed capability table matches the fake device's advertised caps;
(2) a kernel that requires `has_wmma` is never instantiated when the device advertises
`has_wmma=false`; (3) `KNJ_BUILD_ARCH` mismatch causes create() to refuse (exit 6
equivalent) before any device backend is handed out; (4) passing `KNJ_BUILD_ARCH`
through the environment works, but `-D` does not (the environment path is the supported
spell).
