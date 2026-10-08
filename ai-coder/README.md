# AI-coder worksheets — Kanjoos component build sheets

One worksheet per component (C1–C24) plus the tooling and the doc-derived
operating-point decision. Each worksheet is self-contained: it carries the
component's ownership, interface, state machine, invariants, failure mode,
acceptance test, the exact spec lines it must satisfy, the existing code/tooling
it can lean on, the open measurement gates that gate "done", and a worked
micro-example so a coder can sanity-check before integrating.

How to use: hand a single worksheet to an AI coder (or write the component
yourself against it). Do **not** merge a component until its worksheet's
"acceptance test" passes on the real target (gfx1201) or is explicitly gated as
"gfx1031-only / not yet measurable". When a component needs another component's
interface, the worksheet states the interface contract — implement to the
contract, not to the current in-progress implementation.

## Read this first

* **Source of truth for numbers:** `docs/00-verified-facts.md` (measured).
  Every numeric claim in a worksheet that is not labelled **ASSUMPTION** traces
  back to a section there.
* **Source of truth for the operating point:** `docs/01-architecture.md` §4.2
  and `docs/00-verified-facts.md` §9.8 — W3 g128 is the operating point.
* **Source of truth for the KV architecture:** `docs/09-kv-engine-architecture.md`
  (supersedes `01` §2 and `04` §6 where they disagree).
* **Source of truth for interfaces:** `docs/02-components.md` — the C1–C24
  interface contracts are stable; implement to them.
* **Machine audit:** `python tools/check_docs.py` (exit 0 = doc matches tools)
  and `python tools/kvroof/kv_roofline.py models/qwen3-30b-a3b/config.json --check`
  (exit 0 = 71 claims agree). Run both before and after any component that
  touches a documented number.
* **Two artifacts for P0-7 source1:** `tools/route/route_local.json` (Qwen1.5-MoE-A2.7B-Chat,
  80.5% new-expert, 37% top-25%) is what `check_docs.py` and the doc quote.
  `tools/route/route_local_prose.json` (79.5%, 36%) is the replication-run artifact
  from the 5-language study. They are different runs of the same model — do not
  conflate them. The doc and the machine audit both use `route_local.json`.

## Status legend in each worksheet

* **MEASURED** — settled by a number in `00` §7–§9.
* **CLOSED** — the question is answered; only implementation remains.
* **PARTIAL** — measured on one axis, still open on another.
* **OPEN** — not yet settled; do not build the thing that depends on a settled answer until it is.
* **ASSUMPTION** — labelled as such in the docs; carry the settlement task.
* **GATED** — implementation is possible but "done" is gated behind a measurement that is not yet done.

## Component index

### Runtime and substrate
* [C1](c1-runtime.md) — Runtime core and stream scheduler
* [C2](c2-device.md) — Device abstraction layer (arch split)
* [C3](c3-autotuner.md) — ISA probe and kernel autotuner
* [C21](c21-profiler.md) — Profiler and telemetry (first-class deliverable, own doc `06-profiling.md`)
* [C24](c24-platform.md) — Platform layer (Windows + Linux)

### Model and expert pipeline
* [C4](c4-loader.md) — Model loader (GGUF → resident buffers + cold index)
* [C5](c5-compiler.md) — Expert compiler (offline: BF16 → kernel-native packed store)
* [C6](c6-directory.md) — Expert directory and NVMe object store
* [C18](c18-router.md) — Router (exactly the model's router, unbiased)

### Residency and transfer
* [C7](c7-host-tier.md) — Host tier: warm pool + bounded pinned DMA staging
* [C8](c8-slot-allocator.md) — GPU hot expert slot allocator
* [C9](c9-residency.md) — Residency manager (where every expert/KV page is)
* [C10](c10-prefetch.md) — Prefetch predictor (L0–L5)
* [C11](c11-transfer.md) — Transfer engine (all bytes that move)
* [C12](c12-admission.md) — Admission and eviction
* [C19](c19-cpu-fallback.md) — CPU / direct-memory expert fallback

### KV
* [C13](c13-radix.md) — Radix KV index (prefix reuse)
* [C14](c14-kv-cache.md) — KV cache engine (paged KV)
* [C15](c15-kv-codecs.md) — KV codecs (FP8/FP16/TurboQuant)
* [C16](c16-attention.md) — Attention kernels (prefill + decode)

### Compute
* [C17](c17-expert-gemm.md) — MoE expert GEMM

### Speculation and sessions
* [C20](c20-speculation.md) — Speculative decoding (MTP first)
* [C23](c23-session.md) — Session scheduler and server

### Memory budget
* [C22](c22-budget.md) — Memory budget manager (VRAM×RAM profile matrix)

### Tooling (already exist; worksheet = what to preserve/extend)
* [tools-isa-probe](tools-isa-probe.md)
* [tools-kvroof](tools-kvroof.md)
* [tools-route](tools-route.md)
* [tools-bench](tools-bench.md)
* [tools-i7](tools-i7.md)
* [operating-point](operating-point.md) — the W3 g128 decision from `01` §4.2 / `00` §9.8

## How to verify a worksheet before coding

For each worksheet, the "acceptance test" column is the gate. Before handing a
worksheet to a coder, confirm:

1. The spec lines cited actually exist at the cited paths (grep the doc).
2. The numeric claims cited are current (`check_docs.py` + `kv_roofline.py --check`).
3. The "depends on" gates are marked MEASURED/OPEN as stated; do not start a
   component whose gate is OPEN unless you intend to open a new question.
4. The worked micro-example is runnable on the host (CPU) if it is a CPU reference,
   or guarded for gfx1201 if it is a device test.

## Conventions for the coder

* Device code is HIP/C++17, compiled with the ROCm clang at
  `G:\ROCM10RT-gfx1201\lib\llvm\bin\clang.exe` (or the gfx1201 `hipcc`).
  `-nogpulib` is the default build mode — no device libm; `__expf` is the native
  quarter-rate instruction and is not required to be correctly rounded.
* **No `gridDim` in any kernel** on gfx1201 (`-nogpulib` → `__ockl_get_num_groups`
  unresolved). Grid-stride loops take the stride as a kernel argument.
* **No 64-bit indexing needed**; `uint32_t` suffices for 512 MB.
* `+wavefrontsize32` is inert on gfx1201 (natively wave32); do not enable it.
* Arch guard: every tier-C driver refuses to run unless `KNJ_BUILD_ARCH` matches
  `hipGetDeviceProperties().gcnArchName`. Mismatch → exit 6; unconfigured → exit 7.
  Pass the arch through the **environment**, not `-D` (hipcc re-spawns clang through
  a command string that eats quotes and a `-DKNJ_BUILD_ARCH="gfx1201"` spelling
  fails with "use of undeclared identifier").
* An arch-mismatched binary **runs and lies** on gfx1201 (kernel never dispatched,
  outputs keep their `hipMemset` value, timing table prints nonsense). So the guard
  is mandatory, and the full-output correctness sweep is the only thing that catches it.
* Check every output. A strided element sample, a passing exit code and a plausible
  number are all consistent with total nonsense. Full-output correctness against a
  host oracle is the default, not an enhancement.
* Per-page partials merged left-to-right, ascending pages, ascending keys within a
  page, global fp32 max first — that is the canonical reduction order (I7 contract).
  A single running accumulator produces different bits from identical inputs.
* A codec that changes values is a **declared precision reduction**, priced and
  recorded per page — never conflated with a tier move (I7 finding).
* Metadata is 3 B/group (fp16 scale + uint8 zero-point) for expert packs and KV
  group codecs, whatever the bit width.
