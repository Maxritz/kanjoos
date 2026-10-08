# 08 — Roadmap

Phases are ordered by **what unblocks the most**, not by what is most fun to
build. Phase 0 is almost entirely measurement, because the plan's two largest
performance questions — does gfx1201 get a WMMA path, and is RDNA2's int dot
real — are open, and both of them change what later phases should be built.

---

## Phase 0 — Measure before building (gate for everything)

No inference code. This phase exists so that P1–P7 are not built on assumptions.

| task | question | output |
|---|---|---|
| **P0-1** Platform census | PCIe gen/width, NVMe model + sequential + random read, host copy path, RAM profile, ReBAR state, queue topology | numbers in `00-verified-facts.md`; `kanjoos doctor` implemented (C24) |
| **P0-2** WMMA gate | **CLOSED** - gfx1201 emits `v_wmma_f32_16x16x16_f16` via the `_gfx12` builtin + `+wavefrontsize32`; gfx1031 has no matrix units | `00-verified-facts.md` �1.1 |
| **P0-3** RDNA2 dot | **CLOSED** - `sdot4` lowers to native `v_dot4c_i32_i8`, not a shift/add expansion | `00-verified-facts.md` �1.2 |
| **P0-4** ReBAR | full aperture on both targets, Windows and Linux | verified matrix |
| **P0-5** gfx1201 int8 | is the existing DP4A emulation still necessary given WMMA i8 exists? | measurement that either retires or keeps it |
| **P0-6** Transfer baseline | **PARTIALLY CLOSED** - measured 2026-10-05: 27.9 GB/s streaming pinned, **13.6 GB/s for a single 2.47 MB expert copy**. Remaining: D2H and pageable-vs-pinned at 2/4/8/16 MiB | `00-verified-facts.md` §8.1; extent + in-flight defaults |
| **P0-7** Routing locality | **CLOSED** - measured 2026-10-05 from two independent sources (real 48 gate matrices of Qwen3-30B-A3B applied to real hidden states; and a full self-consistent forward of Qwen1.5-MoE-A2.7B-Chat). **Answer: locality is weak.** 59-81% of a token's experts are new; 70% of the bank is needed for a 95% hit rate; one missed expert per layer costs 8.1-8.9 ms/token | `00-verified-facts.md` §9.6; `tools/route/route_locality.py`; `route_target-gates.json`, `route_local.json` |

**Consequence of P0-7, measured 2026-10-05 (`00-verified-facts.md` §9.6):** there
is no small hot set to keep. Partial expert residency costs 3-9 ms/token for a
few GiB of KV, which is a worse trade than a W4 to W3 repack — so **context
cannot be bought by evicting experts**. The lever is the weight format, and it
is now a capacity decision (W3 frees ~3.5 GiB, W2 frees ~6.6 GiB), not a
bandwidth one. Class B is not viable for experts either: same wire, same
verdict as for KV, reached independently.

**Consequence of P0-6, measured 2026-10-05 (`00-verified-facts.md` §8):** one
expert's arithmetic at 100% of derived peak is **7.68 us** against **181 us** to
move its weights over PCIe — transfer-bound by ~23x, even at the best case of 8
distinct experts. This is a settled input to Phase 1, not an open question.

- **WMMA and GEMM tiling are demoted** from "the thing to optimise" to "hours,
  not weeks". They do not move the decode number.
- **Weight bytes are the only lever that matters.** W2/W3, expert
  deduplication, and routing locality (P0-7) dominate everything else.

Two follow-ups have since closed the open ends (§8.9–8.10):

- The benchmark kernel was retuned from **0.8% to 5.2% of peak** (324 → 48.8 µs),
  and the number is now trustworthy because the *ceiling* is known: at M=32 the
  projection has **21.3 FLOP/byte against a machine balance of 66.7**, so
  **32% of peak is the hard limit regardless of the code**. Raising M is the
  only lever; tiling is not.
- The bit-width ladder is measured, and discouraging in a useful way: W4 → W2
  buys only **1.9×**, because the arithmetic side is already nearly free.
  Bit-width alone does not rescue this either. Touching fewer experts was the
  last large lever — and P0-7 has now measured it, and it is **negative**: the
  hit-rate curve is close to linear in slots, so there is no cheap subset to
  hold. Bit-width is therefore not merely the *only* lever left, it is the
  *whole* lever left.

**Exit gate:** P0-2 and P0-3 are already closed by measurement. P0-6 and P0-7
must have answers before Phase 1 is signed off. Both now do, and P0-7's answer
is the one that changes the plan's ambition: expert reuse across tokens is low,
so the engine's honest ceiling on a 12 GiB card drops and the design leans
harder on RAM
residency, CPU fallback and speculation.

---

## Phase 1 — Substrate

| component | deliverable |
|---|---|
| C1 | runtime, streams, op graph, event ring |
| C2 | backend abstraction + capability probing |
| C3 | autotuner with a persisted, arch-keyed cache |
| C21 | **profiler v1** — the table, the floor, `doctor` header |
| C24 | Windows + Linux platform layer, pinned pool, ReBAR checks |
| C4 | GGUF loader, cold index, trunk resident, no expert reads at load |

Building the profiler **first**, before there is anything to profile, is
deliberate: every later phase is verified by it.

**Exit gate:** an empty forward pass over a small dense model produces the
reference table with a non-zero measured floor, and the same binary runs on both
targets and both platforms.

---

## Phase 2 — W4 experts end to end (W3 is the operating point)

| component | deliverable |
|---|---|
| C5 | W4 group-128 compiler, kernel-native packing, manifest |
| C6 | expert directory, coalesced extents, journal, checksums |
| C7 | warm pool + bounded pinned pool |
| C8 | slot pool with event-bounded lifetimes |
| C17 | grouped W4 expert GEMM, validated against a host oracle |
| C9 | residency state machine, VRAM + RAM tiers |

The minimum first milestone, from the expert spec, unchanged: BF16 checkpoint →
W4 packing → NVMe store → RAM warm → VRAM slots → router selects → missing
experts prefetched asynchronously → grouped GEMM → all selected expert outputs
accumulated → matches the W4 reference within tolerance. **W4 is the first
implemented pack; W3 g128 is the operating point** (`00` §9.8, `01` §4.2) — the
W4 milestone is the vehicle that gets the engine to a real resident decode, and
the W3 repack is a one-function change on the same compiler once the milestone
lands. Do not delay Phase 2 waiting for W3; build W4, then repack.

---

## Phase 2b — W3 repack (one compiler change on the Phase 2 kernel)

The Phase 2 milestone builds W4 end to end. The operating point is W3 g128
(`00` §9.8, `01` §4.2): 11.79 GiB resident, 3.46 GiB KV budget, 37.8 K tokens
of FP16 context, 126 of 128 experts resident. Getting there from W4 is one
change in the compiler, on the same kernel and the same manifest shape.

| what changes | detail |
|---|---|
| compiler flag / constant | `W4_GROUP_EXPERT = 128` stays; the pack constant changes from
  `0.5 + 3/128 = 0.5234` B/weight to `3/8 + 3/128 = 0.3984` B/weight — i.e.
  the nibble payload becomes a 3-bit payload in the same 128-weight group with
  the same 3 B metadata (fp16 scale + uint8 zero-point). One constant in the
  pack header, one GEMM template instantiation. |
| kernel | unchanged — the same `v_dot8_i32_i4`-backed W4 kernel is reused;
  only the pack/unpack constants and the per-expert byte stride change. |
| manifest / store | unchanged shape — the canonical NVMe store is repacked offline
  by C5; the on-disk layout is the same 128-weight groups, just 3-bit payload
  instead of 4-bit. |
| SNR gate | **required, not optional.** The 00 §8.10 caveat records an 81.5 dB
  SNR penalty that killed the int4 activation path (`gemm_w4.hip`). A real W3
  design must beat that SNR penalty on the target prompts before the repack is
  admitted as the operating point. Bytes are the easy half of W3; accuracy is
  the half that has not been attempted. |
| exit criterion | the W3 resident decode matches the W4 reference within tolerance
  on deterministic prompts (same tolerance C17 uses for the W4 oracle), AND the
  SNR gate passes on the target corpus. If either fails, the operating point
  stays W4 and the KV budget stays zero — the engine does not silently trade
  accuracy for context. |

**Do not block Phase 2 on this.** Build W4, land the milestone, then repack to
W3 as a follow-on. The repack is cheap precisely because Phase 2 already landed
the compiler, the manifest and the GEMM; W3 is a constant change on top of an
existing path, not a new path.

---

## Phase 3 — KV: paging, radix reuse, codec

| component | deliverable |
|---|---|
| C14 | paged KV, layer slabs, block tables |
| C16 | attention prefill + decode, paged, GQA/MQA |
| C13 | radix index, chunked-hash-tree leaves, shared blocks |
| C15 | FP8 KV on gfx1201; FP16 on gfx1031; TurboQuant 4bit-nc behind a gate |

**Exit gate:** a KV page survives evict+reload bit-identically; a repeated-prefix
benchmark shows the TTFT win; FP8 KV is within tolerance of BF16 on deterministic
prompts. On hybrid models, rollback returns "replay suffix" (Tail-Replay shape)
rather than refusing.

---

## Phase 4 — Prediction and prefetch

| component | deliverable |
|---|---|
| C10 | L1 (previous token) → L3 (cross-layer) → L5 (adaptive horizon) |
| C11 | transfer engine: io_uring + DirectStorage/IOCP, coalescing, priorities |
| C12 | score-based eviction, NVMe admission control |
| C19 | CPU expert fallback with the transfer-vs-compute cost model |

**Exit gate:** `KV_MISS_STALL_TIME / TOTAL_DECODE_TIME` is under the configured
threshold on a representative workload; prefetch hit rate is published per model
and beats every fixed horizon tried; `--force-cold` produces honest
`storage-bound` reporting rather than a stall.

---

## Phase 5 — Speculation

| component | deliverable |
|---|---|
| C20 | MTP drafter (zero extra load I/O) |
| C20 | one DFlash / DFlash2 / DSpark loader with an arch compatibility check |
| C20 | DSpark confidence head wired into the prefetcher's early-stop |

**Exit gate:** bit-identical to non-speculative under fixed seed; MTP adds zero
load-time H2D; `experts touched / accepted token` is reported and adapts draft
width.

---

## Phase 6 — Autonomy and multi-session

| component | deliverable |
|---|---|
| C22 | budget manager across the 4 × 6 VRAM/RAM profile matrix |
| C23 | session scheduler, prefix-homogeneity batching, HTTP surface |
| C21 | telemetry dashboards; engine asserts the profiling acceptance criteria |

**Exit gate:** every profile row runs and reports the profile it chose and the
measurement that justified it; a shared-prefix multi-request workload beats FIFO
batching on TTFT; background compaction provably never delays a decode step.

---

## Phase 7 — Research (gated, optional, experimental)

### 7.1 Open, large

- No codebase exists yet. Until C1 is built, the entire Phase 1 deliverable is missing.
- The gaps in the specification tree that predate implementation are listed in
  §7.2 and consolidated in `docs/MISSING-ITEMS.md`.

The following are absent from every present document, or named only as a
placeholder. None of them is a correctness blocker; each is the kind of
omission a first sprint review should close before it becomes a structural
assumption.

### 7.2 Items missing from every present doc

1.  **Error model and failure taxonomy.** `02` names per-component failure
    modes in prose, and `06` names `STALL_TIME_MS` as a measured failure. What
    is missing is a single taxonomy the whole engine converges on: which failures
    are retriable, which drop a request, which degrade silently, which print a
    user-facing line, which are escape-hatch bugs, and which map to a
    telemetry counter. Absent today; worth writing before the first request can
    fail.
2.  **Security boundary for the HTTP surface and the cold store.** `02` C23
    mentions an OAI/Anthropic-compatible HTTP surface; `04` names NVMe
    admission and a cold store. What is missing is any statement of what a
    tenant may reach: the model files, the NVMe quota, the warm pool, other
    sessions’ prefixes, the profiler output, the tuning cache, the host fs.
    Absent today; the first user-facing server deserves it before day one.
3.  **Recovery story.** `04` and `09` describe NVMe write atomicity and a
    crash-safe journal. What is missing is the walk-back procedure: what
    `kanjoos` does on a dirty shutdown at the start of the next run, which
    journal entries it replays, which in-flight transfers it discards, which
    partially-written expert extents it quarantines, and what the user sees.
    Absent today; necessary before the cold store is load-bearing.
4.  **Multi-session correctness invariants.** `02` C23 names concurrent
    prefix-sharing and states that two sessions sharing a prefix must produce
    identical shared-span logits. What is missing is the invariant list that
    makes that testable: refcount invariants on shared KV blocks, what happens
    to a shared block when one session evicts it and another still needs it,
    what happens under cancel, and what constitutes a breach. Absent today.
5.  **Latency SLO and tail definition.** `06` profiles median device time and
    `04` prices Class B by ms/step. What is missing is any stated tail: what p99
    or p95 means here, how many samples define it, whether it is per-step or
    per-token, and what the SLO is for Class A, Class B and cold-start prefill.
    Absent today; a latency SLO is the thing that turns `STALL_TIME_MS` from a
    number into a decision.
6.  **Testing contract for the driver/launch abstraction.** `01` §7, `03`,
    `04` and `07` all assume a HIP/device layer behind C2. What is missing is
    any statement of what the abstraction exposes to the rest of the engine, what
    a mocked driver must satisfy for the CPU-only test tier, and what the
    CPU-reference path contract is. Absent today; the first integration test
    wants it.
7.  **Load, save, checkpoint and reload of a *live session*.** `04` names
    persistence of shared prefixes and suspended sessions; `09` describes Class C
    suspend. What is missing is the exact session checkpoint format, what is
    captured (KV pages, warm-resident experts, router state if any, suspended
    draft state), what is not, the reload path, and the failure mode if reload
    finds a changed model fingerprint. Absent today; needed for ‘suspend and
    resume’ to be real.
8.  **Fused kernel contract.** `03` names fusion targets and `06` names
    `ffn-activate`, `residual` and `attention-mix` as profiler classes. What is
    missing is the naming and versioning contract for fused kernels: which
    fusions are guaranteed, which are opportunistic, what a regression in a
    fusion looks like in the profiler, and what the fallback is when a fusion is
    disabled. Absent today; wanted before tuning claims about overlap can be
    trusted.
9.  **Tokenizer and detokenization boundary.** The architecture document
    assumes a tokenizer exists and `07` assumes a Python/HF-style vocab is
    available, but no doc says where tokenization lives (client, server, a
    bundled embedding), what the byte/unicode/tooling contract is, what happens
    on an unknown token, and how the embedding table is loaded and versioned
    alongside the model. Absent today; the HTTP surface is not real without it.
10. **Mixed-precision KV ladder, when it is ever built.** `09` §6 sketches a
    32-bit / 8-bit / 4-bit ladder over context distance and mentions RoPE
    bucketing. What is missing is any decision document: what triggers
    promotion/demotion between rungs, whether the ladder is per-session or
    global, how the warm-tier resident context is priced under the ladder, and
    whether the ladder is a Class A feature or Class B only. Absent today;
    `09` §6 is a sketch, not a commitment.

### 7.3 Items missing from individual docs

- **`02-components.md`:** missing per-component acceptance test named in `08`;
    missing failure taxonomy (item 1 above); missing the CPU-reference contract
    for the device layer (item 6); missing error codes / user-visible strings.
- **`03-kernels.md`:** validation matrix present and good; missing a
    cross-arch kernel source layout statement (one file per kernel, arch selected
    by `#if` behind C2) beyond the rule in §3; missing the namespacing/versioning
    of kernels behind C16/C17 (item 8); missing a statement of what is left
    SIMT-by-necessity on gfx1201 (the fallback path) and what is left SIMT on
    gfx1031 (the only path).
- **`04-memory-tiering.md`:** §6 extended by `09` §5/§6; missing recovery walk-back
    (item 3), missing the session checkpoint format (item 7), missing the pinned
    pool under memory pressure behavior statement beyond the shrink order.
- **`05-speculation.md`:** acceptance criteria present; missing the exact
    verification batch size / draft width interaction with the expert-union cost
    beyond the profiler assertion; missing how speculative state is checkpointed
    under item 7.
- **`06-profiling.md`:** output contract and floor present and excellent; missing
    the tail/SLO definition (item 5), missing the JSON/CSV schema version for
    downstream parsing, missing how the table is aggregated across a batch of
    requests (per-request or merged).
- **`07-build-platforms.md`:** build layout and toolchain facts present; missing
    a versioning/tuning-cache invalidation statement beyond C3’s existing rule.
- **`08-roadmap.md`:** phases and gates present; missing the error taxonomy (item 1),
    the security model (item 2), the recovery story (item 3), the multi-session
    invariants (item 4), the SLO (item 5), the device-layer test contract (item 6),
    the session checkpoint format (item 7), the fusion contract (item 8), the
    tokenizer boundary (item 9), and the KV ladder decision (item 10).
- **`09-kv-engine-architecture.md`:** the most complete doc and the right place for
    the KV/weight reframe; missing the KV ladder decision (item 10), missing the
    session checkpoint format (item 7) at the architecture level, missing the
    recovery walk-back (item 3) that makes the cold store load-bearing.

### 7.4 Versioning, compatibility and migration

These are absent as documents and only lightly touched as scattered rules:

- **Model/format versioning.** `04` names a manifest and hashes; `06` names
    cache keys that include model fingerprint. What is missing is the
    compatibility story: which changes to a model config or a packed expert store
    are backward-compatible, which force a reload, which force a full recompile
    (C5), and what the user-visible error is when a cache is stale.
- **API versioning.** The HTTP surface is described by shape (OAI-compatible) but
    no doc describes versioning of that surface, what a client may rely on
    staying stable, or the deprecation story.
- **Profiler / telemetry output versioning.** `06` defines a contract but not its
    stability or schema version across releases.
- **Configuration schema versioning.** `kanjoos.toml` is described as a starting
    point; no doc states the schema version, which keys are stable, which may be
    added, or what happens on an unknown key.
- **Documentation versioning.** The docs tree has no stated version, no changelog,
    and no stated relationship between a doc, the commit it was written against,
    and the tool version that validates it (`check_docs.py`, `kv_roofline.py
    --check`).

### 7.5 What is NOT in the code and should not be invented to fill a doc gap

To keep the missing-items list honest, these are **not** missing: they are
intentional non-goals or out-of-scope unless a future phase says otherwise.

- Training / fine-tuning — non-goal (`01` §8).
- Multi-GPU — non-goal (`01` §8), SPLASH-style decoupling recorded for later.
- Quantising the attention trunk or router by default — non-goal (`01` §8).
- Expert-only BF16→ternary in the production path — experimental behind a quality
    gate, not a default (`01` §8).
- Making NVMe look like VRAM — non-goal (`01` §8).
- A general tiering engine — `09` §10.2 says build Class A first, add tiering
    only when a real session exceeds it.
- WMMA tiling — demoted; the measured kernel is within 16% of an intensity ceiling
    tiling cannot move (`00` §8.9).
- CPU expert fallback as a performance path — correctness guarantee first,
    performance path later (`09` §10.2).

### 7.6 What the missing-items list does and does not imply

It does **not** mean the engineering plan is incomplete in a blocking way. The
substantive constraints — transfer-bound physics, measured ISA, measured routing
locality, measured bandwidths, the W3 operating point, the context-class ladder,
the I7 contract, the profiler contract, the kernel plan, the build plan, the test
plan, the risk register — are all present and most are measured.

It **does** mean that a first implementation sprint will hit these the moment it
tries to make the engine user-facing, recoverable, testable, and observable in
production. The ordering above is the recommended order to close them: the items
that touch correctness and observability first (1, 5, 6, 4), then the items that
touch the server and cold store (2, 3, 7), then the items that touch
long-term maintenance and polish (8, 9, 10, versioning).

---

## Phase 7 — Research (gated, optional, experimental)

Only after the above is stable and measured.

| item | gate |
|---|---|
| mixed W8/W6/W4 per-expert precision allocation | sensitivity model + routing-stability metrics + global allocation under a real memory budget |
| expert-only ternary (CAT-Q-style) | full calibration pipeline, per-expert quality metrics, routing metrics, BF16-source quality envelope established by our own measurements |
| BITCOS vs five-trit storage | measured size **and** kernel cost per tensor; automatic selection. Do not assume BITCOS is smaller — it is ~3% larger for CAT-Q Qwen3-30B-A3B once scales are counted. |
| out-of-core / streaming attention | only after the KV path is stable; must report storage-bound honestly |
| flash-decode tuning to FA4's structure | two-tile ping-pong + selective rescaling, measured against the straightforward implementation |
| TurboQuant 3-bit variants | **rejected by the vLLM study on accuracy.** Revisit only with our own evidence. |

---

## Acceptance criteria (whole engine)

**Functional**

```
[ ] complete-expert execution for every routed expert, every profile
[ ] no expert loss, no silent route changes under cache pressure
[ ] no FP16 materialisation on the expert miss path
[ ] KV survives VRAM eviction and RAM eviction bit-identically
[ ] KV survives process restart via the NVMe tier
[ ] a cache page from a different model/config/codec is rejected, not misread
[ ] concurrent sessions share prefixes and produce identical shared-span logits
[ ] cancellation releases in-flight pages without use-after-free (ASAN/UBSAN run)
[ ] a crash during an NVMe write leaves committed objects intact
[ ] speculation is bit-identical to non-speculative under a fixed seed
```

**Performance**

```
[ ] no synchronous NVMe read in the attention or expert path
[ ] H2D/D2H overlap compute — visible as idle < wall in the profile table
[ ] transfer batches coalesce adjacent pages; extents in the tuned range
[ ] prefetch hit rate published per model, better than every fixed horizon
[ ] WARM tier measurably lowers KV recompute rate versus no-WARM
[ ] prefix cache measurably lowers TTFT on repeated prefixes
[ ] every run reports its residency label honestly, including storage-bound
```

**Regression**

```
[ ] forward matches the CPU reference within a stated per-format tolerance
[ ] deterministic sampling reproduces under a fixed seed, with and without speculation
[ ] no attention drift from cache round-trips
[ ] no stale KV after a model or config change
[ ] an ROCm upgrade that moves the ISA matrix fails CI rather than silently
    losing performance
```

---

## Risk register

| risk | impact | likelihood | mitigation |
|---|---|---|---|
| **Expert reuse across tokens is low** | MoE streaming is I/O-bound; tok/s ceiling is set by disk bandwidth, not compute | medium | measured in P0-7 before committing; RAM residency, CPU fallback, speculation, and honest `storage-bound` reporting are the answers |
| **WMMA numerics diverge from SIMT** | the gfx1201 matrix path is silently wrong | high | WMMA is built first *and* the SIMT path stays a compiled-in fallback; P1-3 A/Bs them against the Tensile oracle |
| **Expert reuse across tokens is low enough to invalidate MoE** | the tiering design itself is wrong, not just its tuning | medium | P0-7 runs before Phase 1 sign-off; if reuse is poor the design falls back to RAM-resident experts + CPU spill rather than deeper NVMe tiering |
| **NVMe wear from KV persistence** | early SSD death on a consumer machine | low | admission control in C12, TTL, shared-prefix-only default policy, bytes-per-token asserted in CI |
| **Windows/Linux divergence** | the same engine behaves differently; the user's actual platform is worse than the tested one | high | platform differences confined to C24; same test suite on both; `doctor` prints resolved facts |
| **ReBAR off in BIOS** | VRAM appears capped at 256 MiB; every profile is fiction | medium | refuse to start with a one-line explanation rather than thrashing |
| **Pinned memory exhaustion** | transfer collapse mid-run | low | bounded pool (C7), `VirtualLock`/`mlock` RLIMIT checked at init |
| **Hybrid (recurrent) models break prefix reuse** | the 44-file model collection cannot use the radix cache | medium | recurrent layers flagged in the index; bounded suffix replay instead of refusal |

---

## What "better than Strata and Edge0" means, measurably

Both are credible projects and this plan takes ideas from both: Edge0's
activation-predictive expert prefetching, Strata's "load into RAM, read from
storage when RAM is not enough" shape, and the prior work already recorded in the
sibling kanjoos project's `REFERENCES.md`.

The claims that would justify the word "better" are all measurable, and all are
acceptance criteria above:

1. **Both targets are first-class**, with the ISA actually probed rather than
   assumed — including an RDNA2 path that works without matrix units.
2. **The residency is predictive and honest.** Not "an SSD-backed VRAM cache";
   an engine that says which tier it is bound by and adapts its prefetch horizon
   to the measured bandwidth.
3. **The profiler is a first-class component** with an output contract and a
   measured instrumentation floor, so claims about where time goes are claims
   that survive inspection.
4. **Speculation is lossless and priced correctly** — MTP first because it is
   free, the drafter families behind one loader, and the confidence head wired
   into the transfer scheduler so drafting stops when verification would stall.
5. **Every quality decision is gated on measurement**, including the ones the
   literature loves: TurboQuant 3-bit is rejected because it was measured, and
   ternary experts stay experimental until our own calibration exists.