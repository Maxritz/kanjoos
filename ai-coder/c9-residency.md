# Worksheet — C9 Residency manager

**What it owns:** where every expert and KV page currently is.

**Why it exists (from `02-components.md` — C9 section):** the residency table is the single source of truth for where every expert and KV page is (ABSENT, NVME_RESIDENT,
  LOADING_NVME, RAM_RESIDENT, LOADING_RAM, VRAM_RESIDENT, PREFETCHED, EVICTING_VRAM, EVICTING_RAM, PINNED, INVALID). Transitions are asynchronous and observable (every transition
  emits a C21 telemetry event). A kernel never asks where something is — it asks `ready()`. If false, the scheduler has already failed over to the CPU path (C19) rather than stalling.
  A demand request that cannot be satisfied from RAM immediately escalates to NVMe and returns "not ready" — never a blocking read.

**Spec lines that must hold:**
* `02-components.md` — C9 section (verbatim: states, ResidencyManager API, the rules, the done-when).
* `01-architecture.md` §6 (control flow: C9 `ready()` is the only thing kernels ask; if false, the scheduler has already failed over to the CPU path, not stalled; a demand that cannot be
  satisfied from RAM immediately escalates to NVMe and returns "not ready" — never a blocking read).
* `09-kv-engine-architecture.md` §5 (I1: no expert is ever lost — every routed expert is recoverable from RAM or NVMe; a cache miss changes latency, never the result. I7: tier movement is
  invisible to the result — evicting and reloading a KV page or expert reproduces the same numbers, bit for bit. These are the invariants C9 defends.)
* `04-memory-tiering.md` §5.1 (the discipline: predict → plan → prefetch → RAM → VRAM → attention; the attention path never issues a storage read; if a required page is not in VRAM when its
  layer arrives, the scheduler has already either found it in RAM and issued H2D early enough, found it in NVMe and issued the read early enough, or failed to and taken the CPU/expert fallback (C19)
  rather than stalling; a hard stall is a last resort and a telemetry event (STALL_TIME_MS), never the mechanism.)

**Interface (verbatim `02-components.md` C9):**
```cpp
class ResidencyManager {
public:
    Residency locate(ExpertId) const;
    void      request(ExpertId, KVTier);      // demand
    void      prefetch(ExpertId, KVTier, PrefetchPriority);
    void      promote(ExpertId);  void demote(ExpertId);
    void      tick();                          // called once per layer
    bool      ready(ExpertId) const;          // the only thing kernels ask
};
```

**Rules (verbatim):**
* Transitions are asynchronous and observable. Every transition emits a C21 telemetry event.
* **A kernel never asks where something is.** It asks `ready()`, and if false the scheduler has already failed over to the CPU path (C19) rather than stalling.
* A demand request that cannot be satisfied from RAM immediately escalates to NVMe and returns "not ready" — never a blocking read.

**Acceptance test (verbatim "done when"):**
* A full-execution trace over a real model shows zero synchronous NVMe reads in the attention path.
* `KV_MISS_STALL_TIME / TOTAL_DECODE_TIME` is under the configured threshold.

**Gates:**
* Depends on C8 (slots — C9 places experts into C8's slots; C9's `ready()` for an expert is "the slot is resident and the load event has fired"), C11 (transfer — C9 enqueues NVMe reads and H2D
  transfers via C11; C9's "not ready" return is what makes C11's async transfers non-blocking), C19 (fallback — if C9's `ready()` is false for an expert when its layer arrives, the scheduler has
  already failed over to C19, not stalled), C21 (telemetry — every transition emits a C21 event; STALL_TIME_MS is a C21 counter; the acceptance test's ratio is a C21 number), C10 (prefetch — C9's
  `prefetch()` is what C10 calls to get things into VRAM before their layer arrives), C14 (KV — C9's `locate()`/`request()`/`prefetch()` for KV pages; the KV side of C9 is what C14's paged KV
  consults), C12 (admission/eviction — C9's `demote()` is what C12 calls to evict; C9's residency table is what C12's admission policy reads).
* The "zero synchronous NVMe reads in the attention path" gate is the hard invariant (I4: no synchronous NVMe read on the critical path). A coder must be able to trace a full execution and assert that
  no attention path op waited on an NVMe read. This is enforced by a debug assert **plus** the profiler (C21) — the profiler is the persistent, non-debug proof. A coder must not implement a "just block on
  NVMe" path as a shortcut.
* The `ready()`-is-the-only-thing-kernels-ask rule is the structural guarantee that makes the above possible. A coder must not let a kernel ask C9 "where is this" or "when will this arrive" — it asks
  `ready()`, and if false the scheduler (not the kernel) has already failed over. This is the thing that prevents a kernel from blocking on a slow tier.
* The "demand that cannot be satisfied from RAM immediately escalates to NVMe and returns not ready — never a blocking read" rule is the escape hatch for the case where even RAM is not fast enough.
  A coder must not implement a "block until RAM has it" path — if RAM does not have it in time, the demand escalates to NVMe (still async, still returns not ready) or the scheduler has already
  fallen back to C19. The "never a blocking read" is the rule.

**Coder notes / pitfalls:**
* `ready()` is the only thing kernels ask. A coder must not let a kernel call `locate()`, `request()`, or any other C9 query as part of the step loop's hot path. The kernel asks `ready()`, and the
  scheduler (C9/C10/C11/C19) has already done the work of making it ready or failing over. This is the structural rule that makes "no synchronous NVMe read on the critical path" enforceable.
* The state machine is large (11 states) and transitions are asynchronous. A coder must keep it accurate and observable (every transition emits a C21 event). The observability is not decorative — it is how
  C21 reconstructs the profile (transfer time, idle time, stall time) and how the acceptance test proves "zero synchronous NVMe reads in the attention path" and the `KV_MISS_STALL_TIME / TOTAL_DECODE_TIME`
  ratio.
* A demand that cannot be satisfied from RAM immediately escalates to NVMe and returns "not ready" — never a blocking read. This is the rule that makes NVMe a capacity tier, not a critical-path tier. A coder must
  not implement "block until NVMe has it" — if NVMe does not have it in time, the scheduler has already fallen back to C19 (the CPU/expert fallback) for that expert, not stalled the step. The fallback is
  the escape hatch, not a stall.
* The tick() is called once per layer. A coder must implement the per-layer advance — prefetch horizons, eviction scoring, promotion/demotion — as a per-layer tick, not as a continuous background thread that
  races the step loop. The tick is the synchronization point.
* C9's `ready()` for an expert is "the slot is resident and the load event has fired" (C8's HIP-event-bounded lifetime). A coder must not implement `ready()` as "the transfer was issued" — issued is not ready;
  completed is ready. The HIP event is the proof of completion.
* C9's KV side: the KV residency is separate from the expert residency (KV has its own index, C14, and its own tiers, but shares C9's machinery). A coder must keep expert residency and KV residency as
  separate sub-tables within C9, with the same state-machine discipline, because KV pages and expert objects have different sizes, different eviction priorities, and different lifetimes.

**Worked micro-example (host, sanity-check before device):**
Implement a fake C9 with the 11-state machine, a fake C8 (slots with HIP-event-bounded lifetime), a fake C11 (async transfers that complete at a stated time), a fake C19 (fallback), and a fake step loop.
  Then: (1) for a demand expert that is not resident, assert C9 enqueues an async transfer (via C11) and returns — never blocks — and that if RAM does not have it in time, the demand escalates to NVMe and
  still returns "not ready" (the "never a blocking read" gate); (2) assert the kernel asks only `ready()`, and if false the scheduler has already failed over to C19 for that expert — never stalls (the
  "`ready()` is the only thing kernels ask" gate); (3) run a full-execution trace and assert zero synchronous NVMe reads in the attention path — the I4 gate; (4) assert every state transition emits a C21
  telemetry event (the observability gate); (5) assert `tick()` advances the per-layer state (prefetch horizons, eviction scoring, promotion/demotion) and that a prefetch issued early arrives in VRAM before
  its layer (the "prefetch → RAM → VRAM → attention" discipline); (6) assert that if a KV page or expert is evicted and reloaded, the result is the same (bit for bit, for KV — I7) or the same expert
  contribution (for experts — I1) — the tier-movement-is-invisible gate.

**Files a coder should read before starting:**
* `docs/02-components.md` — C9 section.
* `docs/01-architecture.md` §6 (control flow — `ready()` is the only thing kernels ask; the discipline; the fallback, not a stall).
* `docs/09-kv-engine-architecture.md` §5 (I1 — no expert ever lost; I7 — tier movement invisible to the result).
* `docs/04-memory-tiering.md` §5.1 (the discipline; hard stall is a last resort and a STALL_TIME_MS telemetry event, never the mechanism).
* `docs/08-roadmap.md` — Acceptance criteria (no synchronous NVMe read in the attention or expert path; every run reports its residency label honestly, including storage-bound).
* `ai-coder/c8-slot-allocator.md` (C8's slots — C9's `ready()` is "slot resident and load event fired"; the HIP-event-bounded lifetime).
* `ai-coder/c11-transfer.md` (C11 — C9 enqueues async transfers; "not ready" is what makes them non-blocking).
* `ai-coder/c10-prefetch.md` (C10 — C9's `prefetch()` is what C10 calls; the prefetch discipline).
* `ai-coder/c19-cpu-fallback.md` (C19 — the fallback when C9's `ready()` is false; not a stall).
* `ai-coder/c14-kv-cache.md` (C14 — the KV side of C9; separate sub-table with the same discipline).
* `ai-coder/c12-admission.md` (C12 — C9's `demote()` is what C12 calls; C9's residency table is what C12's admission reads).
* `ai-coder/c21-profiler.md` (C21 — every transition emits a C21 event; STALL_TIME_MS; the acceptance-test ratio is a C21 number).
