# Worksheet — C8 GPU hot expert slot allocator

**What it owns:** VRAM slots for experts.

**Why it exists (from `02-components.md` — C8 section):** the MoE weight space is one VA reservation sized to what fits, sub-allocated into slots — one allocation,
no per-expert `hipMalloc`. A slot is never overwritten while a submitted kernel may still read it; lifetime is bounded by a HIP event, not by wall-clock reasoning.
Slot size is fixed by the expert geometry for a given model. Slot count per profile: 6 GiB → small fixed pool; 8 GiB → moderate; 12/16 GiB → as large as the remaining
budget allows after KV. C22 splits it.

**Spec lines that must hold:**
* `02-components.md` — C8 section (verbatim: ExpertSlot, SlotPool API, the rules, the done-when).
* `01-architecture.md` §3.1 (VRAM ceiling, the allocation split: VRAM_CEILING − MODEL_DEVICE − WORKSPACE − SAFETY = HOT_KV + EXPERT_SLOTS. C8's slot count comes from the
  EXPERT_SLOTS side of this split.)
* `04-memory-tiering.md` §2 (VRAM profiles: ceiling, KV block, layer slab, lookahead tokens, H2D in flight, prefetch extent, expert slots — the expert-slots column is what C8 sizes
  per profile; 16 GiB → "largest that fits"; the KV/expert split is the most consequential policy in the engine).

**Interface (verbatim `02-components.md` C8):**
```cpp
struct ExpertSlot {
    uint32_t layer, expert;
    uint64_t vram_offset, byte_size;
    uint64_t last_use, generation;
    uint32_t use_count, state;   // FREE LOADING RESIDENT IN_USE EVICT_PENDING
};

class SlotPool {
public:
    SlotHandle acquire(uint32_t layer, uint32_t expert, size_t bytes);
    void       pin(SlotHandle);         // released only by stream event
    void       release(SlotHandle);
    void       retire(SlotHandle, hipEvent_t last_use_event);
};
```

**Rules (verbatim):**
* **A slot is never overwritten while a submitted kernel may still read it.** Lifetime is bounded by a HIP event, not by wall-clock reasoning.
* Slot size is fixed by the expert geometry for a given model, so slots are a slab: one allocation, sub-allocated, no per-expert `hipMalloc`. The whole MoE weight space is one VA
  reservation sized to what fits, which is what makes 6 GiB and 8 GiB profiles work at all.
* Slot count per profile: 6 GiB → small fixed pool; 8 GiB → moderate; 12/16 GiB → as large as the remaining budget allows after KV. C22 splits it.

**Acceptance test (verbatim "done when"):**
* No use-after-free under an eviction stress test with the sanitiser build.
* Slot occupancy ≥ 80% under a routing-heavy workload.

**Gates:**
* Depends on C9 (residency — C8's slots are what C9 places experts into; C9's `ready()` is what kernels ask, not C8's state directly), C11 (transfer — a slot being loaded is a
  transfer in flight; C8's LOADING state is what C9 sees while C11 brings the bytes in), C17 (GEMM — C17 reads from C8's slots; a slot must not be overwritten while C17's kernel may still
  read it — the HIP-event-bounded lifetime is what makes this safe), C22 (budget — the slot count per profile comes from C22's split).
* The "no use-after-free under eviction stress with the sanitiser build" gate means the HIP-event-bounded lifetime must be real and tested under stress. A coder must not free a slot on a
  wall-clock timeout or a "I think the kernel is done" heuristic — the release/retire must be driven by a HIP event (the kernel's completion event), and the sanitiser build must be able to
  catch a use-after-free. The eviction stress test is: evict and reload slots under a routing-heavy workload, with overlapping launches, and assert no use-after-free. This is the same
  page-local-vs-global discipline as C17/I7 — a slot indexed with the wrong offset is an out-of-bounds read. The sanitiser build catches it.
* The "slot occupancy ≥ 80% under a routing-heavy workload" gate means the slot pool must be sized to the workload, not over-allocated to "be safe". A coder must be able to measure slot
  occupancy (used / total) under a routing-heavy workload and report it. Under 80% means the pool is oversized (wasting VRAM that could be KV); over 100% means slots are being evicted
  too often (cold misses). The 80% target is the observance point, not a hard bound — but it is the measured signal that tells you the pool is right-sized.

**Coder notes / pitfalls:**
* The HIP-event-bounded lifetime is the core safety mechanism. A slot is `pin()`-ed (released only by a stream event), and `retire()` takes the last-use event. A coder must not free a slot until
  that event has fired on the stream the consuming kernel was launched on. This is the mechanism that makes "a slot is never overwritten while a submitted kernel may still read it" enforceable,
  not a comment. The sanitiser build + eviction stress test is the proof.
* One VA reservation, sub-allocated, no per-expert `hipMalloc`. A coder must not call `hipMalloc`/`hipFree` per expert. The whole MoE weight space (for the experts that fit) is one allocation,
  sub-allocated into slots of fixed size (the expert geometry for the model/pack). This is what makes 6/8 GiB profiles work (the VRAM is one reservation, sub-allocated, rather than a set of
  per-expert allocations that would fragment it).
* Slot size is fixed by the expert geometry for a given model/pack. For W4 g128 on Qwen3-30B-A3B: one expert in one layer = 2.4 MiB (`kv_roofline.py` section G.1: "one expert in one layer =
  2.4 MiB at W4"). So a slot is 2.4 MiB, and the slot pool's total VRAM is slot_count × 2.4 MiB. A coder must compute slot size from the model/pack geometry (via `kvroof`), not hardcode 2.4 MiB —
  the same component on a different model or pack has a different slot size.
* The slot count per profile is set by C22, not by C8. C8's job is to manage the slots (acquire, pin, release, retire, evict) within the count C22 gives it. A coder must not let C8 decide its own
  size — that is C22's job (the KV/expert split is the most consequential policy in the engine, and it is set by C22 from the profile + measurement).
* The slot state machine (FREE / LOADING / RESIDENT / IN_USE / EVICT_PENDING) is what C9 observes. A coder must keep it accurate — a slot that is LOADING must not be used by C17 until it is
  RESIDENT/IN_USE and the load event has fired; a slot that is EVICT_PENDING must not be reused until the retire event has fired. The state machine is the contract between C8 and C9/C17.

**Worked micro-example (host, sanity-check before device):**
Implement a fake SlotPool backed by a fixed-size VA reservation (one allocation, sub-allocated into fixed-size slots), with the state machine (FREE/LOADING/RESIDENT/IN_USE/EVICT_PENDING) and
  the HIP-event-bounded lifetime (pin/release/retire with a fake event that fires at a stated time). Then: (1) acquire slots for a routing-heavy workload (many experts across many layers) and assert
  slot occupancy can reach ≥ 80% without overrun — the occupancy gate; (2) acquire a slot, launch a fake kernel that reads it, pin the slot, retire it with a fake last-use event, and assert the slot
  is not freed/reused until the event fires — the HIP-event-bounded-lifetime gate; (3) under an eviction stress test (acquire, use, evict, reload, overlapping launches), run the sanitiser build and
  assert no use-after-free — the use-after-free gate; (4) assert a slot in LOADING state is not usable by the consuming kernel until it transitions to RESIDENT/IN_USE and the load event fires — the
  state-machine contract; (5) compute slot size from the model/pack geometry via `kvroof` (W4 g128 on Qwen3-30B-A3B → 2.4 MiB) and assert the pool's total VRAM = slot_count × slot_size — the
  geometry-derived slot size gate.

**Files a coder should read before starting:**
* `docs/02-components.md` — C8 section.
* `docs/01-architecture.md` §3.1 (VRAM allocation split — C8's slot count comes from the EXPERT_SLOTS side).
* `docs/04-memory-tiering.md` §2 (VRAM profiles — the expert-slots column per profile; the KV/expert split policy).
* `docs/00-verified-facts.md` §8.10 (one expert in one layer = 2.47 MB at W4 for Qwen3-30B-A3B — the slot size; `kv_roofline.py` section G.1 prints "one expert in one layer = 2.4 MiB at W4").
* `ai-coder/c9-residency.md` (C8's slots are what C9 places experts into; C9's `ready()` is what kernels ask).
* `ai-coder/c17-expert-gemm.md` (C17 reads from C8's slots — the HIP-event-bounded lifetime is what makes this safe; the page-local-vs-global discipline).
* `ai-coder/c11-transfer.md` (a slot being loaded is a transfer in flight — C11 brings the bytes in; C8's LOADING state is what C9 sees).
* `ai-coder/c22-budget.md` (C22 sets the slot count per profile — C8 manages within it).
* `tools/kvroof/kv_roofline.py` section G.1 (one expert in one layer = 2.4 MiB at W4 — the slot size the coder must derive, not hardcode).
