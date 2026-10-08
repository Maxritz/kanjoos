# 02 — Component specification

Twenty-four components. Each entry states what it owns, the interface other
components may depend on, its state machine, the invariants it defends, how it
fails, and what "done" means as a test.

Component IDs are stable; `08-roadmap.md` refers to them by phase.

---

## C1 — Runtime core and stream scheduler

**Owns:** the op graph, stream set, event ring, submission cadence.

A decode step is not a function call, it is a queue of ~500 tiny ops. Whether
they overlap is worth more than most kernel micro-optimisation, so this component
exists to make overlap expressible and measurable.

```cpp
enum class OpClass { Embed, Norm, Attention, AttnMix, Proj, FFNAct,
                     Residual, Recurrent, MoERouter, MoEGEMM, KVWrite,
                     KVRead, KVQuant, Spec, Transfer, Head, Bias, Sample };

struct OpRecord {
    OpClass     cls;
    uint64_t    dev_start_ns, dev_end_ns;   // from the event ring
    uint64_t    host_begin_ns, host_end_ns; // CPU-side, around the launch
    uint32_t    stream;
};

class Runtime {
public:
    StreamId compute_stream();          // where math goes
    StreamId transfer_stream();         // where copies go, may share compute
    StreamId aux_stream();              // norms/quant/rope, overlappable

    // Non-blocking. Returns immediately; the op is timed by C21.
    void submit(Op op);

    // Walks the event ring, updates C21 accumulators, retires transfer ops,
    // unblocks anything waiting on residency (C9).
    void poll();

    void submit_all();                  // one fence point, then poll
};
```

**Rule:** every op is submitted to a stream and timestamped. Nothing calls
`hipDeviceSynchronize()` inside the step loop; a full sync exists only at
`submit_all()` for the sampler, and C21 counts how often it is hit.

**Invariants:** op order on one stream is program order; transfer ops never
share a stream with the compute that reads their destination without an event
dependency.

**Fails by:** falling back to a single stream when queue-family topology does
not permit more (measured at startup, P0-1), which costs overlap but not
correctness.

**Done when:** `--profiling` shows a non-zero overlap factor (device busy time
< wall time on a MoE decode step) on gfx1201, and the same test on gfx1031
documents the achieved overlap rather than assuming zero.

---

## C2 — Device abstraction layer

**Owns:** the arch split. Everything above C2 is arch-agnostic.

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

class Backend {                      // virtual, one impl per arch
public:
    static std::unique_ptr<Backend> create();   // probes, never guesses
    const DeviceCaps& caps() const;
    virtual void launch_attention(const AttentionPlan&) = 0;
    virtual void launch_expert_gemm(const ExpertPlan&) = 0;
    virtual void launch_kv_write(...) = 0;
};
```

**Rule:** caps come from `tools/isa_probe`-equivalent runtime probing compiled
into the binary, cross-checked against `hipDeviceProp_t`. A kernel that needs a
cap the device does not advertise is never instantiated.

**Done when:** the same binary, on either box, selects the right backend and
prints the capability table that C21 shows in the profile header.

---

## C3 — ISA probe and kernel autotuner

**Owns:** turning "the compiler accepts this" into a runtime decision.

`tools/isa_probe/` (already in this repo) is compile-time; C3 is the runtime
half: on first launch for an arch it runs a short benchmark sweep over the
compiled kernel variants and writes the winner to a per-arch cache
(`kanjoos-tune/<arch>.json`).

```cpp
struct TuningKey { std::string arch, kernel, quant; int m, n, k, tile; };
struct TuningResult { TuningKey key; double us; int variant; };
class Autotuner {
public:
    void calibrate(const TuningSpace&);     // bounded, once per arch
    const TuningResult& pick(const TuningKey&) const;
};
```

**Rule:** autotuning is bounded and cached. It never runs mid-request. A missing
entry falls back to the analytic default, not to a guess-and-hope.

**Done when:** a tuned run reproduces within 5% of its own tuning measurement,
and the cache file invalidates correctly when the binary hash or arch changes.

---

## C4 — Model loader

**Owns:** turning a GGUF on disk into resident buffers and a cold index.

* mmap the file, honour `general.alignment` (32/64/128 — assuming 32 silently
  misreads every offset; this is recorded in the project's RESEARCH.md).
* Sequential-read hints per platform: Linux `posix_fadvise(WILLNEED)` +
  `madvise(MADV_WILLNEED)`; Windows `FILE_FLAG_SEQUENTIAL_SCAN` /
  `SetFileInformationByHandle(FileDispositionInfo)`. Plus `madvise(MADV_HUGEPAGE)`
  for the RAM tier — expert pages are read once, so THP is nearly free.
* Build a **cold index** at load: tensor directory, expert offsets grouped by
  layer into contiguous extents (C6), and the KV geometry.
* Never read expert weights at load. Load reads the trunk and the *directory*.

**Done when:** load time for a 30B-class MoE is bounded by the trunk size, not
the model size, and a cold-start on a 4 GB/s disk stays under a stated number
for a 40 GiB model.

---

## C5 — Expert compiler (offline)

**Owns:** canonical checkpoint → kernel-native packed expert store.

Pipeline, and the order matters:

```
BF16 checkpoint
  -> expert extraction (per layer, per expert, gate/up/down)
  -> activation-aware calibration (MoEQuant-style expert-balanced sampling:
     low-frequency experts MUST get their own calibration examples)
  -> candidate evaluation per expert: W8 / W6 / W4 / ternary
  -> precision allocation under the RAM+VRAM budget of the target profile
  -> packing into the kernel-native format for BOTH arches
  -> manifest + hashes
```

Output manifest records, per expert: source hash, packed hash, precision, format,
group size, NVMe offset, byte size, quality loss, routing loss. Ternary is
selected by **measured** size and kernel cost against five-trit packing, not by
assumption — the earlier spec's correction stands (BITCOS is ~3% *larger* than
five-trit for CAT-Q Qwen3-30B-A3B once scales are counted).

**Invariants:** deterministic — same source hash gives byte-identical output.
Reproducible from source hash alone.

**Done when:** re-running the compiler on the same checkpoint produces an
identical manifest hash, and a corrupt source hash is detected rather than
compiled.

---

## C6 — Expert directory and NVMe object store

**Owns:** the addressable unit that the residency manager moves.

```cpp
struct ExpertObject {
    uint32_t layer, expert;
    uint8_t  precision, format;
    uint16_t group_size;
    uint64_t nvme_offset, nvme_size;
    uint64_t packed_hash;
    float    quality_loss, route_loss;
};
```

* **Logical unit = one expert.** **Physical I/O unit = a coalesced multi-expert
  extent** (a whole layer slab where the packing is contiguous). Residency
  decides per expert; I/O moves 2–32 MiB extents. Confirmed by measurement as
  the right split — a 4 KiB random read of expert payload is the failure mode
  this avoids.
* Content-addressed, immutable, `KVP`-style container: header, model
  fingerprint, quant descriptor, token/layer metadata, checksum, payload index,
  payload. Aligned offsets, versioned format, atomic publish, crash-safe journal
  + manifest.
* Verify checksum on read; on mismatch, re-read from the canonical checkpoint
  and quarantine the object.

**Done when:** a crash during a write leaves committed objects intact and
readable; a corrupt object is detected by checksum and replaced, with a telemetry
counter that fires.

---

## C7 — Host tier: warm pool, pinned DMA staging

**Owns:** RAM. This is the component the performance argument rests on, and it is
also the one that can take the machine down.

```cpp
class WarmPool {                 // pageable, huge-page-hinted, 2 MiB aligned
    void*  allocate(size_t bytes, PoolTag tag);
    void   release(void* p);
    size_t used() const; size_t available() const;
    void   shrink_to(size_t bytes);         // under memory pressure
};
class PinnedPool {               // bounded, page-locked, DMA-only
    Chunk  acquire(size_t bytes);           // 2–32 MiB chunks
    void   release(Chunk);
    size_t capacity_bytes() const;
};
```

Rules:

* **The whole machine is never pinned.** Pinned pool caps are 2 GiB (24/32 GiB
  hosts), 3 GiB (48), 6 GiB (96). Everything else is pageable and copied through
  a pinned bounce buffer in large aligned chunks.
* Chunk size is tuned per host (P0-1 measures the host copy path; the workspace
  note already records ~19 GB/s cached file reads).
* Windows: `VirtualLock` on a pre-committed reserve. Linux: `mlock` +
  `MAP_POPULATE`, with an RLIMIT check up front so a missing capability fails
  loudly at init rather than mid-transfer.

**Fails by:** shrinking under pressure in the order given in §3.2 of
`01-architecture.md`.

**Done when:** RSS stays inside the declared budget under a 96 GiB-profile
synthetic load, and a transfer loop pinned-bursts without ever exceeding the
pinned cap.

---

## C8 — GPU hot expert slot allocator

**Owns:** VRAM slots for experts.

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

Rules:

* **A slot is never overwritten while a submitted kernel may still read it.**
  Lifetime is bounded by a HIP event, not by wall-clock reasoning.
* Slot size is fixed by the expert geometry for a given model, so slots are a
  slab: one allocation, sub-allocated, no per-expert `hipMalloc`. The whole
  MoE weight space is one VA reservation sized to what fits, which is what makes
  6 GiB and 8 GiB profiles work at all.
* Slot count per profile: 6 GiB → small fixed pool; 8 GiB → moderate; 12/16 GiB →
  as large as the remaining budget allows after KV. C22 splits it.

**Done when:** no use-after-free under an eviction stress test with the
sanitiser build, and slot occupancy ≥ 80% under a routing-heavy workload.

---

## C9 — Residency manager

**Owns:** where every expert and KV page currently is.

States: `ABSENT, NVME_RESIDENT, LOADING_NVME, RAM_RESIDENT, LOADING_RAM,
VRAM_RESIDENT, PREFETCHED, EVICTING_VRAM, EVICTING_RAM, PINNED, INVALID`.

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

* Transitions are asynchronous and observable. Every transition emits a C21
  telemetry event.
* **A kernel never asks where something is.** It asks `ready()`, and if false
  the scheduler has already failed over to the CPU path (C19) rather than
  stalling.
* A demand request that cannot be satisfied from RAM immediately escalates to
  NVMe and returns "not ready" — never a blocking read.

**Done when:** a full-execution trace over a real model shows zero synchronous
NVMe reads in the attention path, and `KV_MISS_STALL_TIME / TOTAL_DECODE_TIME`
is under the configured threshold.

---

## C10 — Prefetch predictor

**Owns:** the thing that makes the RAM tier pay for itself.

Layered, cheapest first:

| level | signal | cost |
|---|---|---|
| L0 | current token's routing | free |
| L1 | previous token's routing for the same layer | free, nearly |
| L2 | recent per-layer expert history (run-length of last N tokens) | ~free |
| L3 | cross-layer: current router output predicts next-layer experts | one small matmul |
| L4 | request-level activation matrix (MoE-Infinity style) | per request, amortised |
| L5 | adaptive horizon from observed bandwidth + hit rate (ExpertFlow) | free |

The untrained version (L1) is the cheapest test and the one to ship first: the
project's own routing traces (`KNJ_TRACE_EXPERTS`) already answer whether it
predicts anything. Then L3, then L5.

```cpp
struct Prediction {
    std::vector<ExpertId> candidates;
    float   confidence;        // calibrated, not raw softmax
    uint32_t horizon_layers;
};
Prediction predict(uint32_t layer, const RoutingState&);
```

* Prefetch is an **optimisation, never a correctness requirement**. A wrong
  prediction must cost bandwidth, never a wrong answer.
* Prefetch is cancelled or deprioritised when the demand set is at risk of
  missing its deadline. Throughput never wins against the current request.

**Done when:** measured prediction accuracy on real traces (top-8 overlap with
next-token routing) is published per model, and the adaptive horizon demonstrably
beats a fixed horizon of any single value.

---

## C11 — Transfer engine

**Owns:** all bytes that move, and the schedule they move on.

```
NVMe --io_uring/O_DIRECT (Linux) | DirectStorage/IOCP (Windows)--> RAM
RAM  --async H2D, DMA, aligned, coalesced--> VRAM
VRAM --async D2H--> RAM
RAM  --> NVMe (admission-controlled)
```

```cpp
struct Extent { void* dst; const void* src; size_t bytes; uint64_t key; };
class TransferEngine {
public:
    TransferId submit_nvme_read (const Extent&, int priority);
    TransferId submit_h2d       (const Extent&);
    TransferId submit_d2h       (const Extent&);
    TransferId submit_nvme_write(const Extent&);
    void poll();  void flush();
};
```

Rules:

* Coalesce adjacent logical objects into one physical transfer; never issue a
  4 KiB random application-level read for a payload.
* 4 KiB alignment, large sequential extents, pre-registered host buffers, many
  outstanding requests. Extent size is tuned per host (64 KiB … 8 MiB).
* **Reads beat speculative writes**, always.
* Linux: `io_uring` with `O_DIRECT` where it wins, `RWF_HIPRI` priority for demand
  reads, `fallocate`d files for the cold store. Windows: DirectStorage with an
  IOCP/overlapped fallback; DirectStorage gives many-small-reads and low CPU
  overhead but does *not* make the SSD a VRAM extension, so the staging into GPU
  addressable memory is still our job.
* Demand reads get a deadline; prefetch reads do not get to starve them.

**Done when:** NVMe read bandwidth reaches a stated fraction of the device's
measured sequential rate with extents in the tuned range, and H2D/D2H overlap
compute — both visible in the C21 table as `transfer` device-time vs idle-time.

---

## C12 — Admission and eviction

**Owns:** what gets demoted, and what is worth writing to NVMe at all.

Placement score (weights tuned from telemetry, structure fixed):

```
score = 0.30*recency + 0.25*frequency + 0.20*reuse_prediction
      + 0.15*shared_prefix_value - 0.10*transfer_cost
```

Priority order that is never violated:
`PINNED > ACTIVE > IN-FLIGHT > PREFETCHED > WARM > COLD`.

NVMe admission:

```
persist  iff  reuse_probability * recompute_cost > write_cost + read_restore_cost
```

Initial policy: persist shared prefixes, persist suspended sessions after a TTL,
persist repeatedly-reused cold segments, persist nothing else. This is the
wear-control rule and it is not negotiable.

**Done when:** SSD write bytes per generated token stays under a configured
bound on a prefix-heavy workload, and eviction never touches an in-flight or
event-referenced page.

---

## C13 — Radix KV index

**Owns:** prefix reuse across requests and across turns.

```
TOKEN PREFIX
     |
  RADIX TREE  (block-granular, chunked-hash-tree leaves to keep lookups O(1))
     |
  KV block ids
     |
 RESIDENCY TABLE (C9)
     |
  VRAM | RAM | NVMe
```

* Blocks are content-hashed over `(model fingerprint, block size, canonical token
  ids, position offset)`. A hit returns a **shared, immutable, refcounted** KV
  block list — several live sessions pointing at the same block.
* The cache key is I5 in full: model/arch/tokenizer/attention-format/RoPE/KV
  codec/impl-version. Anything not in the key is not a key.
* **Chunked hash tree**, not a naive trie walk: the Feather/CHT lesson is that
  tree-traversal CPU cost is comparable to the GPU work it saves in a small
  engine. Cheap lookup matters more than an elegant structure.
* Hybrid models (gated-delta-net stacks) cannot roll a recurrent state back to an
  arbitrary prefix. Where a model has recurrent layers, the index stores the
  token-addressable KV only, and the session path replays a bounded suffix into
  the recurrent state — the Tail-Replay shape, which is the principled version of
  the fallback the current code already warns about. Hybrid layers are flagged in
  the index so a rollback request on a hybrid model returns "replay suffix",
  never "impossible".

**Done when:** a repeated-prefix benchmark shows TTFT reduction proportional to
shared prefix length, and two concurrent sessions sharing a prefix produce
identical logits on the shared span.

---

## C14 — KV cache engine

**Owns:** the paged KV the attention kernels read.

```
address = (model_id, layer_start, layer_count, token_start, token_count, kv_format)
```

* Layer slabs. A decode layer consumes one layer's K/V; paging all 48 layers to
  serve one wastes the transfer. Slab = 2 layers on 6 GiB, 4 on 8/12/16 GiB.
* Block table per session; physical blocks from the WARM pool or NVMe.
* Geometry is computed from the loaded config, never hard-coded:
  `bytes/token = 2 * layers * kv_heads * head_dim * sizeof(codec)`.
* Recurrent layers keep a separate state array with its own budget and its own
  residency, because it is a fixed-size state and does not page.

**Done when:** a KV page survives eviction and reload bit-identically (I7), and
context length scales to the profile's ceiling with the reported residency label
(VRAM-resident / RAM-resident / storage-assisted / storage-bound).

---

## C15 — KV codecs

**Owns:** how KV bytes are represented in each tier.

| tier | gfx1201 | gfx1031 |
|---|---|---|
| VRAM | FP8 E4M3 (native WMMA f8) | FP16/BF16 |
| RAM | same | same |
| NVMe | FP8, or TurboQuant 4bit-nc when admitted | FP16 |

* FP8 is the default **because of the hardware**: gfx1201 has a native FP8 path at
  2× FP16 throughput and rocWMMA supports `f8/f32/f32` there. gfx1031 has no
  FP8 at all, so proposing FP8 KV there would be proposing a kernel that does not
  exist.
* TurboQuant 4bit-nc is an **opt-in second tier behind a quality gate**, not a
  default: the vLLM study found it 2.3–3.7× capacity at 40–52% throughput cost
  versus BF16, useful only when KV is the binding constraint. 3-bit variants are
  rejected by that study on accuracy and are not offered.
* Quantisation is **online and fused into the KV write**: a KV page is written
  already in its storage representation, so there is no separate quantise pass on
  the miss path (I2).
* The codec is part of the cache key. A page written as FP8 is never read as
  BF16.

**Done when:** FP8 KV matches BF16 KV within the configured tolerance on
deterministic prompts, and the codec round-trips through the NVMe tier without
drift.

---

## C16 — Attention kernels

**Owns:** the largest single device-time consumer in a decode step (the profile
table in the reference example shows `attention` at 95% of device time).

Two kernels, not one:

* **Prefill**: tiled over Q blocks × KV blocks, GQA-aware head grouping,
  streaming softmax.
* **Decode**: single Q row per head, the whole point of the tiering work.

Structure borrowed from FA4 and HipKittens, translated to what RDNA actually has
(§8 of `00-verified-facts.md`):

* **two-tile ping-pong**: two independent KV blocks in flight per workgroup, so
  one block's softmax overlaps the other's matmul.
* **selective rescaling**: rescale the running max only when it actually changes,
  not every block. Real VALU savings on a SIMT datapath.
* **SGPR/VGPR split**: address arithmetic and LDS copies issued from the scalar
  unit while the vector unit does the math.

Implementation split:

| | gfx1201 | gfx1031 |
|---|---|---|
| math | WMMA `f16`/`bf16`/`f8` via the `_gfx12` builtin + LDS tiles | `v_pk_fma_f16` + LDS tiles, 2 lanes/op |
| KV format | FP8 or FP16, dequant-to-FP16 in the prologue | FP16 |
| fallback | SIMT path, always compiled | the only path |

No library to wrap: ROCm FlashAttention does not support RDNA4. This is written
from scratch and validated against a CPU reference.

**Done when:** the decode kernel matches a CPU reference within tolerance for
GQA, MQA and MLA at head_dim 64/128/192, and prefill throughput is within a
stated fraction of the device's measured WMMA ceiling.

---

## C17 — MoE expert GEMM

**Owns:** the other half of decode device time, and the reason any of this works.

```
router -> group tokens by expert -> active expert set -> waves
       -> grouped gate/up GEMM -> activation -> grouped down GEMM
       -> weighted accumulation into the residual
```

* **Per-token expert launches are prohibited.** Group or die on load.
* Waves: when the active set exceeds the slot pool, execute in waves and
  accumulate every selected expert's contribution. Contributions are never
  dropped (I3).
* Weights arrive kernel-native packed from C5. No dequantise-to-FP16 staging on
  the miss path (I2).

| format | gfx1201 | gfx1031 |
|---|---|---|
| W8 | WMMA i8 + scale, or dequant-to-fp16 + WMMA f16 | `v_dot4_i32_i8` inline asm — native, measured |
| W6 | unpack + WMMA f16 | `v_dot4_i32_i8` + unpack (group 64) |
| W4 (default) | unpack + WMMA f16 | **`v_dot8_i32_i4` — int4 dot consumes packed nibbles with no unpack** |
| FP8 | native WMMA f8 | unavailable |
| ternary | optional, experimental, quality-gated | optional |

**Done when:** grouped GEMM output matches a materialised reference within
tolerance, and the profiler shows expert GEMM device time is not dominated by
unpack arithmetic (if it is, the pack format is wrong, not the kernel).

---

## C18 — Router

**Owns:** choosing the experts. It must be exactly the model's router.

* Read `exp_probs_b.bias` under **both** spellings — the prior defect where the
  wrong spelling silently ran expert selection un-biased is the reason. Host
  memory, freed on the host, never a device pointer.
* Top-k selection is exact. A sampled-boundary top-k with a mandatory exact
  certification pass is acceptable (the HPC-Ops result: sampling controls
  common-path work, never the answer) and testable against the current
  exhaustive selection.
* Routing stability is measured, not assumed: quantised hidden states can change
  later routing. The compiler records top-1 agreement, top-k agreement, Jaccard,
  logit RMSE and router margin change.

**Done when:** un-biased execution is impossible to reach by configuration, and
routing-agreement metrics are reported for every compiled model.

---

## C19 — CPU / direct-memory expert fallback

**Owns:** the escape hatch when an expert will not arrive in time.

The central observation from Fiddler: the activation tensor is far smaller than
the expert weights, so for small batches it can be cheaper to compute a missing
expert on the CPU than to move it over PCIe.

```cpp
struct FallbackCost { uint64_t transfer_us, cpu_us; bool prefer_cpu; };
FallbackCost estimate(ExpertId, uint32_t batch_tokens);
```

* Enabled when `batch_tokens <= threshold` **and** the expert is not resident
  **and** predicted transfer cost exceeds CPU cost.
* With ReBAR on, the warm pool can be mapped into the GPU address space as a
  last-resort direct-access path — still PCIe-bound per access, so it belongs
  here, never in attention.
* Must not block unrelated GPU work and must not mutate the canonical
  representation.

**Done when:** on a 6 GiB profile, cold-expert requests complete within the
latency SLO instead of stalling, and the CPU-fallback counter in telemetry is the
thing that tells you whether the predictor is good enough.

---

## C20 — Speculative decoding

**Owns:** converting leftover bandwidth into tokens.

See `05-speculation.md` for the full design. In one line: **MTP first** (the head
is already resident and costs zero I/O), then one loader for DFlash/DFlash2/
DSpark, then DSpark's confidence head wired into C10 so drafting stops early when
the next verification pass would stall on cold experts anyway.

---

## C21 — Profiler and telemetry

**Owns:** the `--profiling` subsystem. Full specification in `06-profiling.md`,
because it is a first-class deliverable with its own output contract, its own
instrumentation-floor measurement, and its own acceptance tests.

Short version: per-component `ops / %dev / dev us / idle us / host us`, emitted in
the fixed column layout of the reference example, with the instrumentation floor
measured and reported alongside, so every number is interpretable.

---

## C22 — Memory budget manager

**Owns:** turning detected hardware into the policy numbers.

```cpp
struct Profile { uint64_t vram_ceiling; uint32_t token_block, layer_slab,
                 size_t prefetch_bytes; size_t expert_slots; size_t warm_kv;
                 uint32_t lookahead_tokens; int h2d_inflight; };
Profile select();  // from (VRAM class x RAM class) and measured bandwidth
```

Rows for VRAM ∈ {6, 8, 12, 16} GiB × RAM ∈ {16, 24, 32, 48, 64, 96} GiB. Values
are **starting points overridden by measurement**, never constants: the autotuner
(C3) rewrites block size, slab depth, prefetch depth and transfer extent from
observed bandwidth and hit rate.

```toml
# kanjoos.toml — a starting point, not a decision
[gpu]
vram_class = "auto"          # 6 | 8 | 12 | 16
vram_ceiling_gib = 0         # 0 = measured
token_block = 32
layer_slab = 4
prefetch_mib = 8

[ram]
warm_fraction = 0.65
min_free_gib = 8
pinned_pool_gib = 3

[nvme]
cache_fraction = 0.35
max_cache_tib = 2

[profile]
profiling = "off"            # off | counters | full | table
```

**Done when:** a run on each VRAM class reports the profile it selected and the
measurement that justified it, and overriding any value in the file visibly
changes the run.

---

## C23 — Session scheduler and server

**Owns:** multiple requests, and choosing which to run.

* OAI/Anthropic-compatible HTTP surface (the shape Strata serves; being
  better means being compatible, not novel).
* Batching: **prefix homogeneity beats batch size** for KV locality — the
  scheduler prefers requests that share a prefix over simply filling the batch.
* Sessions carry `{session_id, priority, prefix_hash, active_token_range,
  required_blocks, prefetch_deadline}` into C9/C10/C11.
* Background maintenance (NVMe eviction, compaction, autotune writes) never
  starves an active inference.

**Done when:** a mixed shared-prefix workload shows a measurable TTFT win versus
FIFO batching, and a background compaction pass provably does not delay a decode
step.

---

## C24 — Platform layer

**Owns:** Windows 11 and Linux differences, and the ROCm packaging.

**Windows 11**
* DirectStorage preferred, IOCP/overlapped fallback.
* `VirtualLock` for the pinned pool; `FILE_FLAG_SEQUENTIAL_SCAN` /
  `SetFileInformationByHandle` for read-ahead.
* hipcc.bat on PATH; `--offload-arch` per target; no `HSA_OVERRIDE_GFX_VERSION`
  needed (both targets are natively supported in the Linux channel; Windows
  gfx1031 is best-effort and must say so).
* ReBAR verified through HIP device properties, not assumed.

**Linux**
* `io_uring` + `O_DIRECT`, 4 KiB aligned, large extents, pre-registered buffers.
* `madvise(MADV_WILLNEED | MADV_HUGEPAGE)`, `mlock` bounded by RLIMIT.
* ReBAR via `/sys/class/drm/card*/device/resource` and the `amdgpu` resize state.
* The gfx1031 **kernel packs** (`blas/fft/rand` kpack) are installed and used as
  the microbenchmark baseline; the `torch`/`torchvision`/`rccl` packs are
  irrelevant to the engine and are documented as such so nobody tries to link
  them.

**Rule:** every platform difference goes behind an interface in C24. No
`#ifdef _WIN32` above this layer, ever.

**Done when:** the same binary and the same test suite pass on Windows and Linux
for both targets, and a single `kanjoos doctor` command prints the resolved
platform facts (ReBAR state, pinned pool, extents, queue topology, capability
table) that C21 then uses in its report.