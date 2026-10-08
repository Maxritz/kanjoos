# 04 — Memory tiering and residency

This is the part that decides whether the engine is usable. Kernels decide
whether it is fast once the data is there; tiering decides whether the data is
there in time.

---

## 1. ReBAR is a prerequisite, not an optimisation

Every VRAM number in this plan assumes a full aperture.

* **Off:** a consumer card exposes a 256 MiB BAR1. A 16 GiB card is then a
  256 MiB card as far as the GPU is concerned, and every profile in this document
  is fiction.
* **On:** the aperture covers the full device, and the budget tables below are
  real.

Therefore `kanjoos doctor` refuses to start a decode session if ReBAR is off and
the model does not fit in the aperture it reports. It says so in one line and
tells the user the BIOS setting. Silent thrashing here is the worst possible
outcome, because it looks like a performance bug.

| platform | how to verify |
|---|---|
| Linux | `/sys/class/drm/card*/device/resource` size vs `hipDeviceProp_t.totalMem`; `amdgpu` resize state |
| Windows | `hipDeviceGetAttribute(hipDeviceAttributeTotalMemory)` vs board VRAM; PCI resource sizes |
| both | functional check: allocate more than the reported aperture and observe it fail cleanly, not corrupt |

ReBAR also enables **host-mapped warm memory**. That is a legitimate last-resort
path in C19 — a kernel can read a warm expert over the BAR without a staging
copy — but it is still PCIe latency per access, so it belongs in the expert
fallback, never in attention.

---

## 2. VRAM profiles

Detected VRAM class → starting policy. Every number is a *starting point* that the
autotuner (C3) may override, and every budget is measured at startup.

| VRAM | ceiling | KV block | layer slab | lookahead tokens | H2D in flight | prefetch extent | expert slots |
|---|---|---|---|---|---|---|---|
| 6 GiB | 5.25 GiB | 16 | 2 | 32–128 | 2–4 | 2–8 MiB | small fixed pool |
| 8 GiB | 7.00 GiB | 32 | 4 | 64–192 | 4–6 | 4–16 MiB | measured |
| 12 GiB | 10.50 GiB | 32 | 4 | 64–256 | 4–8 | 4–16 MiB | measured |
| 16 GiB | 14.00 GiB | 32 | 4 | 128–512 | 8–16 | 8–32 MiB | largest that fits |

The allocation split at startup, in this order:

```
VRAM_CEILING
  - MODEL_DEVICE          (trunk + attention + router + head, resident)
  - WORKSPACE             (activations, LDS, expert staging, logits)
  - SAFETY                (measured high-water mark of the runtime's own allocations)
  = HOT_KV  +  EXPERT_SLOTS
```

The KV/expert split is a policy, not a constant, and it is the most consequential
policy in the engine:

* **More VRAM to KV** → longer context, but more cold experts per token.
* **More VRAM to expert slots** → higher hit rate, but less context.

Default split by profile: 6 GiB → 35% KV / 65% experts. 8 GiB → 45/55. 12 GiB →
50/50. 16 GiB → 45/55, biased back toward experts when the model's active set is
bandwidth-bound (detected at runtime from the C21 profile, not guessed). The
user can pin either side; the engine reports the consequence in tokens/s and
context length rather than hiding it.

### 2.1 The 6 GiB case deserves its own note

At 5.25 GiB with a 30B-class MoE, the model itself may not fit. That is not a
failure mode, it is a workload: either a smaller model, or full RAM residency
with a large warm tier, or a heavily expert-compressed build (C5 mixed precision)
where the expert bank fits W4/W6 and the trunk is small. The engine's job is to
say which of those it is doing and what it costs — which is why C21 reports the
residency label (`VRAM-resident / RAM-resident / storage-assisted /
storage-bound`) on every run.

---

## 3. RAM profiles

```
WARM_KV_LIMIT = min(0.65 * phys_ram, phys_ram - 8 GiB)
PINNED_POOL   = 2 GiB (16/24/32), 3 GiB (48/64), 6 GiB (96)
```

| RAM | WARM KV target | warm experts (typical) | pinned | OS/runtime headroom |
|---|---|---|---|---|
| 16 GiB | 10.4 GiB | small | 2 GiB | 6 GiB |
| 24 GiB | 15.6 GiB | ~14 GiB | 2 GiB | 5–8 GiB |
| 32 GiB | 20.8 GiB | ~20 GiB | 2 GiB | 7 GiB |
| 48 GiB | 31.2 GiB | ~32 GiB | 3 GiB | 10 GiB |
| 64 GiB | 41.6 GiB | ~46 GiB | 3 GiB | 12 GiB |
| 96 GiB | 62.4 GiB | ~70 GiB | 6 GiB | 16 GiB |

At 96 GiB, W4 experts for a 30B-A3B (14.13 GiB) fit entirely warm, and so does a
30B at W6 (20.7 GiB) with room to spare. **That is the configuration where this
engine is genuinely excellent**, and it is worth saying plainly: the design
target is a 96 GiB host with a 12–16 GiB card running a 30B-class MoE at usable
speed, not a 6 GiB card doing the impossible.

One correction the KV roofline forced here: on a **16 GiB card** that W4 bank
is 88% of all VRAM, leaving a **negative** KV budget once attention,
embeddings and the LM head are resident (`00-verified-facts.md` §9.5). And
because routing locality is weak (§9.6), evicting experts to make room for KV
costs 3–9 ms/token — more than a W4→W3 repack frees for free. The host RAM
tier is therefore not a *cache* for this model on this card. It is where the
other weights have to live.

Dynamic shrink is not optional. Under OS pressure the engine stops NVMe
prefetch, then shrinks WARM, then retains only the active request's pages. The
order is fixed and tested.

Pinning stays bounded no matter what. A 96 GiB host must not be pinned to
"simplify transfers"; the pinned pool exists to make DMA efficient in
bounded chunks, not to make the whole machine fast.

---

## 4. NVMe cold tier

```
NVME_QUOTA = min(35% of SSD capacity, 2 TiB), configurable,
             with >= 20% of the device kept outside the quota
```

Two classes of data live here and nothing else does:

1. **The canonical checkpoint**, read-only, read via the cold index (C4/C6).
2. **Admitted KV**: shared prefixes, suspended sessions after a TTL, and
   repeatedly-reused cold segments.

Not persisted: transient one-shot completions, anything with low reuse
probability, and anything whose write cost exceeds the recompute it would save.
On a workload with a few GB/s of compulsory reads already, writing every KV page
would spend the disk budget on data nobody reads twice.

### 4.1 I/O shape

| path | mechanism | extent | notes |
|---|---|---|---|
| Linux | `io_uring`, `O_DIRECT` where it wins, `RWF_HIPRI` for demand reads | 64 KiB – 8 MiB, tuned | pre-registered buffers, many in flight |
| Windows | DirectStorage, IOCP fallback | same, tuned | many-small-reads friendly; staging into GPU-addressable memory is still ours |
| both | 4 KiB aligned, sequential, coalesced | — | never a 4 KiB application-level random read of a payload |

The logical/physical split, restated because it is the single most common design
error in this class of system: **residency decides per expert, I/O moves
multi-expert extents.** Fine-grained decisions, coarse-grained transfer.

---

## 5. Prefetch

### 5.1 The discipline

```
predict  ->  plan  ->  prefetch  ->  RAM  ->  VRAM  ->  attention
```

The attention path never issues a storage read. If a required page is not in
VRAM when its layer arrives, the scheduler has already either:

1. found it in RAM and issued H2D early enough,
2. found it in NVMe and issued the read early enough,
3. or failed to, and taken the CPU/expert fallback (C19) rather than stalling.

A hard stall is a last resort and a telemetry event (`STALL_TIME_MS`), never the
mechanism.

### 5.2 Horizons

Initial policy, per profile, tuned by C3 and adapted at runtime by C10-L5:

| decode shape | horizon |
|---|---|
| batch 1 | 1–2 future layers |
| small batch | 2–4 future layers |
| prefill / large batch | widen only if overlap measurably improves |

The prefetcher self-throttles: it estimates whether a speculative read will land
before its deadline, and cancels or deprioritises reads whose confidence is below
the current deadline pressure. Confidence is calibrated against measured hit
rate, not raw router softmax.

### 5.3 Eviction

Score-based, not LRU:

```
0.30*recency + 0.25*frequency + 0.20*reuse_prediction
  + 0.15*shared_prefix_value - 0.10*transfer_cost
```

with the hard priority order `PINNED > ACTIVE > IN-FLIGHT > PREFETCHED > WARM >
COLD` that no score may violate. In-flight and event-referenced pages are never
evictable.

Shared prefixes get a bonus because they are the highest-value thing in the
cache: a shared prefix is reused by construction, and it is the one object whose
persistence to NVMe clearly pays.

---

## 6. KV cache residency

KV has its own index (C13) and its own tiers, sharing C9's machinery:

* **Block size** 16 tokens (6 GiB) / 32 tokens (8/12/16 GiB).
* **Layer slabs** 2 (6 GiB) / 4 (8/12/16 GiB).
* **Recurrent state** is not paged — it is a fixed-size state per layer with its
  own budget line.
* **Mixed-precision, not mixed-format.** One codec per page, chosen by tier
  default, recorded in the key.

Recompute is the fallback of last resort, and its rate
(`KV_RECOMPUTE_RATE`) is one of the primary KPIs: it is what tells you the warm
tier is too small.

> **EXTENDED by [09-kv-engine-architecture](09-kv-engine-architecture.md) section 5.**
> "Pages move freely between tiers" above is an implicit model and it is the
> wrong one: RAM is reachable only at PCIe speed (13.6 GB/s measured against
> 597 GB/s in VRAM), so a session that spills is not slower, it is a different
> class with a stated step time. Sessions are assigned **class A (resident),
> B (spilled, priced) or C (suspended)** and stay there until admission says
> otherwise. Spilling quietly is how a 20 tok/s claim becomes a 2 tok/s surprise.

### 6.1 The context-class ladder, and the refusal path

This subsection **replaces** the implicit model above. Every number in it is
derived in `00-verified-facts.md` §9 (the weight-format decision is §9.8) and
produced by `tools/kvroof/kv_roofline.py`, which reads the model's
`config.json` rather than assuming its shape. The full operating-point
derivation, including the per-pack hit-rate and miss-cost columns that depend
on the measured P0-7 curve, is in `01-architecture.md` §4.2 and audited by
`kv_roofline.py --check` section K.

A session is admitted into exactly one class and **stays** there. The class is
a promise to the user, made before the first token, and it appears in the API
response. It is not a state the engine drifts into under load.

| | **A — RESIDENT** | **B — SPILLED** | **C — SUSPENDED** |
|---|---|---|---|
| where the KV lives | VRAM (T1) | hot span in VRAM, cold tail in pinned RAM (T2) | NVMe (T3), serialised |
| who chose it | admission (C12), from the budget | admission, **with a stated step time** | the client, explicitly |
| step-time cost of the cold tier | 0 | **measured**, never assumed | 0 — not decoding |
| when it is used | anything that fits the budget | a session that must not lose history | anything not currently decoding |

**Class A**, at 16 GiB with a W4 pack, has **zero** KV budget: the resident set
is 15.31 GiB before workspace. So the honest class-A ceiling on this card comes
from the weight format, not from a codec choice:

| pack | KV budget | FP16 context | FP8 context | experts resident |
|---|---|---|---|---|
| W4 g128 | 0 | — | — | 127 / 128 |
| **W3 g128** | **3.46 GiB** | **37.8 K** | **75.6 K** | **126 / 128** |
| W2 g128 | 6.98 GiB | 76.2 K | 152.4 K | 123 / 128 |

**W3 g128 is the operating point.** It is the only row whose FP16 context fits under
the model's own 40,960-token ceiling while keeping essentially the whole expert
bank resident. W2's extra capacity is mostly unusable — the model cannot attend
that far — and W4's is nonexistent. The decision is a **capacity** decision,
not a bandwidth one: what it costs to buy 1 GiB of KV is in
`01-architecture.md` §4.2 (W4→W3 is free in time and frees 3.46 GiB; evicting
6 experts/layer costs 6–9 ms/token for less capacity than the repack).

**Class B is priced, and the price is disqualifying for chat.** At the measured
13.4–14.7 GB/s, a cold span costs per step:

| cold context | FP16 | FP8 |
|---|---|---|
| 8 K | 54.8–60.1 ms | 27.4–30.0 ms |
| 16 K | 109.6–120.2 ms | 54.8–60.1 ms |
| 64 K | 438.3–480.8 ms | 219.1–240.4 ms |

That is the **whole step budget**, spent before a single FLOP of compute. Class
B exists for exactly one job — a session that may not lose history — and the
admission response must carry the number, not the word "slow".

**Class B does not apply to experts, and this is now measured rather than
assumed.** Because routing locality is weak (`00` §9.6), evicting experts to buy
KV context costs 3–9 ms/token to free a few GiB. The engine must not offer an
expert-spill knob and call it a cache.

### 6.2 The refusal path

When a request does not fit the class-A budget, admission does exactly one of
three things, in this order:

1. **Admit at a shorter context.** The common case. The client is told the cap
   in tokens and the truncation is visible in the response, not inferred.
2. **Admit as class B with the price attached.** Only if the client opts in and
   the stated step time is within the client's own budget.
3. **Refuse.** If neither fits. Refusal is a valid outcome and is never
   silently downgraded to class B.

Three rules make this enforceable rather than aspirational:

* **No silent demotion.** A session is never moved from A to B because load rose.
  If it must move, it is an explicit event with a reason, and the client sees it.
* **No silent codec change.** A codec that changes values is a *declared
  precision reduction* (§9.7 of `00`), not a tier move. Changing precision
  mid-session is an event, not a detail.
* **Refusal is cheap.** Admitting a request the card cannot serve converts one
  bad answer into a bad answer for every request behind it. Refusing early is
  the cheaper failure.

The measurement that makes this section necessary rather than stylistic: on a
16 GiB card, W4 leaves **zero** KV budget. An engine with no admission
controller does not serve a long context slowly — it serves it wrongly, because
somebody has to decide what to overwrite.

**Operating point.** The Class A ceiling on this card is not a KV-capacity
number; it is a weight-format number (`00` §9.8, `01` §4.2). **W3 g128 is the
operating point**: 11.79 GiB resident, 3.46 GiB KV budget, 37.8 K tokens of
FP16 context — the only pack whose context fits under the model's 40,960-token
ceiling while keeping essentially the whole expert bank resident. W4 leaves zero
KV budget; W2 buys capacity the model cannot attend to. The decision was cheap
to make and is machine-audited by `kv_roofline.py --check` section K.

---

## 7. What the KV tier must report

Every run reports, and C21 surfaces these in the profile header:

| label | meaning |
|---|---|
| VRAM-resident | the whole active KV set is in device memory |
| RAM-resident | it is in host memory, being streamed with prefetch |
| storage-assisted | some of it comes from NVMe, and that is visible |
| storage-bound | NVMe is on the critical path and decode is I/O limited |

A run that reports `storage-bound` is telling the truth about itself. That
honesty is a design requirement: it is what stops an I/O problem being
misdiagnosed as a kernel problem, which is the most expensive mistake available
in this class of work.