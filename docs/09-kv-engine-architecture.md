# 09 — Tiered KV engine: architecture from first principles

This document does three things: it derives the engine from the physics rather
than from the component list, it states how the pieces connect, and it checks
the result against the tree that exists. It supersedes the tier framing in
`01-architecture.md` §2 and `04-memory-tiering.md` where the two disagree, and
says explicitly where they disagree.

Everything numeric here is **REASONED, not MEASURED**, unless it is tagged with
a § reference in `00-verified-facts.md`. That distinction is the point of this
document: the design must be defensible before it is fast.

---

## 1. The reframe: two resources, one contention point

Everything in this engine reduces to one question: **what gets to sit in VRAM?**

There are only two things that want VRAM, and they are not comparable:

| resource | size | growth | read pattern |
|---|---|---|---|
| **expert weights** | 14.13 GiB at W4 G128 (fixed) | none — bounded by the model | top-k per token, layer by layer |
| **KV cache** | `tokens × bytes_per_token` | **unbounded in context** | **100% miss, every token, every layer** |

That table is the whole architecture. The asymmetry in the last row is not a
detail, it is the design:

- **Weights are cacheable.** Route locality means the same handful of experts
  recur. A hit costs zero bytes and zero time. Residency converts a transfer
  problem into a lookup problem, and lookups are cheap.
- **KV is never cacheable.** To answer "what did token 4,000 attend to", you
  must *read token 4,000's KV*. There is no reuse, because the question
  changes every token. Its read volume equals its size, every step, forever.

## 2. The crossover, in numbers

Per generated token, at M=1, with every factor now READ from
`models/qwen3-30b-a3b/config.json` by `tools/kvroof/kv_roofline.py`:

```
weights  = 4,718,592 x 8 x 48 x 0.5234 B/w       ~ 948 MB   (W4 G128)
         + attention q/k/v/o                    ~ 474 MB
         + lm_head 151,936 x 2048 x 0.5234      ~ 163 MB
                                                    --------
                                             ~ 1,586 MB  ALL of it
KV       = context x 48 x 4 x 128 x 2 x 2 B      ~ 96 KiB/token
```

The `4` and the `128` were previously an assumption. They are correct — but an
assumption that happens to be correct is a fact waiting for a config change to
become a lie, and four of the five configs in `models/` disagree with it. The
table below is reproduced from the tool, which exits non-zero rather than
falling back.

| context | KV read/token | weight read/token | ratio | dominant |
|---|---|---|---|---|
| 2 K | 201 MB | 948 MB | 0.21x | weights |
| 8 K | 805 MB | 948 MB | 0.85x | weights |
| **16 K** | **1.61 GB** | 948 MB | **1.70x** | **KV** |
| 32 K | 3.22 GB | 948 MB | 3.40x | KV |
| 128 K | 12.9 GB | 948 MB | 13.6x | KV, hopeless |

**KV overtakes weight traffic at roughly 9.6 K context** (948e6 / 98304 B =
9647 tokens)— and it never comes back. Note the crossover is *past* the 8 K
mark, so reading the weight budget and concluding "8 K is fine" is wrong by a
factor of two at exactly the context length most products ship with.

**The 948 MB figure is only the experts.** A decode step also reads the whole
attention projection set (474 MB/token) and the whole LM head (163 MB/token),
because there is one weight read per layer per token and the vocabulary is
151,936 wide. Counting all of it, the expert-only crossover moves from 9.6 K to
**16.1 K**, and the model's own ceiling is 40,960 tokens. The crossover is a
property of what you count, so this document now counts everything and labels
the subset.

`01-architecture.md` §4 sizes the entire plan around the 0.93 GB/token weight
figure. That is the right number for *cold start* and the wrong number for
*steady-state decode*, which is where an inference engine actually lives. Both
must be budgeted, but only one of them gets worse as the product works.

### 2.1 What this kills

- **Streaming KV from NVMe on the hot path.** Unreachable by construction, and
  already forbidden by invariant I4. Good — that one was right.
- **Treating RAM as a symmetric overflow tier for KV.** RAM is reachable only at
  PCIe speed. Paging KV there converts a working set into a wire, and the wire
  is 45x slower than VRAM (597 GB/s measured vs 13.6 GB/s measured, §8.1).
- **"Just batch more tokens."** Batching amortises *weights*. It does not
  amortise KV, because every token needs its own prefix read. Batching makes the
  weight term vanish and leaves the KV term untouched.

## 3. The consequence: KV residency is the product

Everything else is support. So the engine is designed around a single scalar —
**how many KV bytes may be resident, and at what precision** — and everything
else negotiates against it.

This yields a design principle the current docs do not state:

> **KV residency is the product feature. Weight residency is an optimisation.**

Getting that backwards produces a very fast engine that can only remember 900
tokens.

---

## 4. The four tiers, with honest costs

| tier | medium | residency | cost to touch | role |
|---|---|---|---|---|
| **T0** | VGPR/SRAM + LDS | per-kernel working set | ~0 | current window, online-softmax accumulators |
| **T1** | **VRAM** | the KV working set | ~0 (1 TB/s+) | the only tier decode actually uses |
| **T2** | pinned RAM | spilled KV pages | **13.6 GB/s measured** (§8.1) | overflow only, latency-charged |
| **T3** | NVMe | suspended sessions | ~4 GB/s | **never on the decode path** |

The design rule that follows: **T2 is a promise to the user, not a fallback the
engine takes quietly.** If a session needs more than T1 holds, the admission
controller either grants it an explicit context class with a stated step-time
cost, or refuses it (invariant I3). Silently dropping to T2 is how a "20 tok/s"
claim becomes a "2 tok/s" surprise.

---

## 5. Context classes — the state machine that replaces "tiering"

Rather than pages moving silently between tiers, a session is **assigned a
class** and stays in it. The class is a user-visible promise.

### Class A — RESIDENT (the good case)

```
KV entirely in VRAM at a chosen codec.
context <= VRAM_KV_BUDGET / bytes_per_token(codec)
Cost: step time is attention-compute bound. No transfer on the hot path.
```

The budget is **not** free, and this is where the first version of this
document was wrong twice. Both errors are corrected below by
`tools/kvroof/kv_roofline.py`, which derives every number from the model's own
`config.json` (48 layers × **4** KV heads × **128** head_dim — read, not assumed;
the assumption happened to be right, which is the worst way to be right).

**What a 6 GiB KV budget actually buys** (Qwen3-30B-A3B, geometry from config):

| codec | bytes/elem | KV B/token | resident context at 6 GiB |
|---|---|---|---|
| FP16 | 2.0000 | 98,304 (96.0 KiB) | 65.5 K |
| FP8 (E4M3) | 1.0000 | 49,152 (48.0 KiB) | 131.1 K |
| INT8 g64 | 1.0469 | 51,456 (50.2 KiB) | 125.2 K |
| INT8 g128 | 1.0234 | 50,304 (49.1 KiB) | 128.1 K |
| INT4 g64 | 0.5469 | 26,880 (26.2 KiB) | 239.7 K |
| INT4 g128 | 0.5234 | 25,728 (25.1 KiB) | 250.4 K |

Two errors in the previous draft of this table, both found by the tool rather
than by reading it carefully:

1. **INT4 g128 was written as 24 KB/token.** It is 25,728 B = **25.1 KiB**, not
   half of FP16. Every quantised format carries 3 B of metadata per group
   (fp16 scale + uint8 zero-point, `gemm_w4.hip`), so INT4 costs 0.5234 B/elem
   against FP16's 2.0 — a 3.82x reduction, not 4x, and never the ~2x the table
   implied.
2. **The resident-context column did not follow from the bytes column.** With
   INT4 g128 at its true 25,728 B/token, 6 GiB holds 250 K tokens, not the
   256 K that the wrong byte count implied.

And a ceiling the table could not see, because it was reading a config nobody
had opened: **`max_position_embeddings` is 40,960.** The model cannot attend
past 41 K however much VRAM the codec frees. Rows above that are unreachable
capacity, not features.

**But a 6 GiB KV budget does not exist on this card.** At W4 the resident set
for this model is 15.31 GiB — 14.13 GiB of expert bank plus attention,
embeddings and the LM head — and VRAM is 16 GiB. The KV budget is
**negative** before workspace is even subtracted. Class A is not a KV-capacity
problem; it is an allocation problem, and the allocation is decided between
weights and KV, not by a codec. See §5.1.

### 5.1 Class A is an allocation decision, and P0-7 decided it

The bytes above are not free to spend on KV. On 16 GiB of VRAM:

```
resident weights at W4   15.31 GiB   (14.13 expert bank + 1.17 other)
workspace reserve         0.75 GiB
KV budget                -0.06 GiB   <-- negative
```

So "how much context fits?" has no answer until **"how much of the model do
you keep resident?"** has one. That question is P0-7, and it is now measured
(`tools/route/route_locality.py`, results in `00-verified-facts.md` §9.6):

| KV budget | FP16 context | expert slots/layer | LRU hit rate | miss cost/token |
|---|---|---|---|---|
| 0.25 GiB | 2.7 K | 125 of 128 | ~99% | ~0.4 ms |
| 1.00 GiB | 10.9 K | 118 of 128 | ~96% | ~2.5 ms |
| 2.00 GiB | 21.8 K | 109 of 128 | ~93% | ~4.0 ms |
| 4.00 GiB | 43.7 K | 91 of 128 | ~88% | ~6.9 ms |
| 6.00 GiB | 65.5 K | 73 of 128 | ~83% | ~8.7 ms |

**Routing locality is weak.** The target's own gate matrices select a median
115 of 128 experts per layer, and only 41% of a token's top-8 survives to the
next token. There is no small hot set to keep: the hit-rate curve is close to
linear in slot count, and 70% of the bank is needed for a 95% hit rate.

Each miss costs one PCIe hop for one expert in one layer. At the measured
13.4-14.7 GB/s that is 168-184 µs, so a decode step pays **8.1-8.9 ms for a
single missed expert per token**, against 7.68 µs of arithmetic at 100% of peak.
Trimming the bank by 27% to gain 6 GiB of KV turns a 0.8 ms transfer-bound step
into an 8-10 ms one.

> **Expert residency is all-or-nothing, and the W4 pack is exactly the
> constraint that makes it impossible.** The engine cannot buy context by
> evicting experts. It has to buy context by **shrinking the weights** (W3
> frees ~3.5 GiB, W2 frees ~6.6 GiB, §8.10) or by capping context short.

The weight-format decision is now the whole decision (`00` §9.8,
`01` §4.2). At 16 GiB VRAM, 0.75 GiB workspace, same 3 B/group metadata rule as
`gemm_w4.hip` whatever the bit width:

| pack | B/weight | expert bank | resident | KV budget | FP16 context | FP8 context | experts resident |
|---|---|---|---|---|---|---|---|
| W4 g128 | 0.5234 | 14.13 GiB | 15.31 GiB | **0** | — | — | 127 / 128 |
| **W3 g128** | **0.3984** | **10.76 GiB** | **11.79 GiB** | **3.46 GiB** | **37.8 K** | **75.6 K** | **126 / 128** |
| W2 g128 | 0.2734 | 7.38 GiB | 8.27 GiB | 6.98 GiB | 76.2 K | 152.4 K | 123 / 128 |

**W3 g128 is the operating point.** It is the only row whose FP16 context fits
under the model's own 40,960-token ceiling while keeping essentially the whole
expert bank resident. W4 leaves zero KV budget; W2 buys capacity the model
cannot attend to. The decision is machine-audited by `kv_roofline.py --check`
section K.

This inverts the optimistic reading of Class B. Class B was priced for KV; it
now also has to be priced for **experts**, and experts are worse: a KV page
miss is one page, an expert miss is a full 2.36 MiB weight triple at every one
of 48 layers.

### Class B — SPILLED (explicit, priced)

```
hot window in VRAM (T1), cold tail in pinned RAM (T2)
attention splits: hot span computed on device, cold span streamed.
```

The ceiling is arithmetic, not engineering:

```
cold_bytes_per_step = cold_context x bytes_per_token
t_cold = cold_bytes_per_step / 13.6e9      <- MEASURED PCIe, not an assumption
```

| cold context | FP16 | FP8 |
|---|---|---|
| 1 K | 7.4 ms | 3.7 ms |
| 2 K | 14.8 ms | 7.4 ms |
| 4 K | 29.6 ms | 14.8 ms |
| 8 K | 59.2 ms | 29.6 ms |
| 16 K | 118.4 ms | 59.2 ms |
| 64 K | 473.7 ms | 236.9 ms |

**This kills Class B for anything conversational.** A 16 K context at FP8
spends **59 ms per token** on the wire before a single FLOP of compute. That is
17 tok/s with a perfect kernel and zero model forward pass— which is not a product.

An earlier draft of this table claimed 16 K costs 2.4 ms. It was wrong by a
factor of 48 (a unit slip from KB to MB). The error matters because it would
have made spilling look routine instead of unusable, so the corrected number is
recorded next to the wrong one deliberately.

So the conclusion is stronger than "spilling is slow":

> **Spilling KV to host RAM is not a scaling strategy.** For a fixed wire
> budget, context scales with `budget x PCIe / bytes_per_token` and nothing
> else. At the measured 13.4-14.7 GB/s and a 10 ms per-step allowance that is
> **~1.4 K tokens of extra context at FP16 and ~2.8 K at FP8** — the codec
> buys exactly 2x, because it halves the bytes.
>
> Class B exists for one job— a session that must not lose history—and
> it is priced accordingly. Class A is the product.

The earlier phrasing of this claim — "buys about 2 K at FP16 and 8 K at FP8" —
was not reproducible, because **the wire budget was never stated**. Any number
here is budget-relative; `kv_roofline.py --wire-ms` takes the budget as an
argument rather than hiding it in the prose. The direction and the 2x ratio are
what matter, and those survive.

That table belongs in the admission controller, not in a comment.

### Class C — SUSPENDED

Not decoding. Serialise to NVMe (T3). Zero hot-path exposure by construction.

---

## 6. Codecs are capacity, not polish (C15)

Given §2, codec quality is the *primary capacity lever* — it multiplies Class A
context directly.

The valuable one is not a format, it is a **resolution ladder**:

```
tokens 0    .. 2K     FP8   full fidelity   (attention sink + recent)
tokens 2K   .. 32K    INT8  per-group
tokens 32K  .. end    INT4  per-group, coarser RoPE buckets
```

Attention mass concentrates in the recent window for the overwhelming majority
of prompts. Spending 32 bits on token 200,000 and 4 bits on token 1,200 is
backwards. A ladder is **more accurate than uniform INT4 at the same byte
budget**, because uniform INT4 spends its budget where attention is not.

Second-order, and still real: **RoPE bucketing**. Position encoding precision
degrades with distance; pairing coarser RoPE with coarser KV is coherent, and
lets the cold tier drop the high-frequency positional components entirely.

---

## 7. The attention data path (C16) — where latency is actually decided

Decode attention is a reduction over the entire KV. Three regimes, in order of
importance:

1. **Fully resident** — one pass over T1. Standard tiled flash-attention.
2. **Split hot/cold** — hot span resident, cold span streamed over PCIe with a
   **pipelined DMA stream**: chunk *i+1* is in flight while chunk *i* is being
   consumed. The transfer engine (C11) and the attention kernel (C16) must be
   fused into one pipeline; running them as separate phases exposes the full
   PCIe latency and is a 2-3x loss.
3. **Streaming with online softmax** — the partial-result merge must be
   numerically identical to the resident path, or invariant **I7** ("tier
   movement is bit-identical") is violated and the profiler becomes a liar.

> **I7 as originally worded is unsatisfiable, and writing the test is what
> proved it.** The invariant is testable, and the test
> (`tools/i7/i7_bit_identity.py`, exit 0) forced two corrections:
>
> 1. **A quantising codec cannot be bit-identical to full precision.** FP8
>    truncation of a bf16 value changes it by construction. So I7 is
>    **re-scoped**: *tier movement* is bit-identical — the storage-layer codec
>    that moves a page between tiers must be lossless, `decode(encode(x)) == x`
>    bit for bit. A codec that changes values is a **precision reduction**:
>    declared, priced, recorded per page, and never conflated with a move.
> 2. **Bit-identity needs a canonical reduction order, and the obvious
>    implementation does not have one.** FP addition is not associative, so a
>    single running accumulator and per-page partials merged left-to-right
>    give different bits from the same inputs. The contract now *is* the page
>    decomposition: ascending pages, ascending keys within a page, one
>    left-to-right merge of per-page partials, global fp32 max first. Both
>    paths obey it, so identity is structural rather than lucky.
>
> The negative controls matter more than the positives. `i7_bit_identity.py`
> asserts that the **flat** form and the **cold-first merge** form *diverge* —
> and both did, as did a page-local index bug in this repo's own tier-split
> implementation, which produced plausible-looking wrong output until the bits
> were compared.

---

## 8. Component flowchart

```
                        +----------------------------------+
                        |  ADMISSION (C12)                 |
                        |  assign context class A/B/C      |
                        |  price it, or refuse (I3)        |
                        +-----------------+----------------+
                                          |
                    +---------------------v---------------------+
                    |           REQUEST SCHEDULER (C1/C3)       |
                    |   prefix-homogeneous batches, KV budget  |
                    +---------------------+---------------------+
                                          |
        +---------------------------------v---------------------------------+
        |                      MODEL RESIDENCY DIRECTORY (C6)               |
        |   expert slot table: which of 128 experts live in VRAM where     |
        |   +-------------------------------------------------------------+|   |   | VRAM SLOTS (C8)  14.13 GiB at W4 -- does NOT fit with KV   | |
        |   |   residency decided ONCE at load, by budget, not per token  | |
        |   +-------------------------------------------------------------+ |
        |   | RAM  (C7)  pinned pool, cold expert bank                    | |
        |   | NVMe (C11) kernel-native packed store, cold                 | |
        |   +--------------------------------------------------------------+ |
        |   PREFETCH (C10): predict next expert set from router history   | |
        +---------------------------------+---------------------------------+
                                          |
     +------------------------------------v------------------------------------+
     |                      PER-LAYER EXECUTION                               |
     |                                                                       |
     |  +-----------+   +-----------+   +-----------+   +-----------+          |
     |  | C18       |-->| C17       |-->| C16       |-->| C14       |          |
     |  | Router    |   | Expert    |   | Attention |   | KV engine |          |
     |  | top-8     |   | GEMM      |   | + C15     |   | append    |          |
     |  +-----------+   | W4/W8/W3  |   | codec     |   +-----------+          |
     |      |            +-----------+   +-----------+                        |
     |      | miss             |               |                             |
     |      v                  v               v                             |
     |  [C19 CPU fallback]  (C6 slot)   [C13 radix index]                     |
     |  never silently drop (I3)    hit->0 B     miss->T2 cost, priced        |
     +-----------------------------------+-------------------------------------+
                                         |
                              +----------v-----------+
                              | SAMPLER (C4)        |
                              | MTP / draft (C20)   |
                              | confidence stop     |
                              +----------+-----------+
                                         |
                              +----------v-----------+
                              | C21 PROFILER        |
                              | every transfer and   |
                              | tier move attributed |
                              +----------------------+
```

### 8.1 The critical path, marked

```
NORMAL, Class A:   C18 -> C17 -> C16 -> C4        (no I/O at all)
COLD EXPERT:       C10 prefetch (async) -> C17     (must be off critical path)
COLD KV:          C16 split + C11 pipelined DMA   (priced, Class B only)
```

Two paths are allowed to be slow: cold expert load and cold KV stream. Neither
may block without being counted by C21. That is what makes "storage-bound" a
reportable state rather than a mystery (§7 of doc 04).

---

## 9. Control loop, one decode step

```
1.  C12  admit: class, context cap, step-time budget
2.  C1   take next request from the batch
3.  C13  radix lookup -> resident KV nodes for this prefix
4.  C18  route -> top-8 expert ids
5.  C6   residency check:
6.        all 8 resident?  -> proceed, 0 bytes
7.        some missing?    -> C10/C11 prefetch; if not ready, C19 or stall
8.  C16  attention over resident KV (T1 only)
9.  C14  append new KV page for this token (codec chosen by C15)
10. C17  expert GEMMs for the 8 experts
11. C4   sample
12. C21  attribute every byte and every microsecond
13. C12  update residency + eviction; emit class-transition events
```

The ordering matters: **append (9) before expert GEMM (10)** lets the two
overlap, because C17 is weight-bound and C14 is bandwidth-bound. They compete
for almost nothing.

---

## 10. Check against the tree that exists

| area | current state | verdict |
|---|---|---|
| expert GEMM (C17) | measured, correct, 5.2% of peak with a known 32% ceiling (§8.9) | **good** — and now known not to matter |
| weight pack (C17/C5) | W4 G128 settled, 0.625 instr/MAC, SNR measured (§7.6) | **good** |
| ISA capability | 14-probe matrix, EMPTY detection, arch guard (§8.7) | **good** |
| bandwidth | VRAM + PCIe both measured (§8.1) | **good** |
| `01` tier model | streams **weights** NVMe→RAM→VRAM | **wrong axis** — see §2 |
| `04` §6 KV residency | block 16/32, slabs, `KV_RECOMPUTE_RATE` | **underspecified** — no codec ladder, no class ladder |
| **context classes** | present in this doc, promoted to `00` §9.5 | **done** |
| **KV roofline** | `tools/kvroof/kv_roofline.py`, run on five real configs | **done — `00` §9.1-§9.3** |
| **codec ladder** | `04` mentions codecs, no ladder | **open** |
| **split attention over PCIe** | I7 host reference passes; no kernel yet | **open — spec only** |
| C1–C16, C18–C24 | specified, **0% implemented** | **not started** |
| routing locality (P0-7) | **MEASURED**, two independent sources, `tools/route/route_locality.py` | **done — `00` §9.6** |
| attention head config | **read from `config.json`**, never assumed | **done — the assumption was right, and is now a fact** |

### 10.1 What is fixable right now, before any engine code

Items 1, 2, 3 and 6 of the original list are **done**, and their results are in
`00-verified-facts.md` §9. What remains is item 5:

5. **Correct `01-architecture.md` §4** to separate cold-start weight transfer
   from steady-state KV transfer, and to move the plan onto the W3 operating
   point decided in `00` §9.8. The LM head (163 MB/token, 10% of the traffic) is
   already in there as §4.1; what remains is propagating W3 into the component
   sizes.

4. ~~**Add context classes to `04-memory-tiering.md` §6.**~~ **DONE** — §6.1 and
   §6.2 there now carry the explicit A/B/C ladder and the refusal path, with the
   W3 operating point and the Class B price attached.
7. ~~**Decide the weight format against the KV budget.**~~ **DONE** — the
   decision is made and recorded in `00-verified-facts.md` §9.8: **W3 g128 is the
   operating point**. It is the only pack whose FP16 context (37.8 K) fits under
   the model's own 40,960-token ceiling while keeping essentially the whole
   expert bank resident. W4 leaves zero KV budget; W2 buys capacity the model
   cannot attend to. The decision was cheap to make and would have been
   expensive to postpone.

### 10.2 What should deliberately NOT be built yet

- **NVMe on any decode path.** Class C is enough.
- **A general tiering engine.** Class A needs none. Build it only when a real
  session exceeds Class A, with that session in hand.
- **WMMA tiling.** Already demoted (§8.3). The measured kernel is within 16% of
  an intensity ceiling that tiling cannot move.
- **CPU expert fallback (C19) as a performance path.** Keep it as a correctness
  guarantee, not a scheduling option.

---

## 11. Summary

The engine is not a weight-streaming machine that also caches KV. It is a
**KV-resident machine that streams weights**, and the reason is that weights are
cacheable while KV is not.

The consequence is a single organising idea — **context class as a priced,
user-visible promise** — and a single most important number: **at roughly 4 K
of context, KV read volume overtakes weight read volume and never comes back.**

Everything the plan does after that is a consequence of those two facts.