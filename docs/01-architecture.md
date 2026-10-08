# 01 — System architecture

## 1. The one sentence

A MoE model served from disk on a consumer GPU is a **transfer-scheduling
problem wearing a kernel problem's clothes**. The engine wins by (1) shrinking
expert bytes with a kernel-native packed format, (2) predicting expert demand
one layer ahead so reads overlap compute, (3) keeping the hot set in VRAM and
the warm set in RAM so NVMe is only ever a capacity tier, and (4) spending every
byte that frees up on either lower-bit KV cache or speculative decoding.

Everything below is in service of one of those four.

## 2. Tier model

```
                        CANONICAL CHECKPOINT (NVMe, read-only, never on the miss path)
                                     |
                          [C5] expert compiler  (offline)
                                     |
                     kernel-native packed expert store  (NVMe, COLD)
                                     |
                             async, coalesced I/O [C11]
                                     v
                        RAM WARM tier  [C7]     bounded pinned DMA pool [C7]
                                     |
                             async H2D, DMA      [C11]
                                     v
                        VRAM HOT tier [C8]  expert slots + paged KV + workspace
                                     |
                          kernels [C16] [C17] [C14]
                                     v
                                  logits
```

Three tiers, one residency table [C9], one radix index [C13] over KV, one
directory [C6] over experts. RAM is the *default* overflow tier; NVMe is the
*last* tier and is expected to be cold for most sessions.

> **PARTLY SUPERSEDED by [09-kv-engine-architecture](09-kv-engine-architecture.md).**
> The model above is correct for expert **weights** and is the wrong axis for
> KV. Weight reads can hit (route locality); KV reads structurally cannot --
> every token re-reads the whole prefix. At roughly **4 K of context, KV read
> volume overtakes weight read volume per token (at ~9.6 K tokens) and never
> comes back.** Section
> 4 below sizes cold start; steady-state decode is sized in doc 09 section 2.

### The rule that keeps it honest

> NVMe is not VRAM with worse latency. It is storage. The engine's job is to
> never be in a position where it needs a token from NVMe on the critical path.
> When it is, that is a measurable failure (STALL_TIME_MS), not a strategy.

### TOC completeness

The numbered subsections of `01` cover §3, §4, §5, §6, §7, §8 only. The
outline `00-verified-facts.md` promises §1, §2, §3, §4, §5, §6, §7, §8,
§9. `docs/01-architecture.md` is missing a proper §1 (title / one-sentence /
model summary), §2 (tier model diagram), and §9 (a pulled-forward label for the
W3 operating point). `09-kv-engine-architecture.md` now carries §2's diagram
and §9's packing decision, but a reader expecting the `01` outline finds them
in `09`. The sections that do exist in `01` are correct; the gap is one of
structure, not substance.

> **Status: minor structural gap.** Substance is present across `01` + `09`;
> missing is only the header scaffolding and the pulled-forward section. Low
> priority relative to implementation, but visible in a first sprint review.

### Numeric-claim audit (do not trust the tables until this passes)

The numbers in `01` §3–§4 and the derived tables are machine-audit-able:

```
python tools/kvroof/kv_roofline.py models/qwen3-30b-a3b/config.json --check
python tools/check_docs.py
```

- `kv_roofline.py --check` audits the claims in `09` and the derived geometry
  in `01` §3–§4 against the model’s own `config.json`. Exit 0 = doc agrees;
  exit 5 = doc contradicts the config.
- `check_docs.py` re-runs the roofline tool, the I7 test, and the routing
  artifacts, then asserts that the figures quoted in `00` §9 (which `01` pulls
  forward) still match what the tools produce. Exit 0 = every documented figure
  matches; exit 1 = at least one documented figure is stale; exit 3 = a tool
  failed to run, so nothing was checked (that is a failure, not a pass).

**Before quoting any table in `01` as current, run both and confirm they pass.**
A table can be structurally correct and numerically stale at the same time;
these two commands are the thing that catches that.

> NVMe is not VRAM with worse latency. It is storage. The engine's job is to
> never be in a position where it needs a token from NVMe on the critical path.
> When it is, that is a measurable failure (STALL_TIME_MS), not a strategy.

## 3. Budget arithmetic

### 3.1 VRAM

```
VRAM_CEILING     = profile_ceiling(detected_vram)      # C22, reBAR-aware
MODEL_DEVICE     = resident weights (trunk, attention, embeddings, router, head)
WORKSPACE        = activations, LDS double-buffering, expert staging, logits
HOT_KV           = VRAM_CEILING - MODEL_DEVICE - WORKSPACE - SAFETY
EXPERT_SLOTS     = f(HOT_KV)                            # C8 policy, see below
```

Profile ceilings by detected VRAM — note 8 GiB is a first-class row, it is the
most common consumer RDNA4 board and it was missing from the prior spec's table:

| VRAM | ceiling | why |
|---|---|---|
| 6 GiB | 5.25 GiB | driver + runtime + fragmentation reserve |
| 8 GiB | 7.00 GiB | same reserve, absolute |
| 12 GiB | 10.50 GiB | same reserve, absolute |
| 16 GiB | 14.00 GiB | same reserve, absolute |

The reserve is not a guess: it is measured at startup by allocating and freeing
a set of buffers at the sizes the engine will actually use, and taking the
high-water mark. `VRAM_USABLE` is what `hipMemGetInfo` reports after that, and
the engine refuses to start if it is under 3.5 GiB.

### 3.2 RAM

```
WARM_KV_LIMIT  = min(0.65 * phys_ram, phys_ram - 8 GiB)     # KV tier
WARM_EXPERT    = phys_ram - OS_HEADROOM - WARM_KV - PINNED_POOL - SAFETY
PINNED_POOL    = 2 GiB (24/32), 3 GiB (48), 6 GiB (96)      # C7, bounded
```

Dynamic shrink is mandatory, not optional: the engine polls platform-available
memory each scheduling tick and shrinks WARM before it lets the OS swap. Order of
sacredness when memory is tight:

```
PINNED pool  >  active request's KV  >  active request's experts  >  WARM  >  COLD
```

### 3.3 NVMe

```
NVME_QUOTA = min(35% of SSD capacity, 2 TiB), configurable, with >= 20% of the
             device left outside the quota
```

Writes are admission-controlled [C12]: a page reaches NVMe only when
`reuse_probability * recompute_cost > write_cost + read_restore_cost`. The
initial implementation persists shared prefixes and suspended sessions after a
TTL, and nothing else.

## 4. The transfer budget — the number the whole plan is sized against

Qwen3-30B-A3B (48 layers, 128 experts/layer, top-8, hidden 2048, moe_inter 768):

```
per expert  = 3 matrices x 2048 x 768 = 4,718,592 params
expert bank = 4,718,592 x 128 x 48     = 28.99 B params
  BF16                54.0 GiB
  W8  (G=128)         27.4 GiB
  W6  (G=128)         20.7 GiB
  W4  (G=128)         14.1 GiB   <- corrected; see below
active/token = 4,718,592 x 8 x 48      =  1.81 B params
```

The W4 row was 13.8 GiB. The pack costs 0.5234 B/weight, not 0.5 — `gemm_w4.hip`
adds 3 B of metadata (fp16 scale + uint8 zero-point) per 128-weight group — so
28.99 B params is **14.13 GiB**, not 13.8. That 0.33 GiB is not cosmetic: it is
the difference between a KV budget of -0.06 GiB and one that rounds to zero.
`tools/kvroof/kv_roofline.py` derives it and `--check` now flags the old value.

So at W4, **0.95 GB of expert weight per generated token** if there is zero reuse
across tokens. Against the measured ~4 GB/s NVMe and an assumed ~20 GB/s PCIe
4.0 pinned H2D:

| tier | bandwidth | tokens/s ceiling, zero reuse |
|---|---|---|
| NVMe | 4 GB/s | ~4.3 |
| RAM -> VRAM (pinned) | ~20 GB/s (ASSUMPTION, P0-1) | ~21 |
| VRAM-resident slot | 0 bytes | unbounded by I/O |

That table is the plan's justification for its component ordering. Any claim of
"60 tok/s for a 30B MoE on a 12 GiB card" has to come from expert-slot hit rate
and batching, not from a faster disk.

### 4.1 Two corrections this section was living with

Both were found by `tools/kvroof/kv_roofline.py`, which reads the model's
`config.json` instead of assuming its shape. Full working in
`00-verified-facts.md` §9.3.

1. **This is a cold-start budget, not a steady-state one.** The number above is
   "no expert is ever reused across tokens", which is a property of the *load*
   path. What a decode step actually touches every token is larger:

   | component | MB/token |
   |---|---|
   | routed experts (top-8 x 48 layers, W4) | 948 |
   | attention q/k/v/o, read in full per layer | 474 |
   | lm_head (151,936 x 2048), read in full per step | 163 |
   | **total** | **1,586** |

   The LM head is 10% of per-token weight traffic and was missing here
   entirely. It cannot be amortised by routing or batching, because every token
   needs its own logits.

2. **The KV side of the budget is the one that gets worse with use.** Experts
   are cacheable; KV is not. The two numbers must be budgeted separately or the
   second hides behind the first. At the measured crossover, KV overtakes expert
   weight traffic at ~9.6 K tokens of FP16 context and never comes back.

### 4.2 The weight format is now a capacity decision, not a bandwidth one

P0-7 (`00-verified-facts.md` §9.6) measured routing locality from two
independent sources and found it weak: 59-81% of a token's experts are new,
the hit-rate curve is close to linear in slot count, and 70% of the bank buys
a 95% hit rate. The consequence is that **context cannot be bought by evicting
experts** — the miss cost (3-9 ms/token) exceeds the capacity gain. The
remaining lever is the weight format, and it is now sized against the KV
budget it leaves, not against PCIe bandwidth.

Same 3 B/group metadata rule as `tools/bench/gemm_w4.hip` (fp16 scale +
uint8 zero-point), whatever the bit width (`kvroof.pack_bytes_per_weight`).
16 GiB VRAM, 0.75 GiB workspace:

| pack | B/weight | expert bank | resident | KV budget | FP16 context | FP8 context | experts resident |
|---|---|---|---|---|---|---|---|
| W4 g128 | 0.5234 | 14.13 GiB | 15.31 GiB | **0** | — | — | 127 / 128 |
| **W3 g128** | **0.3984** | **10.76 GiB** | **11.79 GiB** | **3.46 GiB** | **37.8 K** | **75.6 K** | **126 / 128** |
| W2 g128 | 0.2734 | 7.38 GiB | 8.27 GiB | 6.98 GiB | 76.2 K | 152.4 K | 123 / 128 |
| W4 g64 | 0.5469 | 14.77 GiB | 15.97 GiB | 0 | — | — | 122 / 128 |
| W8 g128 | 1.0234 | 27.63 GiB | 29.37 GiB | 0 | — | — | 65 / 128 |

**W3 g128 is the operating point.** It is the only row whose FP16 context fits
under the model's own 40,960-token ceiling while keeping essentially the whole
expert bank resident. W2's extra capacity is largely unusable — the model
cannot attend that far. W4 leaves zero KV budget. The decision was cheap to
make and would have been expensive to postpone.

This is a capacity decision, not a bandwidth one. What it costs to buy 1 GiB
of KV:

| what it costs to buy 1 GiB of KV | |
|---|---|
| W4 → W3 repack | **free** in time (0.2 ms/token extra misses), frees 3.46 GiB |
| evicting 6 experts/layer | 6-9 ms/token, for less capacity than the repack |

Note what is *not* on this table: any codec that changes values. Per I7
(`00` §9.7) a quantising codec is a declared precision reduction, and its
capacity is orthogonal to this decision — it multiplies whatever budget the
pack leaves, behind a quality gate.

The full derivation, including the hit-rate and miss-cost columns that depend
on the measured P0-7 curve rather than the config, is in
`tools/kvroof/kv_roofline.py` section K and is audited by `--check`.

## 5. Invariants

These are the statements that must hold in every build. Each is independently
testable and each has a test named in `08-roadmap.md`.

* **I1 — No expert is ever lost.** Every routed expert is recoverable from RAM or
  NVMe. A cache miss changes latency, never the result.
* **I2 — No dequantisation on the miss path.** The bytes that move are the bytes
  the GEMM consumes. `NVMe → RAM → VRAM → GEMM` in a kernel-native format.
* **I3 — The router is authoritative.** Cache pressure never silently changes
  the selected expert set. If an expert cannot be placed, the request stalls or
  falls back to CPU — it does not drop.
* **I4 — No synchronous NVMe read on the critical path.** A kernel launch never
  blocks on storage. Ever. Enforced by a debug assert plus the profiler.
* **I5 — KV cache identity includes the model fingerprint**, arch, dtype, layout,
  RoPE config, tokenizer, and attention implementation version. A cache page from
  a different build is not readable.
* **I6 — Speculation is lossless.** Accepted output is bit-identical to
  non-speculative decoding under the same seed.
* **I7 — Tier movement is invisible to the result.** Evicting and reloading a KV
  page or expert reproduces the same numbers, bit for bit.

## 6. Control flow, one token

```
scheduler picks a request
  |
  |-- C13 radix lookup: longest cached prefix -> reusable KV block list
  |     miss -> prefill the uncovered suffix (chunked), write new KV blocks
  |
  |-- loop over layers:
  |     |-- C18 router -> top-k expert ids for this layer
  |     |-- C10 predictor -> next layer's likely experts (+horizon)
  |     |-- C9 residency: which of those are in VRAM?
  |     |     slot hit    -> nothing
  |     |     RAM hit     -> C11 H2D enqueue (async, overlaps compute)
  |     |     NVMe hit    -> C11 NVMe read enqueue -> RAM -> H2D
  |     |     NVMe miss   -> C6 read-through from the canonical store
  |     |-- C16 attention over paged KV (HOT pages only; RAM pages must already
  |     |     have landed — the planner guarantees this or takes the CPU path)
  |     |-- C17 grouped expert GEMM over the slot-resident set
  |     +-- C20 speculative step if enabled (MTP head or drafter)
  |
  |-- C21 emit token; C22 roll budgets forward; C11 poll completions
  +-- on session end: C9 demote, C12 decide what reaches NVMe
```

The critical line is the third: **attention and expert GEMM only ever see VRAM**.
Everything else is a prefetcher's job. A design that lets a kernel touch a RAM
page is a design that pays PCIe latency inside the critical path.

## 7. What is different between the two targets

| | gfx1031 (RDNA2) | gfx1201 (RDNA4) |
|---|---|---|
| matrix path | none (measured: RDNA2 has no matrix units) | **WMMA, measured working** (`v_wmma_f32_16x16x16_f16`) |
| fp16 throughput basis | `v_pk_fma_f16`, 2 lanes/op | WMMA 16×16×16, **4 VGPRs** per f16 operand (measured; see 03-kernels.md) |
| FP8 | not available | native, 2× FP16 |
| int8 dot | `v_dot4_i32_i8` as inline asm; the `sdot4` builtin also works and lowers to native `v_dot4c_i32_i8` | `v_dot4_i32_i8` as inline asm (the `sdot4` builtin is refused) |
| KV codec default | FP16/BF16 | FP8 E4M3 |
| hipBLASLt | absent | present |
| default VRAM profile | 12 GiB (the 6700 XT box) | 8 / 12 / 16 GiB |
| tier emphasis | RAM-heavy, cold-tolerant | VRAM-hot-heavy, aggressive prefetch |

The engine ships one architecture and two kernel backends behind C2. Policy
parameters differ per arch; the component graph does not.

## 8. Deliberate non-goals

1. Training or fine-tuning of any kind.
2. Multiple GPUs. Single device, plus host memory. (SPLASH's decoupling idea is
   recorded for later, not built.)
3. Quantising the attention trunk or router by default. Experts are the target;
   the trunk stays BF16 until measurement says otherwise.
4. Expert-only BF16→ternary in the production path. It is an experiment with a
   quality gate, per the expert-compiler spec.
5. Making NVMe look like VRAM. Ever.