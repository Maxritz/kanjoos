# 11 — KV-cache prior art, assessed against docs/09

**STATUS: research complete (2026-10-10).** Five papers the user supplied as a
reading list, read and assessed for what they mean to *this* engine — a
single-GPU, bounded-working-set MoE runtime whose governing reframe
([docs/09](09-kv-engine-architecture.md)) is **"KV residency is the product
feature."** Nothing here is a measurement on this machine; every number is the
paper's own claim or arithmetic derived from the paper's stated dimensions
(both labelled). Sources were read in full on their arXiv HTML; the URLs are
listed per section so a claim can be re-checked rather than trusted.

Provenance labels here: `PAPER-REPORTED` = the source states it,
`DERIVED` = computed from dimensions the source states,
`UNVERIFIED` = the source does not disclose enough to check it,
`FOR-US` = an assessment against this repo's design, not a claim from any paper.

---

## 1. MLA: DeepSeek-V2 (arXiv:2405.04434)

Read: <https://arxiv.org/html/2405.04434v5>

**Mechanism.** Keys and values are jointly compressed into a low-rank latent
`c^KV`; per-head K/V are reconstructed through learned up-projections that are
*model weights*, not per-token cache. A decoupled RoPE key is stored
separately so position rotation does not obstruct absorption (key
up-projection absorbs into Q projection; value up-projection absorbs into the
output projection).

**Arithmetic (DERIVED from the paper's dimensions):**
DeepSeek-V2 caches `d_c=512 + 64` (rope key) = **576 elements/token/layer**;
× 60 layers = 34,560 elements/token → **69,120 B/token at fp16**,
34,560 B at fp8, ~25,920 B at the deployed 6 bits/element
(`PAPER-REPORTED` that deployment averages 6 bpe). The headline "93.3%
reduction" is `PAPER-REPORTED` **versus DeepSeek 67B** — a model-to-model
deployment comparison including KV quantisation, not an MLA-only ratio.

**FOR-US.** Our reference model's cache arithmetic stands: 4 KV heads × 128 ×
K,V = 1,024 elements = **2,048 B/token/layer → 98,304 B/token** (fp16) across
48 layers, 49,152 at fp8 (docs/00 §9.2). MLA does not change that number for
an unchanged GQA checkpoint — it changes the *attention representation*.
It is a model architecture, not a page codec: our per-page codec, radix index
and NVMe tier operate on whatever per-token bytes the checkpoint declares.
Making our model MLA-shaped would mean a different model, full stop.

## 2. DeepSeek-V4.1-Flash KV (arXiv:2609.19969)

Read: <https://arxiv.org/html/2609.19969v1>

**Mechanism.** Three things stacked:
- **CSA2** — layers are assigned Full / Reindex / Reuse modes so global KV
  and indexer-K are computed once and shared forward across layers;
  layer-local sliding-window KV stays layer-local.
- **FP4 global KV** — E2M1 with one E4M3 scale per 16 channels (SWA KV stays
  fp8, "more sensitive"); cached values dequantise before attention.
- **SWA bounded replay** — instead of persisting SWA states, replay only the
  most recent window to *approximately* reconstruct them.

**Numbers.** The headline **890 B/token** is `PAPER-REPORTED` for the
*global* KV in HBM and `UNVERIFIED`: the paper does not expose per-layer
dimensions/mode counts to sum it independently. The "1/8 persistent footprint"
claim is likewise `PAPER-REPORTED`, approximate by construction (bounded
replay is a quality trade, reported negligible on their model).

**FOR-US.** CSA2 is not a serving trick that transfers — it changes *what
attention each layer sees* and assumes a sparse-global-attention architecture.
Bounded replay needs the model to have SWA; our reference model declares
`use_sliding_window: false`, so there is nothing to bound. The FP4-group
codec idea (block-scaled 4-bit per 16 channels) is the one transferable
kernel-level idea, and it lands in the same place as everything else codec:
C15, behind a declared-precision-reduction gate (docs/09 §7, I7's re-scope).

## 3. TransMLA (arXiv:2502.07864)

Read: <https://arxiv.org/html/2502.07864v4>

**Mechanism.** Post-training conversion of pretrained GQA → MLA: merge KV
heads into a joint representation, RoRoPE + FreqFold concentrate positional
information into a smaller RoPE slice, then low-rank-compress the non-positional
part. Head merging alone does not shrink the cache; the latent rank does.

**Numbers (`PAPER-REPORTED`).** Conversion tiers: **68.75%** cache reduction
(loss ~1.65% avg on six benchmarks, no retraining, Llama-2-7B), **87.5%**,
**92.97%** — the last two recover only after continued training (500M–6B
tokens in their table). The "10.6× inference speedup" is their single
Llama-2-7B 8K setup, not a general claim. Experiments are SmolLM/Llama;
**Qwen3-30B-A3B is named as a candidate, never measured**.

**FOR-US.** Conversion is a *training project*, and training/fine-tuning is an
explicit non-goal of this repo (AGENTS §10). Filed as design knowledge: if a
future model with MLA attention is adopted, the KV-tier machinery is
representation-agnostic — pages, codecs and the radix index care only about
bytes/token the file declares. No action.

## 4. PiKV (arXiv:2508.06526)

Read: <https://arxiv.org/html/2508.06526v1>, v3, repo
<https://github.com/NoakLiu/PiKV>

**Mechanism.** Expert-sharded KV for MoE serving: a KV entry keyed by
(token t, expert e) is sharded `s(t,e) = (t mod N_tok) ⊕ (e mod N_exp)`
across **GPUs** with circular buffers, page-table lookup at query time, and a
scheduler that fetches/retains query-relevant KV. Evaluation is 8×A800 (v1;
an A100 table caption contradicts it), while the released artifact protocol
is single-A100/4K-context — two different claim sets.

**Numbers (`PAPER-REPORTED`, internally inconsistent, treat as unconfirmed`).**
Headline up to 3.9× memory / 1.7× latency; 1.8–3.2× throughput across
workloads; no measured bytes/token anywhere (`UNVERIFIED` — only symbolic
equations and a "≫24 GB" intro estimate).

**FOR-US.** Multi-GPU KV placement is a non-goal (AGENTS §10), and PiKV's
"expert KV" is not our model's semantics: in a conventional MoE transformer
the router selects **FFN experts**, while attention KV belongs to
layer/token history — expert identity does not name KV pages. What *does*
transfer is one crisp design rule, now recorded:

> Keep two conditioning signals separate — **router → expert-slab prefetch**
> (weights: expert choice is exactly the predictor), and **attention → KV-page
> prefetch/retention** (queries/history). Share the bandwidth budget and the
> scheduler, never conflate the inputs.

Concretely: C10/C11's prefetch planner takes router top-k as a feature for
expert slabs only; KV pages get their own relevance score (access history +
query similarity, cf. CLO/InfiniGen below). Replacing the radix index with a
PiKV-style hash table would be a downgrade — ours is prefix-sharing
structure, not just address resolution.

## 5. Survey: KV cache optimisation strategies (arXiv:2603.20397)

Read: <https://arxiv.org/html/2603.20397v1> (extraction truncated before its
late combination table — that part is unread, stated not omitted).

**Taxonomy (the survey's own names):** Cache Eviction (H2O, SnapKV, NACL,
HASHEVICT, MorphKV, RocketKV, KVzip, Ada-KV) · Cache Compression (KIVI,
KVQuant, MiniCache, PALU) · Hybrid memory solution (PagedAttention, InfiniGen,
LayerKV, INF2, KVPR, Oneiros, CLO) · New Attention Calculation (linear
variants) · Combination Methods (FlexGen, Q-Hitter, ShadowKV, TailorKV).

**What it does NOT contain:** no `q4_0`/`q8_0` by name (KIVI's 2-bit result
is not a result for llama.cpp codecs); no radix/prefix-reuse class by name
(only PagedAttention's shared blocks); **no MoE-specific KV method** in the
retrieved text.

**FOR-US — already in our design:** paged KV with per-page codec (C14/C15),
prefix reuse (C13 radix), NVMe cold tier (docs/09's whole point). **Worth
adopting (mechanisms we lack), in priority order:**
1. **Predictive prefetch with transfer/compute overlap** — InfiniGen
   (CPU-resident KV, predicted prefetch overlapped with GPU work), KVPR
   (overlap recomputation with transfer), CLO (query-similarity reuse with
   critical heads resident). This is the operating layer of the tier we are
   building; docs/09 §2's pipelined DMA stream is the same idea on our side.
2. **Workload-aware residency/eviction accounting expected reuse against
   tier latency** — admission decisions priced in NVMe µs, not LRU alone.
3. **KIVI's asymmetric treatment (per-channel keys, per-token values) as a
   C15 candidate layout** — it is a *quantisation layout* decision, so it
   sits behind the same declared-precision-reduction gate as every codec.
**Rejected:** eviction-as-context-dropping (H2O/StreamingLLM class) as a
*product* strategy — permanently discarding context conflicts with "KV
residency is the product feature"; our tiers keep the full state, they move
it. Eviction only ever applies to cacheable derived copies, declared as such.

---

## 6. Summary table

| paper | mechanism | headline number | transfers to us? |
|---|---|---|---|
| DeepSeek-V2 MLA | low-rank KV latent + absorb | 93.3% vs DeepSeek 67B (`PAPER-REPORTED`, model-vs-model) | no — different architecture; our 98,304 B/token stands for qwen3moe |
| V4.1-Flash | CSA2 sharing + FP4 + SWA bounded replay | 890 B/token global KV (`PAPER-REPORTED`, `UNVERIFIED` sum) | FP4-group codec idea → C15; rest needs sparse/SWA architecture we don't have |
| TransMLA | post-training GQA→MLA conversion | 68.75–92.97% reduction (`PAPER-REPORTED`, Llama/SmolLM) | no action — conversion is training, a non-goal |
| PiKV | expert-sharded KV across GPUs | 3.9×/1.7× (`PAPER-REPORTED`, inconsistent setups) | one rule: router-conditioned ≠ attention-conditioned scheduling; keep signals separate |
| survey 2603.20397 | taxonomy of eviction/compression/hybrid | per-method ratios are the cited papers' own | prefetch+overlap and reuse-priced residency are our biggest genuine gaps |

**Measurement discipline (the reminder that applies to all of it):** every KV
performance claim in this repo — codecs (C15), paging (C14), attention (C16),
prefetch — is measured through the **C21 per-op profiler**: `KNJ_PROFILE_OP`
markers decompose execution into component rows
(`component,ops,dev_ns,idle_ns,host_ns,…`, floor-subtracted, host and device
clocks separated — see [docs/06](06-profiling.md) and
`records/c21-baseline-2026-10-08/*/profile.csv`), and
`tools/c21/profile_diff.py` gates regressions. A qwen3moe path already has
per-op markers on every leaf op; the qwen35 path has coarse markers today and
acquires per-op rows as part of the trunk work. No optimisation from any
paper above lands without a before/after profile through that gate.
