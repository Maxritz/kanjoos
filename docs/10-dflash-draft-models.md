# 10 — DFlash / DFlash2 / DSpark draft models, and the `qwen35` MTP head

`ai-coder/c20-speculation.md` names the families this engine must load — **"one
loader for DFlash / DFlash2 / DSpark, with an arch compatibility check"** — and
never states what is inside one. This document is the missing input: the metadata
and tensor inventory of the four file shapes actually on this machine, read from
the containers, with the command that read them.

**STATUS: reconnaissance complete (2026-10-08); the `qwen35` front end is built,
the three drafters are not.** The engine's forward pass in `src/model/model.cpp`
still implements `qwen3moe` and only `qwen3moe`, so every drafter file below is
refused — but a `qwen35` container is now **read, bound, tokenized and computed**
far enough to prove the front end: see §7 and
`records/qwen35-probe-2026-10-08/README.md` for the 45 reference-verified vectors.
The trunk (48 SSM + 17 attention layers) and the head are refused by name, with
`kanjoos-run` exiting 3 — understood, deliberately not run. Two of those 65 blocks
are now computed and verified on their own (§6.2, §6.3); the **trunk** that would
compose them, and the head, are not. This document exists
so the C20 loader is written against a measured layout instead of a name, and so
the gap is a sized task rather than an unknown.

Provenance of every number here: **MEASURED**, by
`python tools/ggufmeta/gguf_meta.py <file.gguf>` — a read-only container reader
that prints the header, the metadata KV, and the tensor inventory. It is the
"tool output that generated the number" required by AGENTS.md §8, and it is in the
repo so the numbers can be re-read rather than trusted.

---

## 1. What the family is

The three drafters are **block-diffusion speculative drafts** for a Qwen3.8-27B
target, published by `z-lab` / Inco AI (`dflash2`, `dflash`, `dspark` tags:
`speculative-decoding`, `block-diffusion`, `draft-model`, `sglang`). They are not
small language models in the usual sense:

* They are **5-block** networks of width 5120, **non-causal**
  (`dflash.attention.causal = false`) with a sliding window of 2048.
* They are **conditioned on the target model's intermediate hidden states**:
  `dflash.target_layers` names five target layers, and the drafter's `fc` tensor
  maps `5 × 5120 → 5120`. The drafter is an *annex* to a specific target, not a
  standalone model.
* They predict a **block** of tokens at once (`dflash.block_size`: 8, 16 or 7
  depending on the variant), which is the "diffusion" half: the block is refined
  in place rather than produced left to right.
* They carry **no `token_embd` and no `output.weight`**. They reuse the target's
  embedding and head — which is why a drafter is a few hundred MB beside a 27B
  target, and why it cannot be benchmarked without the target.

Three variants share `general.architecture = 'dflash'` and differ in which extra
head they carry:

| variant | extra machinery | file measured (bytes, on disk) |
|---|---|---|
| DFlash (bootstrap) | none: `fc` + per-block attention/FFN | `Qwen3.8-27B-DFlash-bootstrap-Q8_0.gguf` — **1,849,482,752 B** |
| DFlash2 | two **convs** per block + a **selector** (predecessor/successor vocab heads) | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, `…-Q2_K.gguf` (Q2_K = **705,431,072 B**) |
| DSpark | a **confidence head** (`conf_proj`) + a **Markov** pair (`markov_w1/w2`) | `Qwen3.8-27B-DSpark-Q8_0.gguf` — **1,455,376,576 B** |

---

## 2. DFlash2 — the fullest variant (81 tensors, Q2_K, 705,431,072 B)

Metadata, verbatim:

```
general.architecture                dflash
general.name                        Qwen3.8-27B-DFlash2
general.size_label                  1.9B
dflash.block_count                  5
dflash.embedding_length             5120
dflash.feed_forward_length          17408
dflash.attention.head_count         32
dflash.attention.head_count_kv      8
dflash.attention.key_length         128
dflash.attention.value_length       128
dflash.attention.causal             false
dflash.attention.sliding_window     2048
dflash.attention.sliding_window_pattern  [true, true, true, true, true]
dflash.attention.layer_norm_rms_epsilon  1e-06
dflash.rope.freq_base               10000000.0
dflash.block_size                   8
dflash.conv_kernel_size             2
dflash.conv_group_size              16
dflash.selector_rank                256
dflash.selector_top_k               16
dflash.target_layers                [6, 20, 34, 48, 62]
```

Tensors (dims are `[ne0, ne1]` = in, out for this engine's reader; `x5` means one
per block):

| tensor | dims | type |
|---|---|---|
| `blk.N.attn_norm.weight` | [5120] | F32 ×5 |
| `blk.N.attn_q.weight` | [5120, 4096] | Q2_K ×5 |
| `blk.N.attn_k.weight` | [5120, 1024] | Q2_K ×5 |
| `blk.N.attn_v.weight` | [5120, 1024] | Q4_K ×5 |
| `blk.N.attn_q_norm.weight` | [128] | F32 ×5 |
| `blk.N.attn_k_norm.weight` | [128] | F32 ×5 |
| `blk.N.attn_output.weight` | [4096, 5120] | Q3_K ×5 |
| `blk.N.attn_conv_proj.weight` | [5120, 1280] | Q2_K ×5 |
| `blk.N.attn_conv_base` | [5120, 2, 2] | F32 ×5 |
| `blk.N.ffn_norm.weight` | [5120] | F32 ×5 |
| `blk.N.ffn_gate.weight` | [5120, 17408] | Q2_K ×5 |
| `blk.N.ffn_up.weight` | [5120, 17408] | Q2_K ×5 |
| `blk.N.ffn_down.weight` | [17408, 5120] | Q3_K ×5 |
| `blk.N.ffn_conv_proj.weight` | [5120, 1280] | Q2_K ×5 |
| `blk.N.ffn_conv_base` | [5120, 2, 2] | F32 ×5 |
| `enc.output_norm.weight` | [5120] | F32 |
| `fc.weight` | [25600, 5120] | Q2_K |
| `output_norm.weight` | [5120] | F32 |
| `selector_hidden.weight` | [5120, 256] | Q2_K |
| `selector_predecessor.weight` | [256, 248320] | Q2_K |
| `selector_successor.weight` | [256, 248320] | Q2_K |

Facts that matter for a loader:

* **`25600 = 5 × 5120`** and `fc.weight` is `[25600, 5120]`: the five target-layer
  hidden states are concatenated on the **input** side and projected down to one
  5120 vector. A loader that assumes `fc` is an expansion (`5120 → 25600`) reads
  the wrong dimension.
* **`1280 = 1024 + 256`**: a conv takes the 1024-wide attention (or FFN) value
  together with the 256-wide selector vector. `conv_kernel_size = 2` and
  `conv_group_size = 16` mean it is a grouped/depthwise convolution over that
  concatenation, with `conv_base` as its bias-like base term.
* `selector_predecessor` / `selector_successor` are `[256, 248320]`: **248320 is
  the vocabulary** (the same count as `tokenizer.ggml.tokens`), reached through a
  rank-256 bottleneck. Two heads, not one.
* The tokenizer is `gpt2` BPE with `tokenizer.ggml.pre = 'qwen35'` and a **248320**
  vocabulary — 1.63× the 151936 of the reference model in `models/`.

## 3. DFlash (bootstrap) — 58 tensors, Q8_0

Same 5-block skeleton with `block_size = 16`, `target_layers = [2, 17, 32, 47, 62]`,
`sliding_window_pattern = [true, true, true, true, false]` (the fifth block is
**not** windowed), and **no conv and no selector tensors at all**: 11 per-block
tensors ×5 = 55, plus `enc.output_norm`, `fc.weight` `[25600, 5120]`,
`output_norm` — 58 exactly.

This is the minimal member of the family and the natural first target: no conv,
no rank bottleneck, no vocabulary-sized head. A loader written for it is a strict
subset of what DFlash2 needs, except for the block size.

## 4. DSpark — 62 tensors, Q8_0, `general.name = 'Source'`

Shares the DFlash skeleton with `block_size = 7`,
`target_layers = [5, 17, 29, 41, 53]`, `feed_forward_length = 10240`, 40 heads /
8 KV heads, and **Yarn rope scaling** (`factor 32.0`, `original_context_length 8192`,
`beta_fast 32.0`, `beta_slow 1.0`). It adds the two heads C20 already asks for:

| tensor | dims | type | reading |
|---|---|---|---|
| `conf_proj.weight` | [5376] | F16 | 1-D, length `5120 + 256`: a **confidence** projection over [hidden, selector] |
| `conf_proj.bias` | [1] | F32 | one scalar bias |
| `markov_w1.weight` | [256, 248320] | Q8_0 | rank-256 → vocab |
| `markov_w2.weight` | [256, 248320] | Q8_0 | rank-256 → vocab |

`conf_proj` is the head C20 requires to be **wired into C10** ("drafting stops
early when the next verification pass would stall on cold experts"). It is a
scalar per position, which is what makes it wireable: one number, thresholded.

## 5. The `qwen35` target with a **merged** MTP head

The variants the user described as "some have it merged into the GGUF" are a
different thing from the drafters above: the target model itself, carrying a
**multi-token-prediction head** in the same container.

`Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (**12,120,016,960 B**, **866 tensors**,
53 KV entries, `general.architecture = 'qwen35'`) is a **hybrid trunk**:

| group | count | tensors |
|---|---|---|
| SSM blocks | 48 | `attn_qkv [5120,10240]`, `attn_gate [5120,6144]`, `ssm_a [48]`, `ssm_alpha [5120,48]`, `ssm_beta [5120,48]`, `ssm_conv1d [4,10240]`, `ssm_dt.bias [48]`, `ssm_norm [128]`, `ssm_out [6144,5120]` |
| attention blocks | 17 | `attn_q [5120,12288]`, `attn_k [5120,1024]`, `attn_v [5120,1024]`, `attn_q_norm [256]`, `attn_k_norm [256]`, `attn_output [6144,5120]` |
| both | 65 | `attn_norm`, `post_attention_norm`, `ffn_gate [5120,17408]`, `ffn_up`, `ffn_down [17408,5120]` |
| MTP (merged) | 4 | `nextn.eh_proj [10240,5120]`, `nextn.enorm [5120]`, `nextn.hnorm [5120]`, `nextn.shared_head_norm [5120]` |
| embeddings/head | 3 | `token_embd [5120,248320]` IQ2_S, `output.weight [5120,248320]` Q4_K, `output_norm [5120]` |

Two consequences, and they are the whole reason this section exists:

1. **`qwen35` is not `qwen3moe`.** A hybrid SSM + attention trunk with gated
   attention, QK-norm width 256, and untied embeddings is a different forward pass,
   not a configuration of the existing one. Nothing about the MTP head can be
   exercised until that trunk runs.
2. **The MTP head is free where C20 says it is.** `nextn.*` is 4 tensors and no
   separate embedding or head — it rides the target's own `token_embd` and
   `output.weight`, which is exactly the "already resident, zero extra load I/O"
   argument in `ai-coder/c20-speculation.md`. But "merged into the container" and
   "implemented" are not the same claim: the head still needs its `eh_proj`
   fusion of `[hidden, embedding]` (`10240 = 2 × 5120`) and its norms before it
   predicts anything.

Other `qwen35` files on this machine carry the same trunk **without** `nextn.*`
(e.g. `ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf`, **17,442,400,352 B**, 866 tensors,
50 KV entries), so the
merged-MTP and no-MTP shapes are distinguished by the presence of four tensor
names, not by the architecture string.

## 6. What the engine does with these files today — refused, with the reason

Nothing in §1–§5 is implemented. What was built on 2026-10-08 is the `qwen35`
**front end** (§6.1), one **GatedDeltaNet block** (§6.2), one **gated
full-attention block** (§6.3), and the tokenizer's cross-check (§6.4) — one layer
of each kind, verified per layer against an oracle, plus a tokenizer that is now
evidence rather than an assumption. No `qwen35` trunk and no `dflash` loader exist,
so the refusals below are current. Measured refusals, verbatim:

| file | refusal |
|---|---|
| `Qwen3.8-27B-DFlash2-Q2_K.gguf`, `…-DFlash2-Q4_K_M`, `…-DSpark-Q8_0` | `REFUSED: architecture 'dflash' is a *draft* model, not a language model…` — exit 3, naming what a drafter needs (a target, and the C20 loader) instead of the generic architecture message |
| `Qwen3.8-9B-Q6_K.gguf`, `ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf` | `RESULT: qwen35 front end verified …; the trunk and the head are REFUSED.` — exit 3, with §6.1's numbers |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | container + tokenizer verified, then `PROBE STOPPED at layer 0: qwen35: token_embd.weight is IQ2_S, which this engine cannot decode` — exit 3; 363 tensors across 8 types have no decoder here |

**Consequence for benchmarking:** no Qwen 3.8 file on this machine can produce a
number, because no `qwen35` trunk exists yet and three of the five files also
carry quantisation types with no decoder. The refusal is the measurement. The one
model in this checkout that runs end-to-end is
`Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf` (arch `qwen3moe`, 964,653,184 B, 28 layers,
hidden 1024, 3072 FF, 4 experts, top-2), which is why the baseline in
`records/c21-baseline-2026-10-08/` is on that model.

### 6.1 What the engine does with a `qwen35` file now (`--qwen35-ref`, exit 3)

`kanjoos-run` sniffs `general.architecture` before it opens a forward pass. A
`qwen35` container goes to `src/model/qwen35.cpp` instead of dying on the
architecture gate, and that probe reports, in order: the container, the geometry,
the tokenizer, the tensor binding, the quantisation coverage, one layer's input
projections compared against an independent oracle, and finally what is still
missing. Measured on `ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf` (17.4 GB):

| section | result |
|---|---|
| geometry | 65 layers = **48 SSM + 17 attention**, hidden 5120, ff 17408, vocab 248320, 24 q / 4 kv heads × head_dim 256, rope base 1e7 over 64 of 256 dims, ssm inner 6144 / state 128 / groups 16 / dt_rank 48 / conv 4 → conv_dim 10240 |
| layer kinds | `SSSASSSA…SSSSAA`, taken from `attention.recurrent_layers` (exact, per layer) |
| tokenizer | vocab 248320, **9/9 round-trips exact** incl. CJK, emoji, newlines, tab, empty string |
| binding | **866/866 tensors** present with the implied shape, incl. 4 `nextn.*` on `blk.64` |
| quantisation | F32 456, Q4_K 257, Q6_K 79, Q8_0 64, Q4_0 8, Q5_K 2 — all decodable |
| vectors | **45/45 verified element by element** vs `tools/ref_qwen35.py` (5 tokens × 2 layer kinds: embeds bit-exact, norm bit-exact, qkv/gate/q/k/v within 2e-5 abs) |

The oracle's own decoders are cross-checked against gguf-py before it writes
anything, and the C++ dequantisers are cross-checked against gguf-py on real
tensor bytes (`tools/dequant_probe.cpp` + `tools/dequant_crosscheck.py`), so the
agreement is between two implementations that share no code.

### 6.2 The recurrent layer, end to end (`--qwen35-recurrent`, still exit 3)

Added 2026-10-08, same day. `kanjoos-run --qwen35-recurrent` computes one
GatedDeltaNet block on the host — the causal depthwise conv1d over the fused
`[q|k|v]` channels with its state carried across tokens, the per-group delta rule
over a 48 × 128 × 128 state, and the alpha/beta/ssm_a gating — and compares
**12 named vectors** against `tools/ref_qwen35.py --recurrent`:
`layer_x`, `layer_qkv`, `layer_conv_raw`, `layer_conv_silu`, `layer_q_l2`,
`layer_k_l2`, `layer_beta`, `layer_g`, `layer_o`, `layer_state`, `layer_gated`,
`layer_out`.

| measured | value |
|---|---|
| vectors, `blk.0` (recurrent) | **12/12 PASS**, max abs 2.289e-05, worst rel-RMSE 7.2e-07 |
| vectors, `blk.4` (recurrent) | **12/12 PASS**, max abs 8.774e-05, worst rel-RMSE 2.4e-06 |
| zeros / unwritten / mismatching elements | 62 exact zeros (all present in the oracle too), **0 unwritten, 0 mismatching** |
| tolerance | rtol 2e-3 of the reference RMS — the worst rel-RMSE is ~800× inside it |
| state carry | 5 tokens in one call vs 5 separate calls: `out` and `state` **bit-identical** |
| causality | the first 2 tokens from a fresh state vs the prefix of the whole run: **bit-identical** |
| `exp(g)` range | `[0.000474, 0.999999]` at `blk.0`; inside (0, 1], which is the reading check for the stored `-exp(A_log)` |
| cost | 5 token steps of one layer in **353 ms** (host, single-threaded projections) |

A mutation test proves the comparison can fail: dropping `beta` from the delta
rule leaves the eight upstream vectors PASSing and fails exactly the four
downstream ones with 23k–554k mismatching elements
(`records/qwen35-probe-2026-10-08/mutation-drop-beta.log`). The mutation was not
committed; the source was restored byte-identical and re-verified.

Because the oracle and the engine were written from the same two references, the
arithmetic was additionally checked against upstream code read on this machine,
which is what closes a *shared* misreading:

* `src/models/qwen35.cpp` (llama.cpp) 378–447 — the conv runs on the fused tensor
  and the `q|k|v` split is a `ggml_view` at offsets `0`,
  `head_k_dim*num_k_heads`, `2*head_k_dim*num_k_heads`: the GGUF layout is plain
  `[q|k|v]`, not HF's per-group interleave.
* `ggml_compute_forward_ssm_conv_f32` + `build_conv_state` — tap 0 multiplies the
  **oldest** sample and the state is the last `kernel-1` inputs, oldest first.
* `ggml_compute_forward_gated_delta_net_one_chunk` — the decay, `kv_mem`, `delta`,
  outer-product update and output order, with the state stored transposed as
  `s_out[j*S_v + i] = S[i][j]`.
* HF 5.15.1 `Qwen3NextGatedDeltaNet.forward` and `torch_recurrent_gated_delta_rule`
  — `g = -A_log.exp() * softplus(a + dt_bias)`, and `l2norm(x) = x * rsqrt(sum + eps)`
  which is the formula the engine uses (ggml floors the denominator instead; the
  two differ by ~1e-6 relative at these magnitudes).

What is still missing around the recurrence: no residual, no layer composition,
no FFN, no head — `recurrent_layer()` is a probe of one block, not a step of a
running trunk, and nothing in the `qwen3moe` forward pass calls it. (The attention
was on this list when it was written; §6.3 closed it the same day.) `--bench`, `-n`
and logits dumping remain refused on every `qwen35` file.

**Three defects the probe found, all fixed:** the `nextn.eh_proj` expectation was
transposed (`[10240, 5120]`, not `[5120, 10240]`); the projection scratch was sized
from the recurrent layer's `conv_dim` (10240) and overflowed on every attention
layer, where the gated query projection is `2 × q_dim` = 12288; and **Q5_K had no
decoder** although the loader knew its block size, which made two `ffn_down`
tensors in the 27B file unreadable. Q5_K is implemented and verified
byte-for-byte against gguf-py (2 × 89 128 960 values, max abs diff 0.0).

The 12 GB `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` is a different outcome: its
container and tokenizer verify, and the layer probe stops with
`token_embd.weight is IQ2_S, which this engine cannot decode`, because **363 of its
tensors** use eight types with no decoder here (IQ3_S 144, IQ4_XS 96, IQ3_XXS 78,
IQ2_S 17, Q2_K 13, IQ2_XS 9, IQ2_XXS 5, IQ1_M 1). Quantisation is a separate gap
from the trunk and is now reported as one.

What the engine does **not** do with a `qwen35` file: no FFN, no head, no layer
composition — and therefore no logits and no token. §6.2 and §6.3 each add one
*block* on a per-call probing path: each is one layer, not a model. `--bench`, `-n`
and logits dumping are consequently refused on these files rather than silently
reduced.

### 6.3 The gated full attention, end to end (`--qwen35-attention`, still exit 3)

Added 2026-10-08, same day. `kanjoos-run --qwen35-attention` computes one
full-attention block on the host — the fused QG projection and its **per-head**
q/gate split, an RMS norm over `head_dim` 256 for q and k, partial RoPE over 64 of
256 dims (IMROPE, sections `[11, 11, 10, 0]`), causal softmax with GQA, and the
elementwise `sigmoid(gate)` — and compares **11 named vectors** against
`tools/ref_qwen35.py --attention`: `layer_x`, `layer_qg`, `layer_q_norm`,
`layer_q_rope`, `layer_k_norm`, `layer_k_rope`, `layer_v`, `layer_scores`,
`layer_attn`, `layer_gated`, `layer_out`.

| measured | value |
|---|---|
| vectors, `blk.3` (the first attention layer) | **11/11 PASS**, max abs 1.812e-05, worst rel-RMSE 6.706e-07 |
| vectors, `blk.63` (the last attention layer before the MTP block) | **11/11 PASS**, max abs 1.144e-05, worst rel-RMSE 7.307e-07 |
| zeros / unwritten / mismatching | 240 exact zeros in `layer_scores` (all also zero in the oracle), **0 unwritten, 0 mismatching** |
| K/V carry | 5 separate calls vs one prefill: `out` and `attn` **bit-identical** |
| causality | the first 2 tokens from a fresh cache vs the prefix: **bit-identical** |
| cost | 5 token steps of one layer in 329-375 ms (host, projections single-threaded) |
| mutation (not committed) | RoPE disabled → **6 of the 11 FAIL** (`layer_q_rope`, `layer_k_rope`, `layer_scores`, `layer_attn`, `layer_gated`, `layer_out`) while the 5 upstream vectors still PASS |

The conventions this pins, each read out of upstream code on this machine rather
than from a summary — `records/qwen35-probe-2026-10-08/README.md` carries the full
table with the file and line for each:

* `attn_q` is `[q_head(256) | gate_head(256)]` **interleaved per head** (llama.cpp
  `build_layer_attn`'s `ggml_view_3d` with stride `n_embd_head*2`; HF 5.15.1
  `Qwen3NextAttention` agrees) — not `[all q | all gate]`, which is the same number
  of values and a different answer;
* the gate multiplies the attention **output** elementwise (`sigmoid(gate)`), after
  the softmax and before `wo`;
* the rotation is **IMROPE** (`src/llama-model.cpp`: `LLM_ARCH_QWEN35` →
  `LLAMA_ROPE_TYPE_IMROPE`). Its sections select the *axis* and never the
  frequency — `pos * base^(-2i/n_dims)` with the pair index running over the whole
  rotated span — and for a text-only forward pass all four axes carry the same
  token position, so the rule collapses to plain NEOX RoPE over 64 dims. The probe
  prints the per-axis pair counts (`t 11 / h 11 / w 10 / x 0`) so the collapse is
  visible rather than assumed.

What is still missing around it: the trunk wiring (no residual, no layer
composition), the dense FFN (65), the 248320-wide vocabulary head, the 4 `nextn.*`
tensors — no logits, no token. `attention_layer()` is a probe: it keeps a plain
contiguous fp32 `[n_kv_head][head_dim]` cache, which is a correctness fixture and
not the RadixKV design (docs/09).

### 6.4 The tokenizer, cross-checked against an independent implementation

Round-trip (`decode(encode(s)) == s`) does not test a **merge boundary**: a
tokenizer that splits `12345` as `[12][345]` round-trips exactly and is still not
the model's tokenizer, and every measurement downstream of it is then of a
different problem. `tools/tok_crosscheck.py` therefore compares this engine's ids
against `llama-tokenize`'s, element by element, on 25 strings written to files
(never `argv` — Windows hands a native binary its arguments in the ANSI code page):

| model | strings | identical | special-token POLICY | merge-boundary | exit |
|---|---|---|---|---|---|
| `ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf` (`qwen35`, 248320) | 25 | **24** | 1 | **0** | **0** |
| `Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf` (`qwen3moe`, 151936) | 25 | **24** | 1 | **0** | **0** |

The one POLICY string is `tab\there <|endoftext|>`: this engine tokenises raw text
and does not parse control tokens (9 ids, exactly llama.cpp's
`--no-parse-special` answer; llama.cpp's default is 5). `--strict` fails on it.

It found three defects, all fixed: the merge table was keyed by concatenation
instead of `"left right"`, so **no merge ever fired** (`"hello world"` 11 ids → 2;
`"a" × 64` 64 → 8; `"The capital of France is"` 24 → 5, matching llama.cpp in each
case); `bpe()` returned a concatenation, so a chunk that did not merge to one symbol
missed the vocabulary and was absorbed by a per-byte fallback (now each symbol is
looked up and an unknown one is a named refusal); and the pre-tokeniser's character
classes were byte tests (`c >= 0x80` = letter), now generated Unicode 15.0.0 classes
(`tools/gen_unicode_ranges.py` → `src/tokenizer/unicode_ranges.{h,cpp}`, generated).

Both are pinned by mutation, and the second is recorded as the **null result** it
is: restoring the merge-key defect fails **25/25 strings** (exit 1,
`tok-crosscheck-mutation-no-merges.log`), while restoring the byte-class rule changes
**nothing** on any of the 25 (exit 0, `tok-crosscheck-mutation-byte-classes.log`) —
byte-level BPE re-derives the same symbols inside a wider chunk. The class fix is a
real divergence from the reference regex, not a demonstrated behaviour change.
**The 248320-token tokenizer the drafters need therefore exists and is verified**;
what §8 still lists as missing is the *trunk* that consumes it.

## 7. The sized gap, in dependency order

1. **`qwen35` trunk** — 65 blocks (17 attention + 48 SSM), gated attention, QK-norm
   256, untied embeddings, vocab 248320 (+ the `gpt2`/`qwen35` tokenizer). Without
   this there is no target, so the drafters have nothing to condition on and
   **nothing can be benchmarked**; this is the blocker for the user's original
   request.   **Partially done:** the front end (geometry, binding, tokenizer,
   embeddings, one layer's input projections) is built and verified against an
   oracle — §6.1 — and as of 2026-10-08 the **GatedDeltaNet block itself** is
   implemented and verified end to end against that oracle, 12 vectors × 2 layers,
   with a mutation test proving the check can fail — §6.2 — and the **gated full
   attention** likewise as of the same day: 11 vectors × 2 attention layers,
   including the q/gate split, QK-norm, partial IMROPE RoPE and the sigmoid gate,
   verified by a mutation that removes the rotation — §6.3. What remains is the
   trunk *around* those two blocks: residual and layer composition, the FFN, the
   head, and the four `nextn.*` tensors — plus, for the drafters, the tokenizer's
   *pre* rule dispatch (§6.4).
2. **`dflash` loader** (C20's "one loader for DFlash / DFlash2 / DSpark, with an
   arch compatibility check"): one metadata reader, three tensor grammars.
   `permute`-free: the layouts above are already this engine's `[in, out]`
   convention.
3. **The drafter forward pass** — non-causal, sliding-windowed, block-parallel,
   with the selector/convs (DFlash2) or the confidence + Markov heads (DSpark).
4. **Verification and acceptance** (C20/I6) — bit-identical to non-speculative
   under a fixed seed. This is the gate that makes the whole feature a *win*
   rather than a claim, and it cannot be reached before 1–3.

Steps 1–3 are each larger than the C21 work that precedes them. Nothing here is
scheduled; this document sizes the task and records the layouts so that whichever
step is taken next starts from measurement.

## 8. What this document does not claim

* Not that the drafter code paths exist: not one line of DFlash, DFlash2 or
  DSpark is implemented. For `qwen35` the *front end* exists and is verified
  (§6.1), one GatedDeltaNet **block** is implemented and verified (§6.2) and so is
  one gated full-attention **block** (§6.3), but those are probes of one layer:
  no trunk wiring, no FFN, no head, therefore no logits and no token.
* Not that the token counts or dims are complete: the metadata above is what the
  files declare, and `dflash.selector_top_k = 16` in particular is read as a
  *declared* parameter whose use inside the selector is not established here.
* Not that the three variants are interchangeable: they differ in block size (8 /
  16 / 7), target-layer sets, FFN width, head counts, rope scaling and extra heads.
  A loader that treats `dflash` as one shape will be wrong for two of the three.
* ~~Not that a 248320-vocabulary tokenizer is available.~~ **Superseded
  2026-10-08.** The 248320-token `gpt2`/`qwen35` tokenizer is built from the file,
  its merge boundaries are cross-checked against `llama-tokenize` on 25 strings with
  **0 merge-boundary differences** on both a 248320 and a 151936 vocabulary (§6.4),
  and the reference model's own `qwen3moe` ids are unchanged by the fix. The
  `tokenizer.ggml.pre` dispatch is also done and gated: `test_tok_pre_dispatch` loads
  ten single-field fixtures that differ only in `pre`, asserts the two rules produce
  different exact id lists for a mark-bearing string, and refuses an absent or unknown
  `pre` by name; `tools/tok_pre_rules.py` re-derives the name→rule table from
  llama.cpp's own name map and regex table and fails when the two disagree. The two
  `pre` values in play here (`qwen2`, `qwen35`) map to the same regex in llama.cpp,
  so no file on this machine can distinguish them — but the dispatch is in the tree,
  not in the prose.
