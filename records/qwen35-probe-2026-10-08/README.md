# qwen35 front-end probe — Qwen 3.8 files, 2026-10-08

**MEASURED.** The first time any Qwen 3.8 container has been read, bound and
computed by this engine. `qwen35` is a **hybrid SSM + gated-attention trunk**, not
a configuration of `qwen3moe`, so what runs here is a *front end*: metadata,
tensor binding, tokenizer, embeddings and one layer's input projections. The trunk
and the head are **refused, by name**. Exit code is **3** — understood, and
deliberately not run. No logits are produced.

## The files, and what each one got

| file | bytes | result |
|---|---|---|
| `ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf` | 17 442 400 352 | **front end verified**: 866/866 tensors, 45/45 vectors vs the oracle, rc 3 |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | 12 120 016 960 | container + tokenizer verified; layer probe STOPPED: `token_embd.weight is IQ2_S, which this engine cannot decode`; 363 tensors across 8 types have no decoder |
| `Qwen3.8-9B-Q6_K.gguf` | — | front end verified (geometry 33 layers: 25 SSM + 8 attention), rc 3 |

## Commands, verbatim

```sh
export PATH="/c/Strawberry/c/bin:$PATH"        # MinGW runtime, else exit 127
G="G:/More-models/ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf"
D=records/qwen35-probe-2026-10-08

# the independent oracle: geometry-free reference vectors + manifest
python tools/ref_qwen35.py "$G" --layer 0  --tokens 760,6511,314,9338,369 --out $D/layer0
python tools/ref_qwen35.py "$G" --layer 63 --tokens 760,6511,314,9338,369 --out $D/layer63

# the engine, compared against them, full output, element by element
kanjoos-run.exe --model "$G" -p "The capital of France is" --qwen35-ref $D/layer0  --qwen35-layer 0
kanjoos-run.exe --model "$G" -p "The capital of France is" --qwen35-ref $D/layer63 --qwen35-layer 63
```

`probe-layer0.log` and `probe-layer63.log` in this directory are those two runs,
unmodified. `layer0/` and `layer63/` hold the oracle's `.f32` vectors, its
`manifest.json`, and its own dequant cross-check.

## What was verified, and how

**45 vectors, 0 mismatches, 0 unwritten.** 5 tokens (760, 6511, 314, 9338, 369 =
"The capital of France is") through two different layer kinds:

| vector set | n | layer 0 (recurrent) | layer 63 (full attention) |
|---|---|---|---|
| `embed_<id>` | 5 120 | 0.000e+00 (bit-exact) | 0.000e+00 |
| `xnorm_<id>` | 5 120 | 0.000e+00 (bit-exact) | 0.000e+00 |
| `qkv_<id>` | 10 240 | max 2.29e-05, rel-RMSE 2.8e-07 | — |
| `gate_<id>` | 6 144 | max 9.54e-06, rel-RMSE 3.3e-07 | — |
| `q_<id>` (gated) | 12 288 | — | max 6.20e-06, rel-RMSE 4.5e-07 |
| `k_<id>` | 1 024 | — | max 2.62e-06, rel-RMSE 4.3e-07 |
| `v_<id>` | 1 024 | — | max 6.68e-06, rel-RMSE 4.6e-07 |

* The oracle is `tools/ref_qwen35.py`: an independent numpy decode of the same
  GGML blocks and the same arithmetic in float32, written from the block layouts,
  not from this engine's code.
* **The oracle's decoders are themselves checked against gguf-py** (the ggml
  authors' implementation) before any vector is written — see
  `layer0/manifest.json` → `dequant_crosscheck_vs_gguf_py`: `F32`, `Q4_0`, `Q8_0`,
  `Q4_K`, `Q5_K`, `Q6_K` all **identical** (max abs diff 0.0). A reference that is
  only self-consistent proves nothing.
* Exact zeros are reported separately from **unwritten** outputs. `embed_314` and
  `embed_369` legitimately decode to 31 exact zeros (Q4_K: `d * 0 - dmin * 0`) and
  the oracle has the same 31 zeros — those rows PASS. The first version of the
  check called any zero a failure and produced a FAIL whose every measured number
  was perfect; that is fixed in the probe, not in the reference.
* The projections agree to ~1e-5 absolute because the two implementations
  accumulate differently (C++ fuses the decode into the inner product, numpy
  dequantizes then uses BLAS), which is exactly the disagreement a cross-check is
  supposed to expose.

**Geometry** (all from the file's own metadata, printed in `probe-layer0.log`):

```
layers 65 (48 recurrent/SSM + 17 full attention), hidden 5120, ff 17408,
vocab 248320, ctx_train 262144, 24 q heads / 4 kv, head_dim 256,
rope base 1e7 rotating 64 of 256 dims, ssm inner 6144 / state 128 / groups 16 /
dt_rank 48 / conv_kernel 4 -> conv_dim 10240, rms_eps 1e-6, nextn 1 layer on blk.64
layer kinds  SSSASSSASSSASSSASSSASSSASSSASSSASSSASSSASSSASSSASSSASSSASSSASSSAA
kinds source attention.recurrent_layers (per-layer, exact)
```

**Tokenizer**: vocab 248320, bos 248044, eos 248046, `add_bos` false, chat
template present, **9/9 round-trips exact** (ASCII, digits and floats, whitespace
runs, newlines, tab, CJK, emoji, a literal control token, and the empty string).

**Tensor binding**: 866 tensors bound and shape-checked across 65 layers, 4 nextn
tensors on `blk.64`, **0 not usable**. Every quantisation in the file is
decodable: F32 456, Q4_K 257, Q6_K 79, Q8_0 64, Q4_0 8, Q5_K 2.

## Two defects this probe found, and what each one was

1. **`blk.64.nextn.eh_proj.weight` is `[10240, 5120]`, not `[5120, 10240]`.** The
   binding check asserted the transpose. ggml order is dims[0] = input, so the
   fused MTP head takes 2 × hidden in and emits hidden out; the engine's
   expectation was wrong, the file was right.
2. **A heap overrun on every attention layer.** The projection scratch buffer was
   sized from the *recurrent* layer's `conv_dim` (10240), but the gated query
   projection of an attention layer is `2 * q_dim` = 12288. `--qwen35-layer 63`
   died with no output and rc 127; `--qwen35-layer 0` passed. Sized for the widest
   projection of either kind now.

A third gap was found and **closed**: `blk.51` and `blk.55` `ffn_down` are
**Q5_K**, which the loader knew the size of but had no decoder for. Implemented,
and verified **byte-for-byte against gguf-py**: 2 × 89 128 960 values,
`maxabs = 0.000000e+00`, `AGREE` (`dequant-crosscheck-q5k.log`).

## What is refused, with the reason (verbatim from the run)

> **Superseded for items 1 and 2 on 2026-10-08, same day**: the GatedDeltaNet
> recurrence (*The recurrence, end to end* below) and the gated full attention
> (*The gated full attention* below) are both implemented and verified. The list
> printed by today's probe starts at the trunk wiring. The block below is kept as
> the state the first run reported, not as the current one.

```
1. the GatedDeltaNet recurrence (48 layers): conv1d state carry over 4 taps,
   the per-group delta rule over state_size 128 x 16 groups, and the
   alpha/beta/ssm_a gating
2. the gated full attention (17 layers): the q/gate split of attn_q, QK-norm
   over head_dim 256, and sectioned RoPE (64 of 256 dims, base 10000000)
3. the dense FFN (65 layers): ffn_gate/ffn_up/ffn_down with SiLU, no router
4. the vocabulary head: output_norm + output.weight [5120, 248320], so no logits
   can be produced
5. the merged nextn MTP head (1 layer, block 64): the eh_proj fusion of
   [hidden, embedding] C20 asks for
```

For the IQ3_S `-mtp` file, additionally:

```
6. tensors whose quantisation this engine cannot decode: IQ1_M (1) IQ2_S (17)
   IQ2_XS (9) IQ2_XXS (5) IQ3_S (144) IQ3_XXS (78) IQ4_XS (96) Q2_K (13)
```

## What this record does NOT claim

* **Not a working Qwen 3.8 model.** 48 of 65 layers are still unimplemented;
  nothing here produces a token.
* ~~**Not a claim that the RoPE convention is right.**~~ **Closed 2026-10-08:** the
  partial RoPE is implemented and the 11-vector attention comparison tests it —
  including a mutation that removes it and fails 6 of the 11 — see *The gated full
  attention* below. What is still not tested is the **IMROPE axis split** itself:
  this file is text-only, so all four axes carry the same token position and the
  sectioned rule reduces to plain NEOX RoPE over 64 dims (the probe prints the
  per-axis pair counts, `t 11 / h 11 / w 10 / x 0`, so the reduction is visible
  rather than assumed). A multimodal file would be needed to test the rest.
* ~~**The tokenizer's merge boundaries are not cross-checked.**~~ **Closed
  2026-10-08:** `tools/tok_crosscheck.py` compares this engine's ids against
  `llama-tokenize` on 25 strings, element by element — see *The tokenizer's merge
  boundaries* below. It found three defects in this engine, all fixed: the merge
  table was keyed by concatenation instead of `"left right"`, so **no merge ever
  fired**; `bpe()` returned a concatenation, so a chunk that did not merge to one
  symbol missed the vocabulary and was silently absorbed by a per-byte fallback;
  and the pre-tokeniser's character classes were byte tests (`>= 0x80` = letter)
  instead of Unicode classes. Round-trip (`decode(encode(s)) == s`) was passing
  through all three.
* The 12 GB `-mtp` file's layer-kind pattern is **derived** from
  `full_attention_interval` (49 SSM / 16 attention) because that file carries no
  `recurrent_layers` array; ThinkingCap's file does, and gives 48/17. The two
  files therefore disagree, and only one of the two statements can be exact. The
  probe prints which source it used.

---

# The recurrence, one layer, end to end (2026-10-08, same day)

**MEASURED.** `kanjoos-run --qwen35-recurrent` computes one GatedDeltaNet layer's
full output on the host — conv1d state carry, the per-group delta rule, the
alpha/beta/ssm_a gating — and compares **12 vectors**, element by element, against
the oracle's `layer_*` files. Exit code is still **3**: one layer is not a model.

## Commands, verbatim

```sh
export PATH="/c/Strawberry/c/bin:$PATH"      # MinGW runtime, else exit 127
G="G:/More-models/ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf"
D=records/qwen35-probe-2026-10-08

python tools/ref_qwen35.py "$G" --layer 0 --tokens 760,6511,314,9338,369 \
       --recurrent --out $D/recurrent0
python tools/ref_qwen35.py "$G" --layer 4 --tokens 760,6511,314,9338,369 \
       --recurrent --out $D/recurrent4

build/cmake-host/kanjoos-run.exe --model "$G" -p "The capital of France is" \
       --qwen35-recurrent --qwen35-ref $D/recurrent0 --qwen35-layer 0
build/cmake-host/kanjoos-run.exe --model "$G" -p "The capital of France is" \
       --qwen35-recurrent --qwen35-ref $D/recurrent4 --qwen35-layer 4
```

`probe-recurrent0.log` and `probe-recurrent4.log` are those two runs, unmodified.

## Result: 24/24 vectors, 0 mismatches, 0 unwritten, across two layers

| vector | n | layer 0 max abs | layer 0 rel-RMSE | layer 4 max abs | layer 4 rel-RMSE |
|---|---|---|---|---|---|
| `layer_x` | 25 600 | 0.000e+00 (bit-exact) | 0 | 0.000e+00 | 0 |
| `layer_qkv` | 51 200 | 2.289e-05 | 2.365e-07 | 4.768e-06 | 4.197e-07 |
| `layer_conv_raw` | 51 200 | 3.815e-06 | 2.499e-07 | 5.960e-07 | 4.257e-07 |
| `layer_conv_silu` | 51 200 | 1.907e-06 | 2.739e-07 | 4.768e-07 | 4.376e-07 |
| `layer_q_l2` | 10 240 | 2.980e-08 | 2.766e-07 | 4.657e-08 | 4.437e-07 |
| `layer_k_l2` | 10 240 | 3.576e-07 | 2.852e-07 | 6.482e-07 | 4.380e-07 |
| `layer_beta` | 240 | 7.749e-07 | 2.379e-07 | 1.058e-06 | 4.318e-07 |
| `layer_g` | 240 | 6.676e-06 | 5.309e-07 | 8.774e-05 | 2.415e-06 |
| `layer_o` | 30 720 | 6.706e-08 | 6.032e-07 | 1.583e-08 | 1.131e-06 |
| `layer_state` | 786 432 | 9.537e-07 | 4.591e-07 | 1.490e-07 | 8.112e-07 |
| `layer_gated` | 30 720 | 1.049e-05 | 5.394e-07 | 1.335e-05 | 1.197e-06 |
| `layer_out` | 25 600 | 1.907e-05 | 7.249e-07 | 8.106e-06 | 1.279e-06 |

Tolerance is rtol 2e-3 relative to the reference RMS, so the worst observed
rel-RMSE (2.4e-06) is ~800x inside it. The residual is the dequantisation and
accumulation-order difference between C++ (`dequant_dot_f32` fuses the decode into
the inner product) and numpy (dequantise, then BLAS), not a modelling difference.

What the run prints about itself:

```
exp(g) : [0.000474, 0.999999]   layer 0
exp(g) : [0.000000, 0.999997]   layer 4    (printed 0.000000 = an underflowed exp of < -20)
state  : 48 x 128 x 128 = 786432 value(s) (one per head: key, value)
state carry : 5 separate call(s) vs one call -- out BIT-IDENTICAL, state BIT-IDENTICAL
causality   : first 2 token(s) from a fresh state vs the prefix -- BIT-IDENTICAL (max|d| 0.0)
```

`exp(g) in (0, 1]` is the reading check for `ssm_a`: the GGUF stores
`-exp(A_log)`, so g must be negative and the decay must never exceed 1. A sign
flip would make `exp(g) >= 1` and the recurrence would blow up. The probe prints
the range rather than asserting it.

## The check that can fail

A harness that has only ever passed is not evidence. One `delta`-rule mutation was
built and run (**not committed**; the source was restored byte-identical and
re-verified afterwards):

```
delta[j] = (v[j] - kv_mem[j]) * beta[h]   ->   delta[j] = (v[j] - kv_mem[j])
```

Result (`mutation-drop-beta.log`): `layer_o`, `layer_state`, `layer_gated` and
`layer_out` **FAIL** (max abs 1.1e-01 / 1.8e+00 / 4.0e+00 / 7.4e+00; 23k-554k
mismatching elements) while the eight upstream vectors still **PASS**. So the
comparison is live, it localises the defect to the step that is wrong, and it is
not a rubber stamp.

The refusal path is exercised too: `--qwen35-recurrent --qwen35-layer 3` (a
full-attention layer) stops with `qwen35: layer 3 is a full-attention layer; the
GatedDeltaNet recurrence (conv1d carry, delta rule, gating) is not defined for
it`, exit 3 (`refuse-attention-layer.log`).

## Why 12/12 is not just "the oracle agrees with itself"

The oracle and the engine were both written from the same two references, so
agreement alone could hide a *shared* misreading. Every step was therefore checked
against upstream code on this machine, read directly rather than paraphrased:

| question | source, read on this machine | what it says |
|---|---|---|
| is GGUF `attn_qkv` plain q\|k\|v, or HF's per-group interleave? | `src/models/qwen35.cpp` (llama.cpp) lines 378-447 | conv over the fused tensor, then `ggml_view` offsets **0 / `head_k_dim*num_k_heads` / `2*head_k_dim*num_k_heads`** — a plain `[q\|k\|v]` split, no reorder step |
| conv tap orientation | `ggml_compute_forward_ssm_conv_f32` (`ops.cpp:9703`) and `build_conv_state` (`delta-net-base.cpp:449`) | the window starts at the token index and tap `i0` walks forward, so **tap 0 multiplies the oldest** sample; `conv_input = concat(conv_states, qkv)`, i.e. the state is the last `kernel-1` inputs, oldest first |
| decay / delta / update / output order, state orientation | `ggml_compute_forward_gated_delta_net_one_chunk` (`ops.cpp:10895`) | `S *= exp(g)`; `delta[j] = (v[j] - sum_i S[i][j]*k[i]) * beta`; `S[i][j] += k[i]*delta[j]`; `out[j] = sum_i S[i][j]*q[i] * scale` — identical to the engine, with the 1/sqrt(d) scale on the output rather than on q |
| `ssm_a` sign convention | HF 5.15.1 `Qwen3NextGatedDeltaNet.forward`; llama.cpp `qwen35.cpp:373` | `g = -A_log.exp() * softplus(a + dt_bias)`, with llama.cpp's own comment `// -A_log.exp() * softplus` |
| L2-norm formula | HF 5.15.1 `l2norm()` (`modeling_qwen3_next.py:368`) | `x * rsqrt((x*x).sum(-1) + eps)` — **the engine's formula**; ggml uses `1/max(sqrt(sum), eps)` (`ops.cpp:4333`), which differs by ~1e-6 relative at these magnitudes |
| L2 eps, and the head repeat | HF `torch_recurrent_gated_delta_rule`; `qwen35.cpp:426` | `eps = 1e-6` (HF hardcoded; llama.cpp takes the file's own `layer_norm_rms_epsilon`, 1e-6 here); `repeat_interleave(nv/nk)` = 16 key heads -> 48 value heads, `kh = h/3` |

So the layout, the tap order, the delta rule, the sign convention and the norm
formula each have a third-party source that was read, not remembered. The one
named divergence is cosmetic and is recorded rather than smoothed over: for a q/k
head *at* the eps floor the two norms differ materially, which cannot arise for a
real token (the measured per-head norms are ~0.79, i.e. ~1e6 x eps).

## Still open (as of the recurrence section; see the sections below for what has since closed)

* **No logits.** The dense FFN (65), the vocabulary head and the merged `nextn.*`
  MTP head are unimplemented, and the probe's missing list now starts at *the
  trunk wiring* (no residual, no layer composition). One layer is not a model.
  (**Closed for the attention since this section was written** — see *The gated
  full attention* below; the FFN, the head and the trunk wiring are still open.)
* `recurrent_layer()` is a **probe**, not a trunk step: it takes token ids,
  computes the embeddings itself, keeps its state in the probe object and returns
  every intermediate. The `qwen3moe` forward pass does not call it.
* ~~The tokenizer merge boundaries are still only round-trip-checked.~~ Closed
  later the same day — see *The tokenizer's merge boundaries* below, which found
  and fixed three defects in this engine's BPE.
* `head_k_dim == head_v_dim` here (both `ssm.state_size`, 128), so HF's
  `1/sqrt(head_k_dim)` and llama.cpp's `1/sqrt(S_v)` are indistinguishable on this
  file. A file where the two differ could not be expressed through this metadata.
* The 1/sqrt(d) scale placement (on q, as HF does, or on the output, as llama.cpp
  does) is mathematically identical for a scalar; it is one multiply either way.

## Provenance for this section

```
sha256  src/model/qwen35.cpp   3af5bac26471204e0fc5c3e36c0fcca6b74634de0cd8d8776fdad8061fd3bb4a
sha256  src/model/qwen35.h     7a0a233ee0997ac510b51a88895c4fae016dcaa638b246947442f71e61e75b22
sha256  src/cli/main.cpp       1fe151827ae2eda35c3575d3da203cb58ba515d354f3cbd9a5da32f9300a4244
```

**These three hashes are the Phase-48 revision of those files, not the current one** —
the attention section below changed `qwen35.cpp`, `qwen35.h` and `main.cpp`. The
current hashes are in *Provenance, both new sections*, at the end.

`cmake --build build/cmake-host` reports `ninja: no work to do` at these hashes, and
re-running the `blk.0` command reproduces `probe-recurrent0.log` byte-for-byte
except for the two timing lines. The CLI flag is `--qwen35-recurrent`; there is no
build-system change, so tier A/B/C, `ctest -R c21` and `check_docs.py` are
unaffected by this section (all re-run green: rc 0, 1/1, rc 0).

---

# The gated full attention, one layer, end to end (2026-10-08, same day)

**MEASURED.** `kanjoos-run --qwen35-attention` computes one **gated full-attention
layer's** complete output on the host — the fused QG projection and its per-head
split, an RMS norm over `head_dim` 256 for q and k, partial RoPE over 64 of 256
dims, causal softmax with GQA, and the elementwise `sigmoid(gate)` — and compares
**11 vectors**, element by element, against the oracle's `layer_*` files. Exit code
is still **3**: one layer is not a model.

## Commands, verbatim

```sh
export PATH="/c/Strawberry/c/bin:$PATH"      # MinGW runtime, else exit 127
G="G:/More-models/ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf"
D=records/qwen35-probe-2026-10-08

python tools/ref_qwen35.py "$G" --layer 3  --tokens 760,6511,314,9338,369 \
       --attention --out $D/attention3
python tools/ref_qwen35.py "$G" --layer 63 --tokens 760,6511,314,9338,369 \
       --attention --out $D/attention63

build/cmake-host/kanjoos-run.exe --model "$G" -p "The capital of France is" \
       --qwen35-attention --qwen35-ref $D/attention3  --qwen35-layer 3
build/cmake-host/kanjoos-run.exe --model "$G" -p "The capital of France is" \
       --qwen35-attention --qwen35-ref $D/attention63 --qwen35-layer 63
```

`probe-attention3.log` and `probe-attention63.log` are those two runs, unmodified.

## Result: 22/22 vectors, 0 mismatches, 0 unwritten, across two attention layers

`blk.3` is the **first** attention layer of the file; `blk.63` is the last of the 16
before the MTP block (the pattern ends `…SSSSAA`), so the two ends of the attention
stack are both covered.

| vector | n | blk.3 max abs | blk.3 rel-RMSE | blk.63 max abs | blk.63 rel-RMSE |
|---|---|---|---|---|---|
| `layer_x` | 25 600 | 0.000e+00 (bit-exact) | 0 | 0.000e+00 | 0 |
| `layer_qg` | 61 440 | 4.768e-06 | 5.419e-07 | 6.199e-06 | 4.195e-07 |
| `layer_q_norm` | 30 720 | 4.530e-06 | 5.572e-07 | 4.292e-06 | 4.360e-07 |
| `layer_q_rope` | 30 720 | 4.530e-06 | 5.573e-07 | 4.292e-06 | 4.361e-07 |
| `layer_k_norm` | 5 120 | 3.219e-06 | 4.406e-07 | 3.338e-06 | 4.255e-07 |
| `layer_k_rope` | 5 120 | 3.219e-06 | 4.408e-07 | 3.338e-06 | 4.256e-07 |
| `layer_v` | 5 120 | 3.099e-06 | 4.355e-07 | 6.676e-06 | 4.164e-07 |
| `layer_scores` | 600 | 8.941e-07 | 4.419e-07 | 9.239e-07 | 2.570e-07 |
| `layer_attn` | 30 720 | 7.153e-06 | 6.124e-07 | 1.144e-05 | 4.976e-07 |
| `layer_gated` | 30 720 | 6.169e-06 | 6.634e-07 | 6.914e-06 | 5.373e-07 |
| `layer_out` | 25 600 | 1.812e-05 | 6.706e-07 | 8.106e-06 | 7.307e-07 |

Tolerance is rtol 2e-3 of the reference RMS, so the worst rel-RMSE (7.3e-07) is
~2700× inside it, and the residual is again the C++-fuses-the-decode vs
numpy-dequantise-then-BLAS difference, not a modelling difference. Timing (host,
single-threaded projections, 5 tokens of one layer): **329 ms** at `blk.3`,
**375 ms** at `blk.63`. An earlier run of the same command reported 411 ms and
577 ms — a ~1.5× spread on identical work. This is a probe timing on a 16 GB mapped
file, not a benchmark, and it is reported as unstable rather than averaged.

What the run prints about itself:

```
tokens        : 5 760 6511 314 9338 369
heads         : 24 q / 4 kv x 256 dims, scale 1/sqrt(256) = 0.062500; gate is
                per head-dim, sigmoid-ed after the attention
rope          : partial NEOX over 64 of 256 dims, base 1e+07; 32 pair(s) per
                IMROPE axis [t 11, h 11, w 10, x 0] -- text-only, so all four
                carry the token position
kv carry      : 5 separate call(s) vs one prefill -- out BIT-IDENTICAL, attn BIT-IDENTICAL
causality     : first 2 token(s) from a fresh cache vs the prefix -- BIT-IDENTICAL
```

`layer_scores` carries **240 exact zeros in 600 values** — the strictly upper
triangle of the 5×5 causal score matrix (4 of 5 rows × 24 heads × 2.5 average masked
positions). All 240 are zero in the oracle too, so they PASS. Reported separately
from `unwritten`, which is 0 everywhere.

## The readings that make the numbers mean something

The comparison says the engine agrees with the oracle; these say the oracle's vectors
are the shape the convention implies:

* **RoPE touches exactly 64 dims.** `layer_q_rope` minus `layer_q_norm`: max |Δ| is
  6.33 (`blk.3`) / 3.43 (`blk.63`) over dims 0..63 and **0.000e+00 over dims
  64..255** — the file declares `rope.dimension_count = 64`, and dims outside it are
  passed through untouched.
* **RoPE preserves the norm** (it is a rotation): per-head RMS is 1.2328 before and
  1.2328 after, on both layers. A transposed or half-applied rotation does not keep
  the norm *and* agree elementwise, so this is a second, independent reading.
* **The attention rows are distributions:** every row of `layer_scores` sums to 1.0,
  every strictly-above-diagonal entry is 0, and the first token's row is one-hot (it
  attends only to itself).
* **The QK-norm really is a norm over `head_dim`:** per-head RMS after it is 1.23 (a
  256-dim unit vector scaled by the learned weight), not the ~1e-3 a norm over the
  wrong axis would give.

## The check that can fail

As with the recurrence, a mutation was built and run (**not committed**; the source
was restored byte-identical, `sha256` re-checked, and the run repeated):

```
rotate_pair(q, i, pos)  ->  q                 (RoPE disabled for q and k)
```

Result (`mutation-no-rope.log`): **6 of the 11 vectors FAIL** — `layer_q_rope` 2460
mismatching values (first at 6144), `layer_k_rope` 402, `layer_scores` 281,
`layer_attn` 19459, `layer_gated` 17368, `layer_out` 19428 — while `layer_x`,
`layer_qg`, `layer_q_norm`, `layer_k_norm` and `layer_v` still **PASS**. So the
comparison is live, it localises the defect to the rotation and everything
downstream of it, and `layer_out` fails exactly because the first wrong value
propagates.

## Why the attention arithmetic is not just "the oracle agrees with itself"

Same method as the recurrence: every step was read out of upstream code on this
machine, not remembered.

| question | source, read on this machine | what it says |
|---|---|---|
| is `attn_q` really `[q\|gate]` per head, or q then gate as blocks? | llama.cpp `src/models/qwen35.cpp` `build_layer_attn` lines 260-330; HF 5.15.1 `Qwen3NextAttention` | `ggml_view_3d` with stride `n_embd_head*2` and the gate at offset `n_embd_head` — **interleaved per head**, `[q_head(256) \| gate_head(256)]`, not `[all q \| all gate]` |
| where does the gate multiply? | both of the above | elementwise on the attention **output**, after the softmax and before `wo`, `sigmoid(gate)` — not on q, and not on the scores |
| norm over what? | `attn_q_norm.weight` is `[256]` = `head_dim`, and both references call the norm per head | RMS norm **over the 256 head dims** for q and k separately, applied before RoPE |
| which RoPE, and over which dims? | llama.cpp `src/llama-model.cpp`: `case LLM_ARCH_QWEN35: return LLAMA_ROPE_TYPE_IMROPE`; `ggml-cpu` `rope_yarn`'s `is_imrope` branch | **IMROPE**: the sector rule `sector = pair % sum(sections)`; `sections = [11, 11, 10, 0]` and NEOX pair ordering is automatic and cannot be disabled |
| is the theta exponent reset per section? | `ggml-cpu` `rope_yarn` | no — `pos * base^(-2i/n_dims)` with the **pair index running over the whole rotated span**, so the sections choose the *axis*, never the frequency |
| what is an IMROPE axis for text? | same | `t` (temporal), `h`, `w` and a fourth "extra" sector; a text-only forward pass gives all four the same token position, so the rule collapses to plain NEOX over 64 dims. The engine implements the *rule*, and the probe prints the pair counts per axis so the collapse is visible |
| does HF rotate the same pairs? | HF 5.15.1 `apply_rotary_pos_emb` | it splits the head at `rotary_dim` and applies `rotate_half` to the first part — the same as ggml's `rotate_pairs` over `rd/2` pairs. The engine's `rope_head()` rotates pair `i` as `(i, i + rd/2)` |

The measured consequence of the first row is in the numbers: `layer_qg` is 61 440
values = 5 tokens × 24 heads × 2 × 256, and the split is per head — an
`[all q | all gate]` split would still be 61 440 values and would fail every
downstream vector.

## Refusal path

`--qwen35-attention --qwen35-layer 0` (a recurrent layer) stops with
`qwen35: layer 0 is recurrent; the gated attention (QG split, QK-norm, partial RoPE,
causal softmax) is not defined for it`, exit 3 (`refuse-recurrent-layer.log`). The
recurrence's mirror case is `refuse-attention-layer.log`.

## Still open (what the attention closed, and what it left)

* **No logits.** The trunk wiring (no residual, no layer composition), the dense FFN
  (65 layers), the 248320-wide vocabulary head and the 4 `nextn.*` tensors are
  unimplemented; `--bench`/`-n` stay refused on every `qwen35` file.
* `attention_layer()` is a **probe**, not a trunk step: it takes token ids, embeds
  them itself, keeps its K/V cache in the probe object, and materialises 11
  intermediates per call (~730 KB per token). A trunk step needs the residency-aware
  buffer plan (C4/C11), which is deliberately not answered here.
* The **KV cache it carries is a plain contiguous `[n_kv_head][head_dim]` per token
  in fp32** — not the RadixKV layout, not codec-quantised. A correctness fixture, not
  a design.
* `head_k_dim == head_v_dim` on this file, so no attention-side scale-placement
  question can be raised here.
* The IMROPE axis rule is implemented and printed, but only its collapse is
  exercised (text-only). A multimodal input would be needed to test sections 1–3.

## Provenance, both new sections

```
sha256  src/model/qwen35.cpp               f47844ee2eed76cb5ad82ab77f1c3e8554719b3336601ebe0c2721ff34a5cbff
sha256  src/model/qwen35.h                 8f3cb3be3867573468df3aa4bb529cd5235324e6a2eb8a8447bdb3d384875b11
sha256  src/cli/main.cpp                   ebbf7e66393082388b03a0c145e00928bd444808bb3a697bb64c031ee1c288e3
sha256  src/tokenizer/tokenizer.cpp        1e60925353b9b0bf8477044512150774491b98da3a61db0e46bea49fac254618
sha256  src/tokenizer/tokenizer.h          8f1108a77032f5536eebfa65145d8639e45b4cf6cbec9b312318f4159950cc76
sha256  src/tokenizer/unicode_ranges.cpp   125541d915373be583e9beefc5ec7d35a5f6429fa164c1f3fe38267fde88865d
sha256  src/tokenizer/unicode_ranges.h     d558e0d9d4a15fc193f4f57b48d365496e517c6265d51bb09297b859bb92f40b
sha256  tools/gen_unicode_ranges.py        9df5c941eb5e3dfbfbce92695ca1efd3738920004802e5ed17bd47771024701a
sha256  tools/tok_crosscheck.py            58eb6c54e2f123a9a0455627ee25ffc5f73b40dffcbdaa44c9f066dfabac87e3
sha256  tools/ref_qwen35.py                bd211305bb85afee23f0890d7e067f5eb15f1d68c37388fdc67cefc3545afb50
sha256  CMakeLists.txt                     98af2c0acc5fc3016c618363edc3e2c13a9a71ec540ba32e0532b305e94c82b3
```

The `sha256` block above the recurrence section is the **Phase 48** revision of those
files; these are the current ones, and `src/model/qwen35.cpp` is the file that
changed (the attention was added to it). The probe logs in this directory were
re-run at these hashes — `probe-attention3.log`, `probe-attention63.log` and
`tok-crosscheck.log` are current. The round-trip table inside `probe-layer*.log` and
`probe-recurrent*.log` predates the tokenizer fix; their `probe text … id(s)` lines
are unchanged, because that string tokenised to the same 5 ids before and after (see
the tokenizer section below).

---

# The tokenizer's merge boundaries, cross-checked against llama.cpp (2026-10-08)

**MEASURED.** `tools/tok_crosscheck.py` runs two independent tokenizers on the same
bytes and compares the id lists **element by element**:

* `llama-tokenize --ids --no-bos` from the local llama.cpp build, which carries its
  own implementation of the `qwen35` pre-tokeniser;
* this engine, via `kanjoos-run --prompt-file … --dump-tokens -n 0` (or the qwen35
  probe's own `id(s):` line — the tool reads whichever the architecture prints).

Every string goes through a **file**, never `argv`: measured on this machine, a CJK
or emoji prompt handed to a native Windows binary arrives in the ANSI code page (or,
when a conversion ends on a byte the CRT reads as a delimiter, as several
arguments). That is a property of the process boundary, not of either tokenizer, and
it would have produced differences that look like tokenizer defects.

llama.cpp is run **twice** per string — its default (special-token parsing on) and
`--no-parse-special`. A string that matches only the second is reported as a
**POLICY** difference, never as a pass; a string that matches in neither is a
**merge-boundary difference**, which is the failure this gate exists to catch.

## Commands, verbatim

```sh
python tools/tok_crosscheck.py --model "G:/More-models/ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf"
python tools/tok_crosscheck.py --model "C:/Users/rr/OneDrive/Desktop/kraken/models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf"
```

`tok-crosscheck.log` (here) and `../c21-baseline-2026-10-08/tok-crosscheck-qwen3moe.log`
are those two runs.

## Result: 0 merge-boundary differences, on both models

| model | strings | identical | policy | boundary | tokens compared | exit |
|---|---|---|---|---|---|---|
| `ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf` (`qwen35`) | 25 | **24** | 1 | **0** | 210 | **0** |
| `Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf` (`qwen3moe`) | 25 | **24** | 1 | **0** | 208 | **0** |

The one policy string is `tab\there <|endoftext|>`: this engine tokenises raw text
and does not parse control tokens, so it gives 9 ids where llama.cpp's default gives
5 — and exactly the 9 llama.cpp gives with `--no-parse-special`. `--strict` turns
that into a failure; it is kept visible rather than hidden by running llama.cpp in
one mode only.

The set covers digits-vs-letters, leading and repeated spaces, newlines and tabs,
apostrophes, punctuation runs, 64 repeated characters, a long word that must split,
CJK, emoji, and the class-rule probes below. The **qwen3moe** run is the control that
matters for the rest of the repo: the tokenizer is shared, and the one model this
engine actually generates with was re-checked on the same 25 strings.

## The three defects this found, and what each one cost

1. **The merge table was keyed by concatenation.** `bpe()` looked up
   `syms[i] + syms[i+1]`; the file spells a merge as its two halves **separated by a
   space** (`"Ġ Ġ"`, `"h e"`). So *no merge ever fired* — and because a chunk that is
   not a single vocabulary entry then fell through a per-byte path, the output looked
   plausible. Measured before/after on the same file: `"hello world"` 11 ids → **2**
   (llama.cpp 2); `"a" × 64` 64 → **8** (llama.cpp 8); `"The capital of France is"`
   24 → **5** (llama.cpp 5).
2. **`bpe()` returned** the concatenation of the merged symbols, and `encode` looked
   *that* up — so a chunk that did not merge down to exactly one symbol missed the
   vocabulary and was silently absorbed by the byte fallback. The fix returns the
   *symbol vector* and looks up each symbol; a symbol with no token is now a **named
   refusal** (`merges produced a symbol the vocabulary has no token for (bytes …)`)
   instead of a silently different prompt.
3. **The pre-tokeniser's character classes were byte tests**: `c >= 0x80` counted as
   a letter, so non-ASCII symbols and non-ASCII spaces were classified by accident.
   Replaced with classes generated from Python's `unicodedata` (Unicode 15.0.0) by
   `tools/gen_unicode_ranges.py`: 724 letter+mark ranges, 137 number ranges, 10 space
   ranges (`src/tokenizer/unicode_ranges.{h,cpp}`, generated — do not hand-edit).

## The check that can fail, and one that could not

Two mutations were built and run, each restored byte-identically afterwards
(`sha256` re-checked, the tree rebuilt and re-verified). They live in
`../c21-baseline-2026-10-08/` because they were run against the shared tokenizer with
the small model.

**Mutation A — put defect 1 back** (`merged = syms[i] + syms[i+1]`, no space):

```
0/25 strings identical, 0 policy difference(s), 25 merge-boundary difference(s), 550 tokens compared
VERDICT: FAIL -- a merge boundary differs from llama.cpp's on 25 string(s).
```

(`tok-crosscheck-mutation-no-merges.log`, exit 1.) The gate is live, it fails loudly,
and it fails on *every* string rather than on one — the failure mode is the one that
shipped.

**Mutation B — put defect 3 back** (every non-ASCII codepoint forced to `Letter`,
which is what the byte test did): **24/25 identical, 0 boundary, exit 0**
(`tok-crosscheck-mutation-byte-classes.log`). A **null result**, recorded as one:
once merges work, byte-level BPE re-derives the same symbols inside a wider chunk, so
on all 25 strings — including ten written specifically to probe the class rule
(`a🚀b`, `a—b`, `10–20`, CJK around `、`, a NO-BREAK SPACE, an ellipsis, `€100 £5`, a
thin space) — the old classifier is indistinguishable from the new one. Defect 3 is a
real divergence from the reference regex; it is **not** demonstrated to change any
output on this vocabulary. "Three defects found and fixed" without this sentence
would have been the wrong claim.

The A/B was first run on the ten class-rule strings alone (10/10 identical under
both), so the null result is not an artifact of the larger default set.

## What this does NOT prove

* **llama.cpp is not the trainer.** It is an independent implementation of the
  `qwen35` pre-tokeniser (`src/llama-vocab.cpp`, `LLAMA_VOCAB_PRE_TYPE_QWEN35`, the
  qwen2-family regex) and it agrees on 25 strings, but the Qwen 3.8 release's own
  tokenizer was not run. Agreement with a second implementation is what is claimed,
  and the two could still share a misreading of the regex.
* **This engine implements the qwen2-era pre-tokeniser for every `gpt2` file**,
  printing `tokenizer.ggml.pre` but not dispatching on it. The file here says
  `qwen35`, which llama.cpp maps to the same regex, so nothing is distinguishable
  today; a file whose `pre` needs a different rule would silently get the qwen2 one.
  An open item.
* **25 strings is a spot check, not a corpus.** No fuzz pass over random bytes, and
  no comparison against a long natural-language document.
* The **tool itself had a defect** that hid the qwen3moe result: it read only the
  qwen35 probe's `id(s):` marker, so on a `qwen3moe` file it stopped with *"the
  engine printed no id list"* and exit 1. The earlier
  `tok-crosscheck-qwen3moe.REFUSED-tool-defect.log` in the c21 directory is that run,
  kept verbatim. It now reads `--dump-tokens -n 0`'s `prompt ids :` line too, and both
  paths produce the numbers above.
