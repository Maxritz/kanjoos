# Worksheet — tools/kvroof (existing, the number source of truth)

**What it is:** `tools/kvroof/kv_roofline.py` reads a model's `config.json` and derives the KV
roofline, the crossover context, and the Class A/B ceilings. Everything numeric in the KV/weight
story is either measured in `00` §7–§9 or **derived** by this tool from a real config + the measured
bandwidths in `00` §8.1. The two are kept visually distinct everywhere.

**Why it is the source of truth for a coder:** any component that quotes a KV bytes/token, a
crossover, a Class A/B ceiling, a weight-format operating point, or a resident weight total must
trace back to this tool's output. Do not hardcode "48 layers × 4 KV heads × 128 head_dim" — the tool
reads it from the config and refuses (exit 2/3/4) when it cannot.

**What it reads from the config (from `09` §9.1, the tool's section A):**
num_hidden_layers=48, num_key_value_heads=4, head_dim=128, num_attention_heads=32 (GQA ratio 8),
hidden_size=2048, vocab_size=151936, tie_word_embeddings=false, torch_dtype=bfloat16, num_experts=128,
num_experts_per_tok=8, moe_intermediate_size=768, max_position_embeddings=40960. Config sha256
`2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297` (from `models/README.md`).

**Measured bandwidths it is given (not derived) — `00` §8.1:**
VRAM 589–598 GB/s; PCIe streaming pinned 27.9–28.0 GB/s; **one 2.47 MB expert copy 13.4–14.7 GB/s**
(the size that matters). Derived peak 39.3 TFLOP/s packed-f16 (`00` §8.2).

**Key outputs a coder will cite:**
* Section B — KV bytes/token per codec: FP16 98304 (96.0 KiB); FP8 E4M3 49152 (48.0 KiB); INT8 g64
  51456 (50.2 KiB); INT8 g128 50304 (49.1 KiB); INT4 g64 26880 (26.2 KiB); INT4 g128 25728 (25.1 KiB).
  Metadata is 3 B/group (fp16 scale + uint8 zero-point) for every quantised codec — INT4 is NOT half of
  FP16 on disk. KV group = 128 in the doc's codec table.
* Section C — weight bytes per token at M=1, W4: routed experts 948 MB; attention q/k/v/o 474 MB; lm_head
  163 MB; **total 1,586 MB**. Expert-only subtotal 948 MB (the convention the docs have been using; the lm_head
  is 10% of total and was missing from it).
* Section D — crossover: FP16 9.6 K (expert-only) / 16.1 K (total w/ lm_head); FP8 19.3 / 32.3 K; INT8 g64
  18.4 / 30.8 K; INT4 g128 36.9 / 61.6 K.
* Section E — traffic dominance at fixed context (2K/8K/16K/32K/128K): KV overtakes expert weight traffic at
  ~9.6 K and never comes back.
* Section F — attention ridge at M=1, 16 K: FP16 memory-bound 8.3×; INT8 g64 4.4×; INT4 g64 2.3×.
* Section G — Class A ceiling at 16 GiB, 0.75 GiB workspace: **negative** KV budget at W4 (15.31 GiB resident
  before workspace). No fit for any codec at W4.
* Section G.1 — joint residency: KV budget vs expert slots/layer, idealised uniform budget (0.25/0.5/1/2/4/6 GiB):
  125/123/118/109/91/73 of 128 slots, with FP16/FP8 context.
* Section K — weight-format decision: the five-pack table (W4 g128 / W3 g128 / W2 g128 / W4 g64 / W8 g128) with
  B/weight, expert bank, resident, KV budget, FP16 context, FP8 context, slots/layer. **This is the operating-point
  table.** W3 g128: 0.3984 B/weight, 10.76 GiB bank, 11.79 GiB resident, 3.46 GiB KV budget, 37.8 K FP16 / 75.6 K FP8
  context, 126/128 experts. The hit-rate and miss-cost columns depend on the route artifact, not the config — they are
  NOT derived here; the tool says so.
* Section H — Class B price at 13.4–14.7 GB/s, per step, whole step budget: 8 K → 54.8–60.1 ms FP16 / 27.4–30.0 ms FP8;
  16 K → 109.6–120.2 / 54.8–60.1; 64 K → 438.3–480.8 / 219.1–240.4. Plus the fixed-wire-budget extra-context table
  (10 ms allowance: ~1.4 K FP16, ~2.8 K FP8 — the codec buys exactly 2× because it halves the bytes).
* Section I — doc claim audit: 71 claims, exit 0 when they all agree.

**Metadata rule (verbatim, the one thing not to get wrong):** 3 B per group (fp16 scale + uint8 zero-point),
whatever the bit width, matching `tools/bench/gemm_w4.hip`. For expert packs, `kvroof.pack_bytes_per_weight`:
W4 g128 = 0.5234 (0.5 payload + 3/128); W3 g128 = 0.3984 (3/8 payload + 3/128); W2 g128 = 0.2734 (2/8 + 3/128);
W4 g64 = 0.5469 (0.5 + 3/64); W8 g128 = 1.0234 (1.0 + 3/128).

**Audit commands (run before and after any component that touches a documented number):**
* `python tools/kvroof/kv_roofline.py models/qwen3-30b-a3b/config.json --check` — exit 0 = 71 claims agree.
* `python tools/check_docs.py` — exit 0 = 24 documented figures match the tools (this tool + route artifacts + I7).

**What a coder must not break:**
* The "derived, not measured" distinction. A number from section K is derived from the config; a number from
  `00` §8 is measured on the GPU. Do not blur them.
* The 3 B/group metadata rule. Change it and every INT4/INT8 row in the doc becomes wrong by 2×.
* The KV group = 128 for the doc's codec table. The bench `gemm_w4.hip` defines it; the tool matches it; the doc
  quotes it. Do not silently switch to group 64 for KV (group 64 is the default for KV in the packer's own
  definition, but the doc's KV codec table is group 128 — cite the tool, not a memory).
* The "KV bytes/token" for a codec must include the 3 B/group metadata. A row that omits it (e.g. INT4 = 24 KiB
  instead of 25.1 KiB) is the earlier draft's error, flagged by `--check`.

**Files a coder should read before touching this:**
* `tools/kvroof/kv_roofline.py` and `kvroof.py` (the source).
* `docs/00-verified-facts.md` §9.1 (config), §8.1 (bandwidths), §8.10 (bit-width ladder), §9.8 (operating point).
* `docs/01-architecture.md` §4.2 (operating point, capacity decision).
* `docs/04-memory-tiering.md` §6.1 (Class A/B/C ladder and the Class B price table — verified against section H).
* `docs/09-kv-engine-architecture.md` §5.1 (the weight-format table with the W3 row).
