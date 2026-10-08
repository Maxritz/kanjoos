# Worksheet — C5 Expert compiler (offline)

**What it owns:** canonical checkpoint → kernel-native packed expert store.

**Why it exists (from `02-components.md` — C5 section):** the expert weights are never moved in BF16. They are compiled once,
offline, into the kernel-native packed format (W4 by default, W3 as the operating point), with a manifest that records per-expert
quality and routing loss, so the runtime (C17) consumes them kernel-native and the residency manager (C9) and transfer engine (C11)
move the packed bytes — no dequantise-to-BF16 staging on the miss path (I2).

**Spec lines that must hold:**
* `02-components.md` — C5 section (verbatim: the pipeline, the manifest fields, the determinism/invariant, the BITCOS correction, done-when).
* `00-verified-facts.md` §7.6 (W4 pack and group size settled: nibble-packed weights, **unsigned** nibbles `u = q + zp`, int8 activations,
  `v_dot4_i32_i4`... actually `v_dot4_i32_i8` with de-interleaved activation layout, 0.625 instr/MAC; the `v_dot8_i32_i4` alternate rejected
  because it needs 4-bit activations at 81.5 dB worse SNR; group size 128 for expert weights, 64 for KV).
* `00-verified-facts.md` §8.10 (the metadata rule: 3 B per group, whatever the bit width; the bit-width ladder; the 0.5234 / 0.3984 / 0.2734 B/weight
  values; the 81.5 dB SNR caveat that gates W3).
* `docs/01-architecture.md` §4.2 (operating point W3 g128; the repack is a one-constant change 0.5234 → 0.3984; W3 conditional on beating 81.5 dB SNR).
* `docs/08-roadmap.md` §Phase 2b (the W3 repack milestone: compiler constant change, kernel unchanged, manifest/store unchanged shape, SNR gate required, exit
  criterion: matches W4 reference within tolerance AND SNR gate passes).

**Pipeline (verbatim, `02-components.md` C5):**
```
BF16 checkpoint
  -> expert extraction (per layer, per expert, gate/up/down)
  -> activation-aware calibration (MoEQuant-style expert-balanced sampling: low-frequency experts MUST get their own calibration examples)
  -> candidate evaluation per expert: W8 / W6 / W4 / ternary
  -> precision allocation under the RAM+VRAM budget of the target profile
  -> packing into the kernel-native format for BOTH arches
  -> manifest + hashes
```

**Manifest fields (verbatim, `02-components.md` C6, since C5 writes them):** per expert: source hash, packed hash, precision, format, group_size,
nvme_offset, nvme_size, packed_hash, quality_loss, route_loss.

**Invariants (verbatim):** deterministic — same source hash gives byte-identical output. Reproducible from source hash alone.

**BITCOS correction (verbatim, `02-components.md` C5 and `08` Phase 7):** ternary is selected by **measured** size and kernel cost against five-trit
packing, not by assumption — the earlier spec's correction stands (BITCOS is ~3% *larger* than five-trit for CAT-Q Qwen3-30B-A3B once scales are counted).
Do not assume BITCOS is smaller.

**Acceptance test (verbatim "done when"):**
* Re-running the compiler on the same checkpoint produces an identical manifest hash.
* A corrupt source hash is detected rather than compiled.

**Gates:**
* **W4 is the first pack; W3 is the operating point but comes later (Phase 2b).** Do not build W3 into the first compiler milestone. Build W4, then repack.
  The repack is a one-constant change (0.5234 → 0.3984 B/weight) on the same kernel and the same manifest shape — but it is gated by the SNR gate (81.5 dB),
  and it is not the first milestone.
* **Ternary is experimental, quality-gated, not in the production path.** (`01` §8.4; `08` Phase 7.) Do not ship ternary as a default; it is an experiment with a
  quality gate.
* **Activation-aware calibration with expert-balanced sampling is required, not optional.** Low-frequency experts MUST get their own calibration examples. This is the
  "MoEQuant-style" requirement. A calibration that only samples high-frequency experts will over-fit the routing-biased subset and under-represent the long tail — and the
  routing tail is exactly where P0-7 says the experts are (59–81% new per token). Do not skip this or sample it naively.
* **The 81.5 dB SNR caveat gates W3.** A W3 design (3-bit payload, same 3 B/group metadata) must beat 81.5 dB on the target prompts before the repack is admitted.
  So the compiler's W3 path is conditional: it can pack W3, but the repack is not admitted as the operating point until the SNR gate passes. The compiler can produce W3
  packs; the operating-point decision that uses them is elsewhere (see `ai-coder/operating-point.md`).

**Coder notes / pitfalls:**
* The metadata rule is 3 B/group (fp16 scale + uint8 zero-point), whatever the bit width, matching `tools/bench/gemm_w4.hip`. This is the one number that must not drift:
  change it and the doc's INT4/INT8 rows (which include the 3 B/group) become wrong, and `kv_roofline.py --check` flags them. So the compiler's pack header must carry exactly
  3 B/group of metadata, and the manifest's per-expert `packed_size` must reflect payload + 3 B/group.
* The pack is kernel-native for **both** arches (gfx1201 and gfx1031). The same packed bytes are consumed by C17's arch-specific backends (WMMA on gfx1201, `v_dot8_i32_i4`
  / `v_dot4_i32_i8` on gfx1031). The pack format is the common written representation; the arch-specific unpack is in C17. So the compiler's output format is arch-independent
  (one packed representation), and C17's backends are arch-specific (how to unpack/consume it).
* Determinism: same source hash → byte-identical manifest. This is the reproducibility gate. A coder must not introduce a non-deterministic step (timestamp, random seed, iteration order
  that affects the output) into the pipeline. The manifest hash is the proof.
* Corrupt source hash detection: the compiler must verify the source hash before compiling, and refuse (not compile and produce a wrong manifest) on mismatch. This is the
  "corrupt source hash is detected rather than compiled" gate. A wrong manifest from a corrupt source is the same class of defect as a wrong cold index (`c4-loader.md`) — it produces
  wrong bytes with no error.
* The quality_loss and route_loss fields are what make per-expert precision allocation (W8/W6/W4/ternary per expert) possible. They are not decorative. A precision-allocation policy
  (Phase 7: mixed W8/W6/W4 per-expert precision) consumes them. So the compiler must compute them in a way that is comparable across experts and across precisions — same calibration
  corpus, same metric, same scale. A `quality_loss` that is not comparable across experts is decorative and will mislead a precision allocator.
* Group size: 128 for expert weights (the operating point and the default pack). The compiler's group size is a choice that affects both size (0.5234 vs 0.5469 B/weight for W4 g128 vs
  W4 g64) and the GEMM's unpack cost. The settled value is 128 for expert weights; do not make it a free parameter without re-deriving the size and the GEMM cost.

**Worked micro-example (host, sanity-check before device):**
Take a tiny BF16 checkpoint (or a fake one: a few experts, a few layers, known gate/up/down matrices), run the pipeline up to and including packing for W4 g128, and assert:
(1) the manifest's per-expert `packed_size` = (params / group_size) × (payload_bits/8) + 3 × (params / group_size) — i.e. payload + 3 B/group metadata, matching `kv_roofline.py`
  section K's `pack_bytes_per_weight` (W4 g128 = 0.5234 = 0.5 + 3/128); (2) re-running the pipeline on the same source hash produces a byte-identical manifest (same manifest hash);
(3) a corrupt source hash (flip a byte in the source) is detected and refused, not compiled; (4) the calibration step, when given a corpus that over-represents one expert, still
  allocates calibration examples to the low-frequency experts (expert-balanced sampling) — the compiler must not let a biased corpus produce a biased calibration; (5) the same source packed
  for gfx1201 and gfx1031 backends produces the same packed representation (arch-independent pack) — the arch-specific unpack is in C17, not in the compiler's output.

**Files a coder should read before starting:**
* `docs/02-components.md` — C5 and C6 sections (the pipeline, the manifest, the invariants, the BITCOS correction).
* `docs/00-verified-facts.md` §7.6 (W4 pack and group size), §8.10 (metadata rule, bit-width ladder, 81.5 dB caveat).
* `docs/01-architecture.md` §4.2 (operating point, one-constant repack, SNR gate).
* `docs/08-roadmap.md` §Phase 2b (W3 repack milestone, SNR gate, exit criterion).
* `ai-coder/operating-point.md` (the conditional operating-point decision the compiler's output feeds).
* `ai-coder/c17-expert-gemm.md` (the arch-specific unpack/consume side — the compiler's pack must match C17's expected format).
* `ai-coder/c4-loader.md` (the cold index — the compiler writes the manifest that the cold index/directory references).
* `tools/kvroof/kv_roofline.py` section K and `kvroof.pack_bytes_per_weight` (the canonical size function the compiler's pack must match).
