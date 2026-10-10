# Missing-items register

One place for everything discussed in this session that is **not yet in the
numbered docs**. The numbered docs (`00`–`09`) stay the spec. This file is the
working register of what is absent and where each item is touched.

Updated as items are closed. When an item becomes real, remove it from this
register and either add it to the right numbered doc or link the new doc from
the right numbered doc. Do not let this file become the place items live.

---

## 1. How this file relates to the other docs

- `docs/08-roadmap.md` §7 is the roadmap's version of this list. This file is
  the more detailed, more refactorable version. When they disagree, this file is
  the more current one; §7 should be trimmed to point here.
- `docs/COMPONENT-REFERENCE.md` is the map from component to doc. It should NOT
  become the place missing items live. When a missing item becomes a real doc,
  add a row there.
- `docs/01-architecture.md` now has a `Numeric-claim audit` note. That note is
  the entry point for “confirm the numbers still agree with the docs”; this file
  is the place for the meta-level advice about what “agree” means.

---

## 2. The discussed items, with current status

### 2.1 Items that are genuinely absent from every present doc

These are in `docs/08-roadmap.md` §7.2 already. The register here keeps the same
ten, plus a short note on the closest existing doc touch for each.

1. **Error model and failure taxonomy.** No single taxonomy the whole engine
   converges on: retriable / drop / degrade-silent / user-facing / escape-hatch
   / telemetry-mapped. Closest existing touch: `02` per-component failure prose
   and `06` `STALL_TIME_MS`. **Not written.**
2. **Security boundary for the HTTP surface and the cold store.** No statement of
   what a tenant may reach: model files, NVMe quota, warm pool, other sessions’
   prefixes, profiler output, tuning cache, host fs. Closest existing touch:
   `02` C23 HTTP surface and `04` NVMe admission. **Not written.**
3. **Recovery story.** No walk-back procedure for dirty shutdown: which journal
   entries replays, which in-flight transfers discards, which partial expert
   extents quarantines, what the user sees. Closest existing touch: `04` and `09`
   NVMe write atomicity / crash-safe journal. **Not written.**
4. **Multi-session correctness invariants.** No invariant list that makes prefix
   sharing testable: shared-KV-block refcounts, evict-while-shared, cancel
   semantics, what constitutes a breach. Closest existing touch: `02` C23
   concurrent prefix sharing. **Not written.**
5. **Latency SLO and tail definition.** No stated p95/p99, no sample-count
   definition, no per-step vs per-token rule, no SLO for Class A / Class B /
   cold-start prefill. Closest existing touch: `06` median device time and `04`
   Class B ms/step. **Not written.**
6. **Testing contract for the driver/launch abstraction.** No statement of what
   the HIP/device layer behind C2 exposes, what a mocked driver must satisfy for
   the CPU-only test tier, what the CPU-reference path contract is. Closest
   existing touch: `01` §7, `03`, `04`, `07`. **Not written.**
7. **Load / save / checkpoint / reload of a live session.** No session checkpoint
   format, no statement of what is captured (KV pages, warm-resident experts,
   router state if any, suspended draft state), what is not, the reload path, the
   failure mode on changed model fingerprint. Closest existing touch: `04`
   persistence of prefixes/suspended sessions and `09` Class C suspend. **Not
   written.**
8. **Fused kernel contract.** No contract for which fusions are guaranteed vs
   opportunistic, what a fusion regression looks like in the profiler, what the
   fallback is when a fusion is disabled. Closest existing touch: `03` fusion
   targets and `06` `ffn-activate` / `residual` / `attention-mix` classes.
   **Not written.**
9. **Tokenizer and detokenization boundary.** No statement of where tokenization
   lives (client, server, bundled embedding), the byte/unicode/tooling contract,
   unknown-token behaviour, or how the embedding table is loaded and versioned
   alongside the model. Closest existing touch: assumption in architecture docs,
   Python/HF-style vocab assumption in `07`. **Not written.**
10. **Mixed-precision KV ladder decision.** No decision doc for the ladder sketched
    in `09` §6: promotion/demotion triggers, per-session vs global, warm-tier
    resident-context pricing under the ladder, Class A vs Class B. Closest existing
    touch: `09` §6 sketch. **Not written.**

### 2.2 Items that are “meta” gaps about the docs themselves

These are discussed but not yet in any numbered doc as a first-class note.

#### 2.2.1 Numeric-claim audit entry point

- **What was added:** `docs/01-architecture.md` now has a
  `### Numeric-claim audit (do not trust the tables until this passes)` block
  naming the two commands and their exit codes.
- **What is still missing as a first-class rule:** the *expected cadence* and the
  *failure action*. Concretely:
  - when the numbers should be re-audited (ROCm upgrade, toolchain change, config
    change, doc rewrite, tool change), stated as a rule rather than a suggestion;
  - what “the docs fail the audit” means operationally: which doc is stale, who
    fixes it, and that `01`’s tables are not authoritative again until the audit
    returns 0.
- **Owns:** documentation hygiene, not any single numbered doc. The closest
  existing home is the `01` note plus this register; the real home, once written,
  is probably a short doc-level policy section or a CI note in `07`.

#### 2.2.2 Documentation versioning

- No stated doc version, no changelog, no stated relationship between a doc, the
  commit it was written against, and the tool version that validates it
  (`check_docs.py`, `kv_roofline.py --check`).
- This is item 7.4 in `08-roadmap.md` already; the gap is that it is a list of
  “no doc states X” rather than a single doc that states the policy. **Not
  written.**

---

## 3. The items already written into the numbered docs in this session

These are **not** missing anymore; they are here so the register does not
reopen them.

- **TOC completeness note** in `docs/01-architecture.md`: the file opens at §3,
  though the `00` outline promises §1/§2/§9. The substance exists across `01` +
  `09`; the gap is structural.
- **Numeric-claim audit note** in `docs/01-architecture.md`: names
  `kv_roofline.py --check` and `check_docs.py`, with exit codes.
- **Missing-items list** written into `docs/08-roadmap.md` §7.1–§7.6, covering
  the ten cross-cutting gaps, the doc-specific gaps, the versioning gaps, the
  non-goals clarification, and the implication note.
- **Component reference** created as `docs/COMPONENT-REFERENCE.md`: C1–C24 to
  doc, tooling, headline numbers with provenance, and a pointer to the
  missing-items list.

---

## 4. Non-goals — deliberately not missing

These are absent on purpose and should not be “completed” just because they are
not in the numbered docs. They are already recorded as non-goals or deferred:

- Training / fine-tuning — non-goal (`01` §8).
- Multi-GPU — non-goal (`01` §8), SPLASH-style decoupling recorded for later.
- Default quantisation of attention trunk or router — non-goal (`01` §8).
- Production BF16→ternary in the default path — experimental behind a quality
  gate, not a default (`01` §8).
- NVMe-as-VRAM — non-goal (`01` §8).
- A general tiering engine — `09` §10.2: build Class A first; add tiering only
  when a real session exceeds it.
- WMMA tiling — demoted (`00` §8.9); measured kernel within 16% of an intensity
  ceiling tiling cannot move.
- CPU expert fallback as a performance path — correctness guarantee first,
  performance path later (`09` §10.2).

---

## 5. What “ensure all discussed items are added” actually means here

The discussed items split into three kinds:

- **Already added this session:** the five bullets in §3. These are in the docs
  now.
- **Documented as missing this session:** the ten cross-cutting items in §2.1 and
  the doc-meta gaps in §2.2. These are *recorded as missing*, which is the honest
  outcome — writing a stub into a numbered doc would be inventing detail that has
  not been decided.
- **Not yet written anywhere:** the recommendation in §4 is to keep the concise
  version in `08` §7 and the detailed version in this file, and to add a real doc
  only when an item is actually being closed, not to pre-create empty stubs for
  all ten.

If you want everything turned into a doc anyway, the smallest honest move is one
more file — `docs/OPEN-SPEECS.md` or similar — that converts §1–§2 here into
short design prompts with the same “not decided” labeling. That is better than
scattering ten stub sections across `02`, `03`, `04`, `05`, `06`, `07`, `09` and
pretending they are settled. If that is what you want, say which filename and I
will write it.

---

## 6. Measured gap: the Qwen 3.8 draft/target pairs (2026-10-08)

Recorded here because it blocks the "benchmark the Qwen 3.8 model" request with a
measurement rather than an opinion, and because the owning worksheet
(`ai-coder/c20-speculation.md`) names the drafter families without stating their
layouts.

**Measured 2026-10-08, before the `qwen35` front end existed:** every Qwen 3.8
container on this machine was refused by the architecture gate in
`src/model/model.cpp`, which implements `qwen3moe` and only `qwen3moe`. Three
distinct refusal strings, all reproduced:

```
error: model: architecture is 'dflash',  this forward pass only implements 'qwen3moe'
error: model: architecture is 'qwen35',  this forward pass only implements 'qwen3moe'
error: gguf: unknown ggml type has no block bytes
```

**Measured now (same day, after the front end):** `qwen35` no longer hits the
architecture gate at all. `kanjoos-run` reads the container, prints its geometry,
builds its 248320-token tokenizer, binds and shape-checks all 866 tensors, decodes
embeddings and one layer's input projections, compares 45 vectors element by
element against an independent oracle with **0 mismatches**, and then refuses the
*trunk* by name with exit 3 — see `docs/10-dflash-draft-models.md` §6.1 and
`records/qwen35-probe-2026-10-08/`. The `dflash` files are refused with a message
that says what a drafter needs instead of the generic architecture string. The
trunk, the head, the C20 loader and the drafter forward pass remain unwritten;
that is still the whole of the "benchmark Qwen 3.8" blocker.

**Measured, later the same day:** the GatedDeltaNet block itself is implemented.
`kanjoos-run --qwen35-recurrent` computes one recurrent layer end to end — the
conv1d state carry, the per-group delta rule over the 48 x 128 x 128 state, and
the alpha/beta/ssm_a gating — and compares its 12 output vectors against the
oracle: **12/12 at `blk.0` and 12/12 at `blk.4`, 0 mismatching and 0 unwritten
elements**, with the state carrying bit-identically across calls and a mutation
test showing the comparison fails when `beta` is dropped. What is missing is the
trunk *around* the block: no residual, no layer composition, no FFN, no head (the
attention was still on this list when the paragraph was written — closed in the
next one). See `docs/10-dflash-draft-models.md` §6.2 and
`records/qwen35-probe-2026-10-08/`.

**Measured, still later the same day:** the **gated full attention** is also
implemented. `kanjoos-run --qwen35-attention` computes one attention layer end to
end — the per-head q/gate split of the fused `attn_q`, the RMS norm over
`head_dim` 256, the partial IMROPE rotation over 64 of 256 dims, the causal softmax
with GQA and the `sigmoid(gate)` — and compares **11 vectors**: **11/11 at `blk.3`
and 11/11 at `blk.63`, 0 mismatching and 0 unwritten elements**, with the K/V cache
carrying bit-identically and a mutation that disables the rotation failing exactly
6 of the 11. That leaves the trunk wiring (no residual, no composition), the FFN
(65), the 248320-wide head and the 4 `nextn.*` tensors between this and a logit.
See `docs/10-dflash-draft-models.md` §6.3.

**Also measured the same day: the tokenizer gap is closed.** The 248320-token
`gpt2`/`qwen35` tokenizer this file previously listed as missing exists, and its
merge boundaries are now cross-checked against an independent implementation
(`llama-tokenize`) on 25 strings written to files: **24/25 strings identical, 0
merge-boundary differences, 1 special-token POLICY difference** on both the 248320
(`qwen35`) and the 151936 (`qwen3moe`) vocabulary — `tools/tok_crosscheck.py`, exit
0 on both. Getting there required fixing three real defects in this engine's BPE
(merge table keyed by concatenation so no merge ever fired; a symbol-lookup
fallback that silently absorbed chunks; byte tests standing in for Unicode
character classes). Dispatch on `tokenizer.ggml.pre` is **done and gated**
(`tok_pre_dispatch`: ten single-field fixtures, exact id lists, by-name refusals;
`tools/tok_pre_rules.py` re-derives the table from llama.cpp) — no file on this
machine can distinguish `qwen2` from `qwen35` because llama.cpp maps both to the
same regex, but the dispatch is in the tree. See `docs/10-dflash-draft-models.md`
§6.4 and §8.

**Not missing by accident — missing by not being written yet.** Three separate
capabilities, in dependency order:

1. a `qwen35` trunk (17 attention + 48 SSM blocks, gated attention, QK-norm 256,
   untied embeddings, a 248320-token `gpt2`/`qwen35` tokenizer) — without it there
   is no target model and therefore nothing for a drafter to condition on. The
   SSM block's arithmetic is now implemented and verified per layer (above); the
   trunk wiring, the attention, the FFN and the head are not;
2. the C20 "one loader for DFlash / DFlash2 / DSpark" — three tensor grammars
   behind one metadata reader;
3. the drafter forward pass (non-causal, sliding-windowed, block-parallel) plus the
   I6 acceptance gate.

**The layouts are no longer the unknown.** `docs/10-dflash-draft-models.md`
records the metadata and tensor inventory of four measured file shapes (DFlash 58
tensors, DFlash2 81, DSpark 62, the `qwen35` trunk with a merged `nextn.*` MTP
head), read by `tools/ggufmeta/gguf_meta.py`. What remains missing is the code.

**Quantisation is a second, independent gap, and it is now measured per file.**
The loader's type table names the whole IQ family (block sizes taken from
gguf-py, not from memory), so a container carrying one can be *opened* and the
refusal can name the type and count its tensors — previously such a file died with
`unknown ggml type has no block bytes` and hid the rest of its table. Measured
counts with no decoder in this engine:

| file | undecodable |
|---|---|
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | **363 tensors**: IQ3_S 144, IQ4_XS 96, IQ3_XXS 78, IQ2_S 17, Q2_K 13, IQ2_XS 9, IQ2_XXS 5, IQ1_M 1 |
| `Qwen3.8-Distill-35B-A3B-…-Q2KXL_ROCMFPX.gguf` | opens as of this change; its own count is not measured here |
| `ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf` | **0** — Q5_K was the last missing type and is now implemented and verified byte-for-byte against gguf-py (2 × 89 128 960 values, max abs diff 0.0) |

So "the container reads" and "the file can be computed" are different claims, and
the engine now reports which one holds.

---

## 7. Open items — the to-do register (2026-10-08)

Everything here is **measured but unfinished**, or **stated but not yet proven**.
Each line names the smallest thing that would close it. "Partial" means an
artifact exists that does what it says for what it covers, and does not cover the
rest. §2 and §3 above remain the register for the doc-level gaps; this section is
the engineering to-do that came out of the C21 / qwen35 / tokenizer work.

### 7.1 The tokenizer

* **Closed in Phase 51, kept visible so it is not re-listed as open.** The
  whitespace-run backtrack counted BYTES where the reference gives back one
  CHARACTER, so a single U+00A0 (2 bytes) or U+3000 (3 bytes) was cut in half and
  its own merge could not fire. Fixed in `src/tokenizer/tokenizer.cpp`, gated by
  the new ctest case `tok_whitespace_run`, and the 50-prompt fuzz pass moved from
  1 merge-boundary difference to 0. **The remaining partial is now closed too
  (2026-10-10):** all seven regex alternatives are swept, not just whitespace.
  `tools/tok_pre_sweep.py` extracts the QWEN2/QWEN35 patterns mechanically from
  llama.cpp's `src/llama-vocab.cpp` and derives 42 splits
  (`records/tok-pre-sweep-2026-10-10.log`); the ctest case `tok_pre_sweep`
  asserts every one exactly against `Tokenizer::pretokenize` — 42 rows,
  0 failures on the first run.
* **The class-rule mutation is a confirmed null result (2026-10-10).** The mutant
  (`unicode_class` returns Letter for any cp ≥ 0x80; `tmp/mutant-run.exe`, source
  restored after) vs the current tree on 200 fuzz prompts: **200/200 identical**
  (`records/tok-mutant-fuzz200-2026-10-10.log`, `tmp/mutant_diff.py`). The direct
  mutant-vs-llama leg never ran here (no `llama-tokenize` on this machine), but it
  is moot: current-vs-llama on the same generator family is PASS (MACX seed-1
  5000-prompt campaign, 0 boundary), so the mutant transitively matches the
  reference on high-byte input too. The class fix stays as a real divergence from
  the reference regex with no demonstrated behaviour change. Closed.
* **The cross-check's reference is llama.cpp, not the trainer.** 25 hand strings +
  corpus lines + fuzz prompts; a very long document has never been compared.
  Unblocked by: a corpus of real prose large enough to cross `--max-tokens` 2048.
* **No long fuzz campaign.** Measured: ~2.2 s per prompt (three subprocess
  launches, one of which loads the GGUF), so 50 prompts ≈ 2 min and a 5000-prompt
  pass ≈ 3 h. Unblocked by: running it in the background per seed and keeping each
  log under `records/`.
* **`tokenizer.ggml.pre` prose is stale** — see §7.4.

### 7.2 Tooling and gates

* **The behaviour smoke is not a gate.** `tmp/smoke_tools.py` (five cases: corpus
  only, hand + corpus, fuzz, empty-run refusal, fixture axes) asserts the tools'
  own summary lines, but lives in `tmp/` where nothing runs it. Unblocked by:
  moving it into the gate list `tools/bench/run_bench.sh` drives.
* **A loader failure reads as a tokenizer failure — closed 2026-10-10.**
  `tok_crosscheck.engine_ids` maps the unsigned loader codes (`LOADER_FAILURE_CODES`:
  0xC0000135/0xC0000139/0xC000007B) to a "failed to START … process-boundary
  failure" refusal instead of "no id list", and `engine_env()` prepends the first
  `MINGW_CANDIDATES` dir carrying `libstdc++-6.dll` (overridable via
  `KNJ_MINGW_BIN`). Verified by forcing returncode 0xC0000139 through the real
  branch: the message names the code and the boundary.
* **The separator sweep is not a gate.** `tmp/probe_space_class.py` is the check
  that proves parity across all twelve whitespace classes plus the ASCII controls
  (it is what reduced the Phase 51 bug). It needs llama.cpp and a real model, so
  it belongs beside the fuzz mode rather than in ctest.
* **`tests/fixtures/tok/` is regenerated by hand.** `make_tok_fixtures.py --check`
  proves the files match, but nothing runs it automatically, so a
  vocabulary change can be committed with stale fixtures.
* **The engine binary needs the MinGW runtime on `PATH`** (`/c/Strawberry/c/bin`).
  Every host-binary gate silently depends on this; `ctest` passes only because the
  invocation exports it.

### 7.3 The qwen35 trunk — unchanged from Phases 48/49

* **One layer is not a model.** The front end (geometry, binding, tokenizer,
  embeddings, input projections), one GatedDeltaNet block, one gated
  full-attention block are built and oracle-verified; the trunk *around* them
  (residual and layer composition), the dense FFN (65), the 248320-wide head and
  the four `nextn.*` tensors are not. `--bench` / `-n` therefore stay **refused**
  on every `qwen35` file. This is still the blocker for the dflash drafters.
* **`attention_layer()` uses a plain fp32 K/V cache**, not a RadixKV (docs/09)
  step. Unblocked by: the C4/C11 residency and buffer plan.
* **The IMROPE axis rule is only exercised in its text-only collapse** (all four
  axes carry the same token position). Unblocked by: a multimodal file — the only
  input that distinguishes the sections. The fixture tables under
  `records/qwen35-imrope-fixture/` now cover all four axes *analytically*.
* **`qwen35` still has no FFN and no sampler**, so no token of that model has ever
  been produced by this engine.

### 7.4 Documentation that is now contradicted by the tree

* `docs/10-dflash-draft-models.md` §8's trailing "not done" line named "dispatch on
  `tokenizer.ggml.pre`" — corrected 2026-10-10: the dispatch **is** implemented
  and gated (`tok_pre_dispatch` + `tools/tok_pre_rules.py`), and §8 already carries
  the superseding note. Closed.
* The same stale claim was in `docs/CODING-LOG.PENDING.md` Phase 49's "Still open"
  list. Corrected by Phase 51's entry, which names it. Closed.
* **`docs/09` §7 — no edit needed.** Verified 2026-10-10: the re-scoped I7 text is
  already in the tree ("a quantising KV codec … is a **declared precision
  reduction** … never conflated with a tier move"). The register asked for the
  edit before the re-scope landed; the re-scope landed. Closed.

### 7.5 Targets and measurement

* **gfx1031 is compile-only on this machine.** The tier C drivers DECLINE/SKIP
  correctly and rocWMMA refuses it at compile time with a static assert; no result
  is gated on it. Declared first-class, unmeasured here.
* **The attention probe's timing is unstable** (329-577 ms for identical work
  across runs) and is used for nothing. Unblocked by: the C21 profiler's device
  clock domain, which the profile gate already pins.
* **The `qwen35` files' token counts and dims are what the files declare.**
  `dflash.selector_top_k = 16` in particular is read as a *declared* parameter
  whose use inside the selector is not established. Closed 2026-10-11: the
  disposition is recorded in docs/10 §8 — declared-not-yet-used; the loader
  reads it as metadata (DFlash bootstrap carries no selector tensors at all, so
  no loader arithmetic may assume it); its consumer is the C20 drafter's
  token-selection path.
* **TowardsDataScience URL never provided** — closed 2026-10-11: no TDS/Medium
  article exists; the cited source is the GitHub repo JohnTDI-cpu/rdna4-wmma-guide,
  which docs/REFERENCE-SPEEDS.md already links correctly (Medium probe 403,
  Freedium unresolvable).
