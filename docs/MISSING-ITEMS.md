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
