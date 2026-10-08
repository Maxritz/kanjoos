# Worksheet — the W3 g128 operating-point decision

**What it is:** the single most important derived decision in the plan. The weight format stops being a bandwidth
optimisation and becomes a **capacity** decision, because P0-7 removed the last large lever except the weight format
(by showing routing locality is too weak to make partial expert residency worthwhile — evicting experts to buy KV costs
3–9 ms/token for less capacity than a W4→W3 repack frees for free). Same 3 B/group metadata rule as
`tools/bench/gemm_w4.hip` (fp16 scale + uint8 zero-point), whatever the bit width (`kvroof.pack_bytes_per_weight`).
16 GiB VRAM, 0.75 GiB workspace.

**Source of truth for the decision:** `docs/01-architecture.md` §4.2 (the decision narrative + the two tables) and
`docs/00-verified-facts.md` §9.8 (the full five-pack table with hit-rate and miss-cost columns that depend on the measured
P0-7 curve). The derived geometry (bank, resident, KV budget, FP16/FP8 context, experts resident) is audited by
`tools/kvroof/kv_roofline.py` section K (`--check`, exit 0 = 71 claims agree, including all five packs).

**The five-pack table (verbatim, `01` §4.2 / `00` §9.8):**

| pack | B/weight | expert bank | resident | KV budget | FP16 context | FP8 context | experts resident | LRU hit | miss cost |
|---|---|---|---|---|---|---|---|---|---|
| W4 g128 | 0.5234 | 14.13 GiB | 15.31 GiB | **0** | — | — | 127 / 128 | 99.9% | 0.1 ms |
| **W3 g128** | **0.3984** | **10.76 GiB** | **11.79 GiB** | **3.46 GiB** | **37.8 K** | 75.6 K | 126 / 128 | 99.8% | 0.2 ms |
| W2 g128 | 0.2734 | 7.38 GiB | 8.27 GiB | 6.98 GiB | 76.2 K | 152.4 K | 123 / 128 | 99.3% | 0.5 ms |
| W4 g64 | 0.5469 | 14.77 GiB | 15.97 GiB | 0 | — | — | 122 / 128 | 99.1% | 0.6 ms |
| W8 g128 | 1.0234 | 27.63 GiB | 29.37 GiB | 0 | — | — | 65 / 128 | 90.7% | **6.6 ms** |

(The hit-rate and miss-cost columns depend on the measured P0-7 curve in `tools/route/route_target-gates.json` — they are
**not** derived from the config and are **not** in `kv_roofline.py --check`'s audit. The tool says so. They are verified by
`check_docs.py` §9.6, which reads the route artifact. If the route artifact is stale, §9.6 would flag it — and it does not.)

**The decision (verbatim, `00` §9.8 / `01` §4.2):** **W3 g128 is the operating point.** It is the only row whose FP16 context
fits under the model's own 40,960-token ceiling while keeping essentially the whole expert bank resident. W2's extra capacity is
largely unusable — the model cannot attend that far — and it buys 0.3 ms/token of misses for it. W8 is the instructive failure: it
*halves* the bank, and the measured hit-rate curve then charges 6.6 ms/token for it, which is the §9.6 miss cost arriving on schedule.

This is a **capacity decision, not a bandwidth one**, and the ordering is the whole point:

| what it costs to buy 1 GiB of KV | |
|---|---|
| W4 → W3 repack | **free** in time (0.2 ms/token), 3.46 GiB |
| evicting 6 experts/layer | 6–9 ms/token, for less capacity than the repack |

Note what is **not** on this table: any codec that changes values. Per §9.7 a quantising codec is a declared precision reduction,
and its capacity is orthogonal to this decision — it multiplies whatever budget the pack leaves, behind a quality gate.

**Why W3 and not W2 (verbatim):** W2's extra capacity (6.98 GiB KV, 76.2 K FP16 context, 152.4 K FP8) is largely unusable because the
model's own ceiling is 40,960 tokens — it cannot attend 76 K or 152 K tokens. So W2 buys 0.3 ms/token of misses for capacity the model
cannot use. W3 is the only row that both fits under the ceiling and keeps essentially the whole bank resident (126/128). W4 leaves zero
KV budget. W8 halves the bank and the curve charges 6.6 ms/token for it.

**Why W3 and not W4 (verbatim):** W4 leaves zero KV budget at 16 GiB (15.31 GiB resident before workspace). So on a 16 GiB card, W4 is a
capacity dead end — not a bandwidth problem, a capacity problem. W3 frees 3.46 GiB of KV budget for free (0.2 ms/token extra misses, which
is well under the 3–9 ms/token the §9.6 curve charges for evicting experts), and that 3.46 GiB is 37.8 K tokens of FP16 context — under the
40,960 ceiling, usable context.

**Coder implications (what this decision forces on every component):**
* **C5 (expert compiler):** the operating point is W3 g128, but W4 is the **first implemented pack** (Phase 2 builds W4 end to end; W3 is a
  one-constant repack on the same kernel — Phase 2b). Do not delay Phase 2 waiting for W3; build W4, then repack. The repack is cheap precisely
  because Phase 2 already landed the compiler, the manifest, and the GEMM. See `docs/08-roadmap.md` §Phase 2b.
* **C17 (expert GEMM):** must support W4 as the first pack, and W3 as the same kernel with a one-constant change (0.5234 → 0.3984 B/weight).
  The GEMM template instantiation and the pack/unpack constants change; the kernel does not. The 0.625 instr/MAC `v_dot4_i32_i8` path (W4) and
  the 0.250 instr/MAC `v_dot8_i32_i4` path (nibble-packed, both operands) are the measured options; the `v_dot8_i32_i4` path was rejected for W4
  because it needs 4-bit activations (81.5 dB SNR penalty) — see whether that rejection lifts for W3 (it does not automatically; W3 is 3-bit payload,
  and the SNR question for 3-bit payload is the half that has not been attempted).
* **C8/C9 (slots/residency):** with W3 g128 as the operating point, 126/128 experts fit in VRAM at 16 GiB. So expert residency is effectively
  "all or nothing, and all fits." Do not build a partial-expert-eviction-as-cache path — the §9.6 verdict rules it out, and the operating point makes
  it unnecessary at 16 GiB. (At lower VRAM profiles — 6/8/12 GiB — the operating point shifts; C22 sizes the slot pool per profile.)
* **C14/C15 (KV):** with W3 g128, the KV budget is 3.46 GiB = 37.8 K FP16 / 75.6 K FP8 context. So Class A at 16 GiB with the operating point is a
  37.8 K FP16 resident context. Do not design Class A around W4 (zero budget) or around an assumed codec that changes the budget without changing the pack.
* **C12 (admission):** the context-class ladder (A/B/C) and the refusal path are sized against the operating point. Class A at the operating point =
  37.8 K FP16 context resident. Class B (spilled) is priced at 13.4–14.7 GB/s and is disqualifying for chat (8 K cold span = 54.8–60.1 ms FP16 /
  27.4–30.0 ms FP8 — the whole step budget before any FLOP). Class C is suspended. Admission must carry the class and the price, not the word "slow".
* **C19 (CPU fallback):** with W3 g128 at 16 GiB, 126/128 experts fit in VRAM, so the expert miss rate is ~1.6% (2/128) — the §9.6 curve says at 126
  slots the miss cost is ~0.2 ms/token. The CPU fallback is still needed (it is the escape hatch when a slot is not ready in time, or at lower VRAM profiles),
  but the operating point means the common case at 16 GiB is not expert misses — it is the KV cost of whatever context the session is admitted with.
* **C22 (budget):** the VRAM×RAM profile matrix is sized against the operating point. At 16 GiB with W3, the KV budget is 3.46 GiB; at lower VRAM, the
  pack may need to drop to W4 (zero KV budget, all context via Class B/C) or the model may not fit at all (6 GiB — see `04` §2.1, that is a workload, not
  a failure). The profile is a starting point overridden by measurement (C3 autotuner), but the operating point is the reason 16 GiB/W3 is the headline row.

**The SNR gate (verbatim, from `08` §Phase 2b and `00` §8.10):** the 00 §8.10 caveat records an 81.5 dB SNR penalty that killed the int4 activation
path (`gemm_w4.hip`). A real W3 design must beat that SNR penalty on the target prompts before the repack is admitted as the operating point. Bytes are the
easy half of W3; accuracy is the half that has not been attempted. So the operating point is **conditional** on the SNR gate — if W3 cannot beat 81.5 dB on the
target prompts, the operating point stays W4 and the KV budget stays zero. This is the reason Phase 2b is gated (exit criterion: matches W4 reference within
tolerance AND SNR gate passes), not a free repack.

**What a coder must not do:**
* Do not treat the operating point as a bandwidth decision (e.g. "W3 is faster because it moves fewer bytes"). It is a **capacity** decision — W3 is chosen
  because it is the only pack that leaves usable KV budget under the ceiling while keeping the bank resident. The ms/token effect of the repack is 0.2 ms (free),
  not the reason.
* Do not treat W2 as "better than W3 because more KV budget." W2's extra budget is unusable (model cannot attend 76 K / 152 K tokens). More unusable budget
  is not better.
* Do not build a partial-expert-eviction path as if it were a cache. The §9.6 verdict rules it out (3–9 ms/token to free a few GiB, for less capacity than the
  repack), and the operating point makes it unnecessary at 16 GiB.
* Do not skip the SNR gate for W3. The operating point is conditional on beating 81.5 dB on the target prompts. If that is not measured, the operating point
  stays W4.
* Do not quote the hit-rate and miss-cost columns as if derived from the config. They come from the route artifact. A coder implementing the operating point must cite
  `00` §9.8 (which cites the route artifact) and `kv_roofline.py --check` section K (which audits the derived geometry and explicitly excludes the hit-rate/miss-cost
  columns).

**Acceptance test for this decision (what would confirm it is still the right call):**
* Re-run `tools/route/route_locality.py` (or confirm the artifacts are current) — the hit-rate and miss-cost columns must still support the decision (i.e. the
  99.8% LRU hit at 126 slots and the 0.2 ms miss cost must still hold). `check_docs.py` asserts the §9.6 numbers against the artifact; if the artifact is stale or the
  curve shifts enough to change the decision, §9.6 would flag it.
* Re-run `tools/kvroof/kv_roofline.py models/qwen3-30b-a3b/config.json --check` — exit 0 confirms the derived geometry still matches the doc.
* Re-run `tools/check_docs.py` — exit 0 confirms the doc still matches the tools.
* The SNR gate for W3 is **not** yet measured — so the operating point is currently a **conditional** decision: "W3 g128, provided it beats 81.5 dB on the target
  prompts." Until that is measured, the headline operating point is conditional.

**Files a coder should read before touching anything that cites the operating point:**
* `docs/01-architecture.md` §4.2 (the decision, both tables).
* `docs/00-verified-facts.md` §9.8 (the full five-pack table), §9.6 (the P0-7 curve the hit-rate/miss-cost columns come from), §8.10 (the 81.5 dB SNR caveat).
* `docs/08-roadmap.md` §Phase 2b (the W3 repack milestone, the SNR gate, the exit criterion).
* `docs/04-memory-tiering.md` §6.1 (the context-class ladder sized against the operating point, the Class B price table).
* `docs/09-kv-engine-architecture.md` §5.1 (the weight-format table with the W3 row in the KV engine doc).
* `tools/kvroof/kv_roofline.py` section K (the derived geometry, audited).
* `tools/route/route_target-gates.json` (the artifact the hit-rate/miss-cost columns come from — not a config derivation).
