# Worksheet — tools/route (existing, P0-7 routing locality)

**What it is:** `tools/route/route_locality.py` — P0-7. Measures how much of a token's expert
selection survives to the next token, from real router weights, and converts the misses into
ms/token at the measured PCIe bandwidth. Two independent sources, because neither alone is
trustworthy.

**Why it is the pivot for the whole plan (from `00` §9.6, `01` §4.2):** P0-7's answer is "routing
locality is weak, and weak in the way that matters — there is no small hot set." That answer removes
the last large lever except the weight format, and turns the weight format from a bandwidth optimisation
into a **capacity** decision. Any component that assumes a small hot set exists (partial expert
residency as a cache, an expert-spill knob, evicting experts to buy KV) is building on a false premise.

**The two sources (verbatim from `00` §9.6):**
* **Source 1 — self-consistent.** `Qwen1.5-MoE-A2.7B-Chat` (already on disk), run end to end on 768
  tokens of real text with hooks on all 24 gates. Its gates saw its own hidden states. 60 experts,
  top-4, 24 layers.
* **Source 2 — the target's own routing function.** The **real 48 gate matrices of Qwen3-30B-A3B**,
  pulled by HTTP range out of a 61 GB checkpoint (25.2 MB, `tools/route/hf_range.py`), applied to real
  hidden states from Qwen3-1.7B — same family, same hidden width 2048. **This is a proxy and is labelled
  as one everywhere it appears.** It is not the target model's own forward pass, and no claim in this
  section depends on it alone.

**Source 2 artifact (`tools/route/route_target-gates.json`, sha `477d409177424236dab5c4cd8e36a0bce8b788e62389d146a62c9d3853fd861c`):**
* source: `Qwen/Qwen3-30B-A3B` | kind: `target-gates` | caveat: PROXY (real gate matrices, real hidden
  states from `knj_p0…`, not the target's own forward pass).
* aggregate: used_med = 115.0 / 128; ent_med = 5.48 (max 6.54); reuse_med = 0.4086 (max 0.8134);
  c25_med = 83.0 % of selections in the top 25% of experts.
* LRU hit curve (median over layers, static slots → hit): 8→0.332, 13→0.472, 26→0.677 (20%),
  38→0.789, 51→0.862 (40%), 64→0.904 (50%), 77→0.938, 90→0.957 (70%), 115→0.981 (90%),
  128→1.0 (100%). min_slots_lru to reach 90/95/99% = 64/90/128.
* **Top-k reuse from t-1 = 0.4086 → new experts demanded per token = 1 − 0.4086 = 0.5914 = 59.1%.**
  This is the number the doc quotes as "59.1%" (source 2) and "80.5%" (source 1). Both are cited to
  `route_local.json` (source 1) and `route_target-gates.json` (source 2), and `check_docs.py` asserts
  both against those exact files.

**Source 1 artifact (`tools/route/route_local.json`, sha `1946e995af5fd10fa2c2368451adb3724ec71c86b8d9663c5b4543874c8705`):**
* source: `Qwen1.5-MoE-A2.7B-Chat` | self-consistent forward, 24 gates, 768 tokens.
* aggregate: used_med = 60.0 / 60; ent_med = 5.81 (max 5.86); reuse_med = 0.1952; c25_med = 37.0 %;
  reuse_min/max = 0.086 / 0.250.
* **Top-k reuse from t-1 = 0.1952 → new experts demanded per token = 1 − 0.1952 = 0.8048 = 80.5%.**
  This is the number the doc quotes as "80.5%" (source 1) and "19.5%" reuse (source 1). `check_docs.py`
  asserts `100*(1 - r1["reuse_med"])` within 0.15 pt of 80.5 and `"80.5%" in d00` — and it passes,
  because `route_local.json`'s reuse_med = 0.19524 → 80.5%.

**Source 1 replication artifact (`tools/route/route_local_prose.json`, sha `95ca5d11430b82c3696a4597d01590cdc892ec2ab31b41feb1b2778aee6ea8`):**
* This is the **5-language replication run** of the same model (Qwen1.5-MoE-A2.7B-Chat) on Russian and
  Chinese prose (`00` §9.6.1). It is **a different run** from `route_local.json`. Its aggregate:
  reuse_med = 0.2047 → new-expert = 79.5%; c25_med = 36.0%; ent_med = 5.83. That is why comparing the doc
  (80.5% / 37%) against `route_local_prose.json` looked like a 1-pt mismatch — it is, because they are different
  runs. **The doc and the machine audit both use `route_local.json`; do not conflate the two.**

**Source 2 headline figures (verbatim, `00` §9.6):** experts ever selected, median layer = 115/128; routing
entropy median (max) = 5.48 bits (7.0); selections in top 25% of experts = 83%; top-k reused from previous
token = 40.9%; **new experts demanded per token = 59.1%.**

**Source 1 headline figures (verbatim, `00` §9.6):** experts ever selected, median layer = 60/60; routing
entropy median (max) = 5.81 bits (5.9); top-k reused from previous token = 19.5%; **new experts demanded per
token = 80.5%.**

**LRU hit-rate table (verbatim, `00` §9.6, source 2 of 128, source 1 of 60):**
| slots | source 2 (of 128) | source 1 (of 60) |
|---|---|---|
| 20% | 67.7% | 35.2% |
| 40% | 86.2% | 55.3% |
| 50% | 90.4% | 63.5% |
| 70% | 95.7% | 78.3% |
| 90% | 98.1% | 92.1% |
| 100% | 100% | 100% |

(The doc's source-1 LRU columns are asserted by `check_docs.py` at slots 12/24/30/42/54 → 35.2/55.3/63.5/78.3/92.1%,
against `route_local.json` — and they pass. The doc's source-1 LRU@20/40/50/70/90/100% in the §9.6 table are
26/51/64/90/115-equivalent points on the same curve. Note source 1's grid is 4/6/12/18/24/30/36/42/48/54/60;
the doc's "20%/40%/50%/70%/90%/100%" are nominal labels on that grid's nearest points.)

**The miss-cost arithmetic (verbatim, `00` §9.6):** one missed expert in one layer = 2,469,688 bytes at W4;
at 13.4–14.7 GB/s = 168–184 µs; a 48-layer decode step pays **8.1–8.9 ms for a single missed expert per token**,
against 7.68 µs of arithmetic at 100% of peak (`00` §8.3) — a factor of ~1,100.

**LRU hit-rate → ms/token table (verbatim, `00` §9.6):**
| LRU hit rate (slots) | ms/token on the wire (13.4–14.7 GB/s) |
|---|---|
| 100% (128) | 0.0 |
| 95.7% (90) | 2.8–3.1 |
| 90.4% (64) | 6.2–6.8 |
| 86.2% (51) | 8.9–9.8 |
| 67.7% (26) | 20.8–22.8 |

**Verdict (verbatim, `00` §9.6):** evicting experts to buy KV context loses in both directions: it costs
3–9 ms/token to free at most 6 GiB, and 6 GiB buys less context than a single W4→W3 repack frees for free.
**Expert residency is effectively all-or-nothing, and the W4 pack is exactly what makes all-or-nothing impossible
on a 16 GiB card.** The remaining lever is the weight format — W3 frees ~3.5 GiB, W2 frees ~6.6 GiB (`00` §8.10) —
which turns it from a bandwidth optimisation into a capacity decision. **Class B is not viable for experts** — same
verdict as for KV, reached independently, by the same wire.

**5-language replication (verbatim, `00` §9.6.1):** `route_target-gates_{en,ru,zh,es,ar}.json` + `route_local_prose.json`.
The hit-rate curve moves by at most 6 points across five languages and two script families — far less than the
distance between the curve and anything that would make partial residency attractive. Chinese prose is the worst case
(60.1% / 86.3% / 93.8% at 20/50/70%) and even it reaches 90% at ~60% of the bank. The baseline (this repo's own
technical docs) was if anything flattering. **The verdict is a property of MoE routing, not of the text.** The
self-consistent source replicates too: Qwen1.5-MoE-A2.7B on Russian and Chinese prose scores 0.205 reuse (vs 0.195
on the docs) and an LRU curve within 2 points of its own technical-corpus curve at every slot count.

**Acceptance test / what "done" means here:** the artifacts exist, are hashed in `models/README.md` (or should be —
confirm the route artifact hashes are recorded), and `check_docs.py` asserts the doc's P0-7 numbers against them
(source 2 against `route_target-gates.json`, source 1 against `route_local.json`). Exit 0 = the doc matches the
artifacts it cites.

**What a coder must not misstate:**
* Do not say "P0-7 measured a small hot set" — it measured the opposite.
* Do not build an "evict experts to make room for KV" path as a cache — that is the losing direction the verdict
  rules out.
* Do not treat the source-2 proxy as the target model's own forward pass. It is labelled as a proxy everywhere it
  appears, and no claim depends on it alone (source 1 is the self-consistent cross-check).
* Do not conflate `route_local.json` (doc's source 1, 80.5% / 37%) with `route_local_prose.json` (replication run,
  79.5% / 36%). They are different runs of the same model.
* Do not quote the ms/token table as if it were stable to the last digit — PCIe noise is 2–4% (`00` §8), and the
  table is derived from a range (13.4–14.7 GB/s). Quote the range, not a point.

**Files a coder should read before touching this:**
* `tools/route/route_locality.py` and `tools/route/hf_range.py` (the source).
* `docs/00-verified-facts.md` §9.6 and §9.6.1 (the measured result and the replication).
* `docs/01-architecture.md` §4.2 (how P0-7's answer turns the weight format into a capacity decision).
* `docs/04-memory-tiering.md` §6.1 (Class B not viable for experts — reached independently by the same wire).
* `docs/08-roadmap.md` — Phase 0 table (P0-7 CLOSED), risk register (expert reuse low → medium impact, mitigation:
  RAM residency, CPU fallback, speculation, honest storage-bound reporting).
