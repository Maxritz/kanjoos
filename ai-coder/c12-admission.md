# Worksheet — C12 Admission and eviction

**What it owns:** what gets demoted, and what is worth writing to NVMe at all.

**Why it exists (from `02-components.md` — C12 section; `04` §6.2):** the placement score (weights tuned from telemetry, structure fixed) decides what stays and what goes; the hard priority order
  (PINNED > ACTIVE > IN-FLIGHT > PREFETCHED > WARM > COLD) is never violated; the NVMe admission rule (`persist iff reuse_probability × recompute_cost > write_cost + read_restore_cost`) is the wear-control
  rule and is not negotiable; the initial policy persists shared prefixes, suspended sessions after a TTL, and repeatedly-reused cold segments, and nothing else.

**Spec lines that must hold:**
* `02-components.md` — C12 section (verbatim: the placement score, the priority order, the NVMe admission rule, the initial policy, the done-when).
* `04-memory-tiering.md` §5.3 (the eviction score — identical structure: 0.30×recency + 0.25×frequency + 0.20×reuse_prediction + 0.15×shared_prefix_value − 0.10×transfer_cost; the hard priority order
  PINNED > ACTIVE > IN-FLIGHT > PREFETCHED > WARM > COLD that no score may violate; in-flight and event-referenced pages are never evictable; shared prefixes get a bonus).
* `04-memory-tiering.md` §6.2 (the refusal path — admission does exactly one of three things: admit at shorter context, admit as class B with the price attached, refuse; no silent demotion A→B under load;
  no silent codec change; refusal is cheap. C12 is the admission that enforces this.)

**Interface (from `02-components.md` C12 and `04` §6.2):**
* Placement score (weights tuned from telemetry, structure fixed): `0.30*recency + 0.25*frequency + 0.20*reuse_prediction + 0.15*shared_prefix_value - 0.10*transfer_cost`.
* Hard priority order (never violated): PINNED > ACTIVE > IN-FLIGHT > PREFETCHED > WARM > COLD.
* NVMe admission: `persist iff reuse_probability * recompute_cost > write_cost + read_restore_cost`.
* Initial policy: persist shared prefixes, persist suspended sessions after a TTL, persist repeatedly-reused cold segments, persist nothing else.
* Admisssion outcome (from `04` §6.2): one of (a) admit at shorter context — client told the cap in tokens, truncation visible; (b) admit as class B with the price attached — only if the client opts in and the stated step time
  is within the client's own budget; (c) refuse. No silent demotion A→B under load; no silent codec change; refusal is cheap.

**Acceptance test (verbatim "done when"):**
* SSD write bytes per generated token stays under a configured bound on a prefix-heavy workload.
* Eviction never touches an in-flight or event-referenced page.

**Gates:**
* Depends on C9 (residency — C12's eviction acts on C9's residency table; C12's `demote()` is what C9's EVICTING_VRAM/EVICTING_RAM states transition through; the placement score reads C9's recency/frequency/reuse),
  C10 (prefetch — C12's reuse_prediction comes from C10's calibrated confidence), C11 (transfer — C12's transfer_cost is what C11 reports; the NVMe write is what C11's `submit_nvme_write` does), C14 (KV — C12's admission for
  KV pages; the class B price is what C12 attaches to a KV admission; the shared-prefix persistence is what C13/C14 produce), C13 (radix index — shared prefixes are what C12 persists; the shared_prefix_value in the placement score
  is what C13 identifies), C21 (telemetry — the "SSD write bytes per generated token" acceptance is a C21 number; the eviction-of-in-flight page attempt is a C21/counter event), C22 (budget — the NVMe quota and the admission
  threshold come from C22's profile), C24 (platform — the NVMe mechanism (Linux `io_uring`/`O_DIRECT`, Windows DirectStorage/IOCP) is C24; C12's NVMe writes go through C11/C24).
* The "eviction never touches an in-flight or event-referenced page" gate is the hard invariant. A coder must not let the placement score or any eviction decision select a page that is IN_FLIGHT or event-referenced. The priority order
  (PINNED > ACTIVE > IN-FLIGHT > PREFETCHED > WARM > COLD) has IN_FLIGHT above PREFETCHED/WARM/COLD, and "in-flight and event-referenced pages are never evictable" (`04` §5.3) is the rule. A coder must enforce this in C12's eviction
  selection — the score is a ranking within the evictable set, and the evictable set excludes PINNED/ACTIVE/IN_FLIGHT/event-referenced. The acceptance test (eviction stress under a routing-heavy workload) must include in-flight pages and assert
  none are evicted.
* The "SSD write bytes per generated token stays under a configured bound on a prefix-heavy workload" gate is the wear-control proof. A coder must be able to measure SSD write bytes per generated token (C21 number) on a prefix-heavy workload and assert it is under the bound.
  This is the thing that makes NVMe admission control real — the `persist iff reuse_probability × recompute_cost > write_cost + read_restore_cost` rule is what keeps the bound. A coder must not persist everything — the initial policy (shared prefixes,
  suspended sessions after TTL, repeatedly-reused cold segments, nothing else) is the wear-control rule. A coder must not persist transient one-shot completions, low-reuse-probability data, or anything whose write cost exceeds the recompute it would save.

**Coder notes / pitfalls:**
* The placement score's weights (0.30/0.25/0.20/0.15/0.10) are the structure; the values are tuned from telemetry (C21). A coder must implement the structure fixed and the weights tunable (from telemetry), not hardcode the weights as permanent. The structure is the
  invariant; the weights are the tuned parameter. A coder must be able to report the current weights and the telemetry that tuned them (C21).
* The priority order is never violated by the score. This is the rule that makes "in-flight and event-referenced pages are never evictable" structural, not a score outcome. A coder must implement the eviction selection as: (1) build the evictable set = everything except
  PINNED/ACTIVE/IN_FLIGHT/event-referenced; (2) score the evictable set with the placement score; (3) evict the lowest-scoring. The priority order is enforced by the set membership, not by the score — a PINNED/ACTIVE/IN_FLIGHT/event-referenced page has infinite effective cost.
  A coder must not let a high transfer_cost or low reuse_prediction overcome the priority order (that would evict an in-flight page).
* The NVMe admission rule is the wear-control rule and is not negotiable. A coder must not bypass it for convenience (e.g. "persist everything so we never recompute"). The rule is `persist iff reuse_probability × recompute_cost > write_cost + read_restore_cost`. A coder must
  compute all four terms (reuse_probability from C10's calibrated confidence; recompute_cost from the expert/KV recompute cost; write_cost from C11/C24's NVMe write bandwidth; read_restore_cost from C11/C24's NVMe read + H2D bandwidth) and apply the rule. A coder must not persist
  when the rule says not to — the initial policy is the concrete instance of the rule (shared prefixes have high reuse_probability; one-shot completions have ~zero; the rule says persist the former, not the latter).
* The class B price (from `04` §6.1: 8 K cold span = 54.8–60.1 ms FP16 / 27.4–30.0 ms FP8; 16 K = 109.6–120.2 / 54.8–60.1; 64 K = 438.3–480.8 / 219.1–240.4) is what C12 attaches to a KV admission as class B. A coder must not admit a class B session
  without attaching the price (the stated step time), and must not admit it unless the client opts in and the stated step time is within the client's own budget. A coder must not silently demote A→B under load (a session is never moved from A to B because load rose — if it must move,
  it is an explicit event with a reason, and the client sees it). A coder must not silently change the codec (a codec that changes values is a declared precision reduction, priced and recorded per page — not a tier move).
* Refusal is cheap, and is a valid outcome. A coder must not treat refusal as a failure to be avoided — it is the cheaper failure (admitting a request the card cannot serve converts one bad answer into a bad answer for every request behind it). A coder must implement refusal as a clean,
  informative outcome (the client is told it was refused, and why, in terms of the class/price/budget), not as a crash or a silent drop.

**Worked micro-example (host, sanity-check before device):**
Implement a fake C12 with the placement score (fixed structure, tunable weights), the priority order (as set membership — PINNED/ACTIVE/IN_FLIGHT/event-referenced excluded from the evictable set), the NVMe admission rule (all four terms computed),
  and the initial policy (shared prefixes, suspended sessions after TTL, repeatedly-reused cold segments, nothing else). Then: (1) build a fake residency table (C9) with pages in various states (PINNED, ACTIVE, IN_FLIGHT, PREFETCHED, WARM, COLD, event-referenced) and assert
  the eviction selection excludes PINNED/ACTIVE/IN_FLIGHT/event-referenced from the evictable set regardless of score — the "never touches in-flight or event-referenced" gate; (2) assert the evictable set is scored and the lowest-scoring is evicted — the placement-score-as-ranking gate;
  (3) implement the NVMe admission rule and assert a page is persisted only when `reuse_probability × recompute_cost > write_cost + read_restore_cost`, and that a one-shot completion (reuse_probability ~0) is not persisted — the wear-control gate; (4) implement the initial policy and assert
  shared prefixes are persisted, suspended sessions are persisted after a TTL, repeatedly-reused cold segments are persisted, and transient one-shot completions are not — the initial-policy gate; (5) implement class B admission and assert a class B session is admitted only if the client opts in and the stated
  step time (from the class B price table, `04` §6.1) is within the client's budget, and that the price is attached to the admission (not the word "slow") — the class-B-price gate; (6) assert a class A session is never silently demoted to B under load — if it must move, it is an explicit event with a reason
  and the client sees it — the "no silent demotion" gate; (7) assert a codec that changes values is treated as a declared precision reduction (priced and recorded per page), not a tier move — the "no silent codec change" gate; (8) implement refusal and assert a request that does not fit class A and is not admitted as
  class B is refused cleanly with an informative reason (class/price/budget), not a crash or silent drop — the "refusal is cheap" gate; (9) measure SSD write bytes per generated token on a prefix-heavy workload (C21 number) and assert it is under the configured bound — the wear-control proof gate.

**Files a coder should read before starting:**
* `docs/02-components.md` — C12 section.
* `docs/04-memory-tiering.md` §5.3 (the eviction score, the priority order, in-flight/event-referenced never evictable, shared-prefix bonus), §6.2 (the refusal path, the three outcomes, the three rules).
* `docs/04-memory-tiering.md` §6.1 (the class B price table — what C12 attaches to a class B admission: 8 K = 54.8–60.1 ms FP16 / 27.4–30.0 ms FP8; 16 K = 109.6–120.2 / 54.8–60.1; 64 K = 438.3–480.8 / 219.1–240.4).
* `docs/09-kv-engine-architecture.md` §5 (I1, I7 — eviction is bit-identical on reload; a cache miss changes latency, never the result; the refusal/eviction story in the KV engine doc).
* `docs/08-roadmap.md` — Phase 4 exit gate (KV_MISS_STALL_TIME / TOTAL_DECODE_TIME under threshold; prefetch hit rate published per model; `--force-cold` produces honest storage-bound reporting), Acceptance criteria (no synchronous NVMe read on the critical path;
  a cache page from a different model/config/codec is rejected, not misread; a crash during an NVMe write leaves committed objects intact).
* `ai-coder/c9-residency.md` (C9 — C12's eviction acts on C9's residency table; C12's `demote()` transitions through C9's EVICTING states; the placement score's recency/frequency/reuse come from C9).
* `ai-coder/c10-prefetch.md` (C10 — C12's reuse_prediction comes from C10's calibrated confidence; the deadline-aware throttling).
* `ai-coder/c11-transfer.md` (C11 — C12's transfer_cost and NVMe writes; the priority system; the deadline for demand reads).
* `ai-coder/c13-radix.md` (C13 — shared prefixes are what C12 persists; the shared_prefix_value in the placement score; the class B admission for KV pages).
* `ai-coder/c14-kv-cache.md` (C14 — the KV side of C12's admission; the class B price is what C12 attaches to a KV admission; shared-prefix persistence).
* `ai-coder/c21-profiler.md` (C21 — SSD write bytes per generated token is a C21 number; the eviction-of-in-flight-page attempt is a C21/counter event; the admission outcomes are reportable).
* `ai-coder/c22-budget.md` (C22 — the NVMe quota and the admission threshold come from C22's profile; the configured bound on SSD write bytes per token).
* `ai-coder/c24-platform.md` (C24 — the NVMe mechanism; C12's NVMe writes go through C11/C24).
