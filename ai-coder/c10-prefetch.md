# Worksheet — C10 Prefetch predictor

**What it owns:** the thing that makes the RAM tier pay for itself.

**Why it exists (from `02-components.md` — C10 section; `01` §6):** prefetch is what moves expert bytes from RAM (or NVMe) into VRAM before their layer arrives, so the attention path
  never waits on a transfer. It is layered, cheapest first (L0 current token's routing → L1 previous token's routing for the same layer → L2 recent per-layer expert history → L3 cross-layer
  predictor → L4 request-level activation matrix → L5 adaptive horizon from observed bandwidth + hit rate). The untrained version (L1) is the cheapest test and the one to ship first — the project's
  own routing traces (`KNJ_TRACE_EXPERTS`, i.e. P0-7) already answer whether it predicts anything. Then L3, then L5.

**Spec lines that must hold:**
* `02-components.md` — C10 section (verbatim: the L0–L5 table, the Prediction struct, the `predict()` API, the two rules, the done-when).
* `01-architecture.md` §6 (control flow: C10 predictor → next layer's likely experts (+horizon); C10 is what makes the prefetch → RAM → VRAM → attention discipline possible).
* `04-memory-tiering.md` §5 (the discipline: predict → plan → prefetch → RAM → VRAM → attention; the prefetcher self-throttles — it estimates whether a speculative read will land before its
  deadline, and cancels or deprioritises reads whose confidence is below the current deadline pressure; confidence is calibrated against measured hit rate, not raw router softmax).

**Interface (verbatim `02-components.md` C10):**
```cpp
struct Prediction {
    std::vector<ExpertId> candidates;
    float   confidence;        // calibrated, not raw softmax
    uint32_t horizon_layers;
};
Prediction predict(uint32_t layer, const RoutingState&);
```

**Rules (verbatim):**
* Prefetch is an **optimisation, never a correctness requirement**. A wrong prediction must cost bandwidth, never a wrong answer.
* Prefetch is cancelled or deprioritised when the demand set is at risk of missing its deadline. Throughput never wins against the current request.

**Acceptance test (verbatim "done when"):**
* Measured prediction accuracy on real traces (top-8 overlap with next-token routing) is published per model.
* The adaptive horizon demonstrably beats a fixed horizon of any single value.

**Gates:**
* Depends on C18 (router — C10's L0/L1 are the router's own output; C10's "is this a good prediction" is ultimately against the router's next-token selection, which is what C18 produces),
  C9 (residency — C10's `prefetch()` is what C9 calls to get things into VRAM before their layer arrives; C9's `ready()` is the proof that prefetch worked), C11 (transfer — C10's prefetch
  enqueues transfers via C11; C10's self-throttling is what makes C11's transfers non-blocking and deadline-aware), C22 (budget — C10's horizon and aggressiveness are bounded by the VRAM/KV/expert
  budget C22 sets), C21 (telemetry — C10's prediction accuracy and the adaptive-horizon win are C21 numbers; "published per model" means they are C21 columns).
* P0-7 (`00` §9.6) is the gate for "does L1 predict anything?" — the project's own routing traces already answer that (59.1% new experts/token for source 2, 80.5% for source 1 — i.e. top-8
  overlap with next-token routing is ~41% / ~20%, which is the L1 signal). A coder must not assume L1 predicts well — P0-7 says it is weak (the hit-rate curve is close to linear in slots; there is no
  small hot set). So L1's value is real but limited: it predicts ~20–41% overlap, which is worth prefetching, but it is not a "small hot set you can keep in VRAM" signal. The coder must not over-claim L1.
* The "adaptive horizon demonstrably beats a fixed horizon of any single value" gate means C10's L5 must be measured against fixed horizons (1, 2, 3, 4, ... layers) and win. A coder must not implement L5 as a
  single fixed horizon and call it "adaptive". The gate is a comparison against fixed horizons, and the win must be demonstrated (via C21) on a representative workload.

**Coder notes / pitfalls:**
* Prefetch is an optimisation, never a correctness requirement. A coder must not let a wrong prediction cause a wrong answer — the worst a wrong prediction can do is waste bandwidth (a prefetch that brings in an
  expert that is not used). The demand set (C18's actual selected experts) is always satisfied, one way or another (VRAM resident, RAM→H2D, NVMe→RAM→H2D, or C19 fallback). A wrong prefetch never changes the
  selected set or the result.
* Confidence is calibrated against measured hit rate, not raw router softmax (`04` §5). A coder must not use the raw router softmax as the prefetch confidence. The router softmax is not a calibrated probability
  that the next token will use that expert — it is the router's own score, which is not the same as "this prefetch will land before its deadline". The confidence must be calibrated against the measured hit rate
  (the P0-7 curve, or a per-model measured prefetch-hit rate), so that "confidence below deadline pressure" is a real deadline-aware signal, not a softmax threshold.
* The self-throttling rule: prefetch is cancelled or deprioritised when the demand set is at risk of missing its deadline. Throughput never wins against the current request. A coder must implement the deadline-aware
  throttling — if the demand set (the experts the layer will actually use) is at risk of not being ready in time, speculative prefetches are cancelled or deprioritised so the demand transfers get the bandwidth.
  A coder must not let prefetch throughput starve a demand transfer. The demand set always wins.
* L1 (previous token's routing for the same layer) is the cheapest test and the one to ship first, and P0-7 already answers whether it predicts anything. A coder must implement L1 first (it is just "the previous
  token's top-k for this layer, possibly extended by recent history"), measure its top-8 overlap with next-token routing on real traces (P0-7's 41%/20% is the baseline), and report it per model (C21). Then L3 (cross-layer
  predictor — current router output predicts next-layer experts; "one small matmul"), then L5 (adaptive horizon).
* The horizon is a real, measured quantity. A coder must be able to report the horizon (in layers) that C10 used for a prefetch, and the measured landing time vs the layer's deadline. The adaptive horizon (L5) is what
  makes the horizon a function of observed bandwidth + hit rate, not a constant. A coder must not hardcode the horizon — it is set by C22's profile initially (1–2 future layers for batch 1, 2–4 for small batch,
  widen only if overlap measurably improves — `04` §5.2), then adapted at runtime by C10-L5.

**Worked micro-example (host, sanity-check before device):**
Implement a fake C10 with L0 (current token's routing), L1 (previous token's routing for the same layer), a fake C18 (router that produces the next-token routing), a fake C9 (residency), a fake C11 (async transfers
  with a stated completion time and a deadline), and a fake step loop. Then: (1) implement L1 as "previous token's top-k for this layer" and measure its top-8 overlap with the next-token routing on a real trace
  (P0-7's 41%/20% is the baseline C10 must beat or match) — the "published per model" gate; (2) implement the confidence as calibrated against the measured hit rate (not raw softmax) — the "confidence calibrated,
  not raw softmax" gate; (3) implement the deadline-aware self-throttling: when the demand set is at risk of missing its deadline, speculative prefetches are cancelled or deprioritised so demand transfers win — the
  "throughput never wins against the current request" gate; (4) implement L5 (adaptive horizon from observed bandwidth + hit rate) and assert it demonstrably beats a fixed horizon of any single value (1, 2, 3, 4 layers)
  on a representative workload — the "adaptive beats any fixed horizon" gate; (5) assert a prefetch issued with enough lead time lands in VRAM (C9 ready) before its layer's deadline, and a prefetch issued too late is
  cancelled or deprioritised so the demand transfer wins — the "prefetch → RAM → VRAM → attention" discipline; (6) assert a wrong prediction costs bandwidth (a prefetch for an expert that is not used wastes a transfer)
  but never changes the selected set or the result — the "optimisation, never a correctness requirement" gate.

**Files a coder should read before starting:**
* `docs/02-components.md` — C10 section.
* `docs/01-architecture.md` §6 (control flow — C10 predictor → next layer's likely experts (+horizon)).
* `docs/04-memory-tiering.md` §5 (the discipline; self-throttling; confidence calibrated against measured hit rate, not raw softmax; horizons per decode shape).
* `docs/00-verified-facts.md` §9.6 (P0-7 — the routing traces that answer "does L1 predict anything?"; the 41%/20% top-8 overlap is the L1 baseline; the hit-rate curve is the calibration source).
* `ai-coder/c9-residency.md` (C9 — C10's `prefetch()` is what C9 calls; `ready()` is the proof prefetch worked).
* `ai-coder/c11-transfer.md` (C11 — C10's prefetch enqueues transfers; the deadline-aware throttling).
* `ai-coder/c22-budget.md` (C22 — the horizon and aggressiveness are bounded by the VRAM/KV/expert budget).
* `ai-coder/c21-profiler.md` (C21 — prediction accuracy and adaptive-horizon win are C21 numbers, published per model).
* `ai-coder/c18-router.md` (C18 — C10's L0/L1 are the router's own output; the prediction is ultimately against C18's next-token selection).
