# Worksheet — C18 Router

**What it owns:** choosing the experts. It must be exactly the model's router.

**Why it exists (from `02-components.md` — C18 section):** the router is the model's own routing function, not an approximation the engine substitutes.
A wrong router silently changes the answer (and is the kind of defect that compiles cleanly and produces a plausible number — see the defect catalog in `00` §7.5/§8.6).
The router must run unbiased, read the bias under both spellings, do exact top-k, and report routing-agreement metrics for every compiled model so quantisation drift
in routing is visible, not silent.

**Spec lines that must hold:**
* `02-components.md` — C18 section (verbatim: read `exp_probs_b.bias` under **both** spellings; top-k exact, with a sampled-boundary top-k + mandatory exact certification
  pass acceptable; routing-stability measured not assumed; done-when).
* `00-verified-facts.md` §9.6 (P0-7: the routing result — 59.1% new experts/token for source 2, 80.5% for source 1; the hit-rate curve; the miss-cost arithmetic.
  C18 is the thing that produces the routing that P0-7 measured and that C17 consumes. If C18's routing drifts from the model's true routing, the whole residency story
  is built on a wrong routing. So C18 must be exactly the model's router.)
* `09-kv-engine-architecture.md` §5 (the router is authoritative — I3: cache pressure never silently changes the selected expert set; if an expert cannot be placed, the
  request stalls or falls back to CPU — it does not drop. C18 produces the selected set; C9/C17/C19 act on it. C18 must not silently change it.)

**The bias-spelling rule (verbatim, `02-components.md` C18):** read `exp_probs_b.bias` under **both** spellings — the prior defect where the wrong spelling silently ran
expert selection un-biased is the reason. Host memory, freed on the host, never a device pointer.

**Acceptance test (verbatim "done when"):**
* Un-biased execution is impossible to reach by configuration.
* Routing-agreement metrics are reported for every compiled model (top-1 agreement, top-k agreement, Jaccard, logit RMSE, router margin change).

**Gates:**
* Depends on C5 (the compiled model — C18 reads the router weights from the compiled manifest/checkpoint), C17 (the GEMM that consumes the router's selected experts),
  C9 (residency — C18's selected experts are what C9 tries to place), C19 (the fallback — if C18's selected expert cannot be placed, C19 is the escape hatch, not a silent drop).
* The "un-biased execution is impossible to reach by configuration" gate means the bias must be read and applied in a way that a configuration change cannot silently turn off.
  A coder must not make the bias an optional flag that defaults off. The "both spellings" rule means the loader/compiler (C4/C5) must make both spellings available and C18 must
  read the right one (or both, and pick the right one) — and a wrong spelling must not silently run. The prior defect was exactly that: the wrong spelling silently ran expert selection
  un-biased. So the gate is: there is no code path that runs expert selection without the bias, reachable by any configuration.
* The routing-agreement metrics are what make quantisation drift in routing visible. A quantised model's router can change later routing (`02-components.md` C5: "quantised hidden
  states can change later routing"). So C18 (or the compiler C5, on behalf of C18) must report, for every compiled model, the agreement between the quantised router and the BF16 router:
  top-1 agreement, top-k agreement, Jaccard, logit RMSE, router margin change. These are not decorative — they are the signal that tells you whether the quantised router is still the model's
  router. A coder must not skip them or report only one of them.

**Coder notes / pitfalls:**
* The bias is in `exp_probs_b.bias` (the gate/up/down experts each have their own bias? — read the model: Qwen3's router has a gate with a bias; the "both spellings" refers to the
  bias tensor's naming/shape under different checkpoint conventions). The point is: the bias is part of the router, and an unbiased router is a wrong router. C18 must read and apply it, and
  the loader/compiler must expose it under both spellings so the right one can be read regardless of checkpoint convention. Do not assume one spelling.
* Top-k exact is the default. A sampled-boundary top-k is acceptable only with a **mandatory exact certification pass** — i.e. the sampled version runs for speed, but an exact selection
  is also run and the two must agree (or the sampled version is invalidated). This is the "HPC-Ops result: sampling controls common-path work, never the answer" rule. Do not ship a sampled
  top-k without the exact certification pass, and do not let the sampled version be the only one reachable in production.
* Routing stability is measured, not assumed. A coder must not assume a quantised router agrees with the BF16 router. The agreement metrics are the measurement. If the agreement is poor
  (e.g. top-k agreement well below 100%, large logit RMSE, large margin change), that is a signal that the quantised model's router is not the model's router — and the engine must not silently
  use it. The compiler (C5) must report these metrics per compiled model so the operating-point decision (which pack, which precision) can factor in routing drift.
* The router's selected experts are what C17 consumes and C9 tries to place. If C18's selection drifts from the model's true selection, C17 computes the wrong thing and C9 places the
  wrong experts — and both can be "plausible" (the GEMM produces numbers; the residency produces a slot). So the router's correctness is not capturable by a tolerance check on the GEMM output
  alone — it is capturable by the routing-agreement metrics (top-k agreement, Jaccard) against the BF16 router. Do not rely on the GEMM output tolerance to catch a wrong router.

**Worked micro-example (host, sanity-check before device):**
Take a tiny model's router weights (gate + bias, under both spellings), a few hidden-state inputs, and run C18's selection: (1) read the bias under both spellings and assert the
  right one is used (and that a configuration cannot reach a bias-off path — the "un-biased is impossible" gate); (2) run exact top-k and a sampled-boundary top-k, and assert the sampled
  version is only used if the exact certification pass agrees (the mandatory exact pass gate); (3) for a quantised version of the same router, compute and report the routing-agreement metrics
  (top-1 agreement, top-k agreement, Jaccard, logit RMSE, router margin change) against the BF16 router — and assert these are reported for every compiled model (the "reported for every compiled
  model" gate); (4) assert that a hidden-state input produces the same selected expert set as a BF16 reference oracle (the router is exactly the model's router) — and that a wrong bias spelling
  produces a different (wrong) selection, so the "both spellings" rule is doing real work.

**Files a coder should read before starting:**
* `docs/02-components.md` — C18 section.
* `docs/09-kv-engine-architecture.md` §5 (I3 — the router is authoritative; cache pressure never silently changes the selected expert set).
* `docs/00-verified-facts.md` §9.6 (P0-7 — the routing result the router produces and that the residency story is built on).
* `ai-coder/c5-compiler.md` (the compiler that quantises the router and must report routing-agreement metrics; the bias-spelling exposure).
* `ai-coder/c17-expert-gemm.md` (the GEMM that consumes C18's selected experts — C18 is the input to C17).
* `ai-coder/c9-residency.md` (the residency manager that tries to place C18's selected experts — C18 is the input to C9).
* `ai-coder/c19-cpu-fallback.md` (the fallback when C18's selected expert cannot be placed — C18's selection is what C19 falls back around, not a silent drop).
