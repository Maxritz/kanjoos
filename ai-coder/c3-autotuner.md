# Worksheet — C3 ISA probe and kernel autotuner

**What it owns:** turning "the compiler accepts this" into a runtime decision.

**Why it exists (from `02-components.md` — C3 section):** `tools/isa_probe/` is
compile-time; C3 is the runtime half. On first launch for an arch it runs a bounded
benchmark sweep over the compiled kernel variants and writes the winner to a per-arch
cache (`kanjoos-tune/<arch>.json`).

**Spec lines that must hold:**
* `02-components.md` — C3 section (verbatim TuningKey, TuningResult, Autotuner API, rule, done-when).
* `docs/00-verified-facts.md` §0 — "a capability claim must name the emitted
  instruction. An exit code is not evidence." C3's runtime probing inherits this: a
  kernel variant is selected because it was measured faster on this arch, not because
  the compiler accepted it.

**Interface (verbatim):**
```cpp
struct TuningKey { std::string arch, kernel, quant; int m, n, k, tile; };
struct TuningResult { TuningKey key; double us; int variant; };
class Autotuner {
public:
    void calibrate(const TuningSpace&);     // bounded, once per arch
    const TuningResult& pick(const TuningKey&) const;
};
```

**Rule (verbatim):** autotuning is bounded and cached. It never runs mid-request. A
missing entry falls back to the analytic default, not to a guess-and-hope.

**Acceptance test (verbatim "done when"):**
* A tuned run reproduces within 5% of its own tuning measurement.
* The cache file invalidates correctly when the binary hash or arch changes.

**Gates:**
* **Bounded** — the calibration must be a fixed, stated budget (e.g. N variants × M
  repetitions, with a wall-clock ceiling). It is never "run until you're sure".
* **Cached per arch** — `kanjoos-tune/gfx1201.json`, `kanjoos-tune/gfx1031.json`.
  The cache key includes the binary hash so a recompile invalidates it.
* **Fallback is analytic, not guessed** — if a cache entry is missing, pick the
  variant the roofline/arithmetic model predicts, not the most recent or the default.
* Depends on C2 (DeviceCaps, arch), C17/C16 (the kernel variants being tuned),
  C21 (the timing measurements that drive the decision).

**Coder notes / pitfalls:**
* The "reproduces within 5%" gate means the tuner must record the measurement it made,
  and a later run with the same key must come back within 5% — if it doesn't, the cache
  entry is stale and must be invalidated, not trusted.
* "Reproduces within 5%" is the gate, and PCIe noise is real (`00` §8: PCIe cells move
  2–4% between runs of the same binary on an idle machine). So the tuner's repetition
  count and the acceptance tolerance must be set so that 2–4% PCIe wobble does not
  thrash the cache. A variant that is within the noise band of the current winner must
  not bounce the cache.
* The analytic fallback must be documented with the model that produces it (e.g. "for
  W4 g128 expert GEMM at M=32, the predicted winner is variant X because ..."), so a
  missing cache entry is a real decision, not a silent default.
* Do not autotune mid-request. The calibration is a one-time, bounded, startup-cost
  thing. A mid-request autotune would be a correctness hazard (the picked variant could
  change under a running step) and a latency hazard.

**Worked micro-example (host, sanity-check before device):**
Create a fake TuningSpace with 3 variants of one kernel, each with a synthetic
`us` measurement. Run `calibrate()`, then `pick()` for the matching key, and assert:
(1) the picked variant is the fastest in the space; (2) a second `pick()` returns the
same variant from cache; (3) mutating the binary hash and calling `pick()` falls back
to the analytic default (recorded), not the stale cache; (4) the calibration is bounded
— calling it with a space of N variants does not exceed M repetitions per variant.

**Files a coder should read before starting:**
* `docs/02-components.md` — C3 section.
* `docs/00-verified-facts.md` §0 (the standing rule that a capability claim must name the
  emitted instruction — applies to runtime probing too).
* `tools/isa_probe/` (the compile-time half; C3 is the runtime half — read it to see
  what "the compiler accepts this" looks like, then implement the runtime "and it's
  actually faster" half).
