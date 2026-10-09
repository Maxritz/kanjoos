# 06 — Profiling and telemetry

> **STATUS: partly built (2026-10-08).** The C21 core exists in
> `src/profiler/profiler.{h,cpp}` and is driven by `kanjoos-run`, which the root
> build now produces (`knj_runtime` + `kanjoos-run` targets). Implemented and
> verified: `--profiling[=table|json|csv|no-subtract|off]`, `--profile-dir`,
> `--profile-floor`, `--profile-warmup`, `--profile-detail=class|layer`. **Not**
> implemented: `--profiling=counters`, `--profiling=full` (both are refused with a
> message rather than silently redefined) and the `kanjoos serve` spelling.
>
> **Device event backend: BUILT.** `enable_device_backend()` flips the domain to
> `clock device`, and a device layer feeds `device_span(name, start_ns, end_ns,
> stream)` with **offsets from one epoch event per stream** — offsets rather than
> durations, because idle is a gap *between* spans. `tools/bench/q4k_stream_ffn.hip`
> is the first path that reports this way: 27 compute-stream event pairs per pass,
> folded into one `device-layer` row with its own idle gaps, printed beside a host
> row for the same pass. See §4.1 for what that number does and does not claim.
>
> **Residency and transfer footers: MEASURED on that path**, from counters the pass
> already keeps (slots/peak, evictions, demand hits, h2d/d2h/nvme bytes, peak
> pinned staging). They print `NOT MEASURED on this path` only where a path has no
> such counters, which remains the correct thing for it to say.
>
> **Regression gate: `tools/c21/profile_diff.py`.** Diffs two C21 JSON profiles and
> exits non-zero when a judged component's self time regresses past a tolerance;
> re-pinning requires `--reason`, which is written into the reference. See §11.
>
> **Every number in the blocks below is still a format example and none of them
> has ever been produced by a run.** The values are unchanged from the original
> spec; they are not measurements, and must not be quoted as such. (Corollary for
> readers extracting figures: `12.22 tok/s` in section 2 is a placeholder.) A
> real, measured table from this machine is in `docs/CODING-LOG.PENDING.md`
> (Phase 42) — same columns, actual microseconds, on the Qwen3-MoE-4x0.6B GGUF.

This is a deliverable with an output contract, not a debugging aid. If the engine
cannot say where its microseconds went, every other decision in this plan is
guesswork, and the single most common failure in this class of work — an I/O
problem mistaken for a kernel problem — is invisible without it.

---

## 1. Command line

```
kanjoos serve --profiling              # table to stdout at exit
kanjoos serve --profiling=table         # same, explicit
kanjoos serve --profiling=json         # machine-readable, full detail
kanjoos serve --profiling=csv          # per-op rows, for spreadsheets
kanjoos serve --profiling --profile-dir runs/2026-10-05-1130
kanjoos serve --profiling --profile-floor 256     # empty ops for the floor
kanjoos serve --profiling --profile-warmup 8      # discard the first N steps
kanjoos serve --profiling --profile-detail=layer  # class | layer | kernel
kanjoos serve --profiling=no-subtract             # keep raw numbers
kanjoos doctor                         # always prints the capability header
```

`--profiling` never changes what the engine computes. It changes what it
measures, and it says so in its output header.

---

## 2. The output contract

The table is fixed. Columns, order, alignment and the footer line are part of the
interface; a downstream script parses it.

The block below is the **contract**: column names, order and alignment are
normative; the values are illustrative and were never produced by a run.

```
stdout:

  kanjoos 0.1.0   device AMD Radeon RX 9070 XT (gfx1201, 32 CU)   rebar: full aperture
                 vram 14.00/16.00 GiB   ram 62.4/96.0 GiB warm   nvme 4.1 GB/s   pinned 6.0 GiB
                 kv codec fp8   expert format w4   slots 211   profile 12g/96g
                 steps 24   tokens 512   draft 1.31x   accept 0.71   wall 41.882 s   tok/s 12.22

     component           ops   %dev      dev us     idle us     host us
     attention            24  95.4%   535880.80      377.62       49.20
     projections         187   3.2%    17723.32     2026.19      460.80
     recurrent            90   1.1%     6127.59     1170.02      143.80
     attention-mix        72   0.2%     1335.28     1102.27      111.60
     head+sample           1   0.0%       98.20        91.80       56.90
     bias                 18   0.0%       90.03      265.52       66.60
     transfer              1   0.0%       79.80     1087.10     1084.30
     norms                67   0.0%       72.98      784.24      139.40
     residual             48   0.0%       23.40      198.44       73.30
     ffn-activate         42   0.0%       21.52      372.84       67.70
     embed                 1   0.0%        0.48       25.44        2.40
   instrumentation floor, 256 empty ops timed through the same begin/end path: 0.70 us host, 19.54 us device each.

  residency   vram 91.4%   ram 8.1%   nvme 0.5%   slots 211/211   evictions 47   prefetch hit 0.82   stall 0.00 ms
  transfer    h2d 4.19 GB   d2h 0.02 GB   nvme read 0.31 GB   nvme write 0.00 GB   peak pinned 5.8 GiB
  cold store  quota 1.4 TiB   used 812 GiB   write bytes/token 0
```

---

## 3. What each column means, precisely

This is where a profiler usually lies, so the definitions are normative.

| column | definition |
|---|---|
| `component` | the op class, or the layer/kernel when `--profile-detail` says so |
| `ops` | number of timed ops attributed to that component |
| `%dev` | `dev us` for the component ÷ `dev us` summed over all components |
| `dev us` | **GPU busy time** — the interval between a start event and its matching end event on the same stream, summed |
| `idle us` | **GPU idle time attributable to this component** — see §4 |
| `host us` | **CPU time inside the component's submit path** — from entering the component's begin marker to leaving its end marker, excluding blocking waits |

`dev us` never includes idle time. `idle us` is never folded into `dev us`. That
separation is the whole point of the table: an op that is 95% of device time is
not the same problem as an op that is 95% of wall time and 3% of device time,
and the table makes the difference visible in one line.

`%dev` sums to 100% over components (modulo the floor subtraction in §6), not
over wall time. The gap between `Σ dev us` and `wall` is the overlap and the
idle, both of which the residency block reports.

---

## 4. How `idle us` is measured

Idle is the hard part, and getting it wrong makes the whole table a decoration.

```
for each timed op:
    record (class, stream, host_begin, host_end, start_event, end_event)
on poll():
    walk the ring, resolve events to timestamps, and build a timeline per stream
for each op in timeline order:
    idle  = op.start - previous op's end       (same stream, if that stream is
                                               the critical stream)
```

* **Per stream**, then summed across streams. Two streams running concurrently
  each have their own gaps; the engine is overlapped, so a global "gap" would
  report a negative or a lie.
* The critical stream is the one holding the sampler dependency. Idle on that
  stream is the idle that delays the token.
* Because ops are submitted asynchronously, an op's `idle` is bounded by when it
  was *enqueued*, not when the GPU got to it. That is intentional: CPU-side
  starvation shows up as idle, which is exactly what should happen.

**Storage is not idle.** A demand NVMe read that stalls the step appears as a
transfer component with real `dev us` and a large `idle us` on the consumer, plus
`STALL_TIME_MS` in the residency block. The engine must never let a storage wait
disappear into a generic idle number.

---

## 4.1 The device clock, and what its numbers claim

A host-only path has no device events, so `dev` *is* the op's own interval and
the report declares `clock host`. A device backend changes that, and the
arithmetic is fixed:

* **Offsets, not durations.** The device layer records a begin and an end event on
  one stream, then at flush time asks for each event's offset from a single
  **epoch** event recorded on that same stream. A duration cannot express idle —
  idle is a gap *between* two spans — so the layer supplies start and end and the
  profiler computes the gaps.
* **Per stream.** Gaps are summed per stream. Two concurrent streams have their
  own idle, and one global gap would be a lie about an overlapped engine.
* **Idle is proven, not imputed.** A gap counts as idle only when it sits between
  two **outermost** spans on a stream. A gap inside a parent span is not idle: the
  report already attributes that interval to the parent. The consequence is that a
  span containing a queue-empty stall is *not* split into work + idle. That
  asymmetry is deliberate — idle is claimed only where it can be proven.
* **Self time, the same rule as host rows.** A span contained in another
  contributes to its parent's child total, so a parent and its child never both
  claim the same microseconds.
* **The floor has two clocks.** The host floor is measured through the host
  begin/end path; a *device* floor must be measured through the event path
  (`set_device_floor`). Until it is, the dev rows are published **raw**, because
  subtracting a host floor from device time is an arithmetic error wearing a
  correction's clothes. The multiplication is per **measurement** — `ops` for a
  host-marker row, span count for a device-only row.
* **Resolution is stated.** `hipEventElapsedTime` reports float milliseconds, so a
  span below roughly 0.1 µs is inside the timer's noise. Isolated empty event
  pairs on this machine measure a floor of **0.5–0.8 µs** (min of 64); the report
  names the floor and the resolution instead of implying precision it lacks.
* **A dropped span is a stated loss.** A ring that fills reports `N dropped`, so a
  profiler that lost measurements never reads as a faster run.

What a `device-layer` span does **not** claim: it is the device-timeline window
that the layer's work occupies in stream order. Nothing here samples execution-unit
occupancy, and the report says so rather than calling the window "GPU busy".

## 5. Instrumentation

```cpp
enum class OpClass { Embed, Norm, Attention, AttnMix, Projection, FFNAct,
                     Residual, Recurrent, MoERouter, MoEGEMM, KVWrite, KVRead,
                     KVQuant, Spec, Transfer, Head, Bias, Sample };

class OpScope {                       // RAII, zero-cost when disabled
public:
    OpScope(OpClass c);
    ~OpScope();
    void  set_device_events(hipEvent_t start, hipEvent_t end);
};
```

* Host timing: `rdtsc`-equivalent (`__rdtsc` / `clock_gettime(CLOCK_MONOTONIC_RAW)`)
  through a per-thread accumulator. RAII so no code path can forget the end.
* Device timing: a **preallocated ring of `hipEvent_t` pairs**, never a
  `hipEventCreate` inside the step. The ring is sized once at init for the
  maximum ops per step and reused; `hipEventRecord` is the only per-op cost.
* The ring is drained by `Runtime::poll()` (C1), which is also where transfer
  completions are retired and residency is unblocked — one pass, one lock.

### 5.1 The component taxonomy

The classes in the reference table, plus the MoE-specific ones:

| class | covers |
|---|---|
| `embed` | token + position embedding lookup |
| `norm` | RMSNorm / LayerNorm, fused where fused |
| `attention` | the attention kernel proper |
| `attention-mix` | head output mixing, gating, RoPE application |
| `projections` | q/k/v/o, dense FFN, gate/up/down projections |
| `ffn-activate` | activation functions |
| `residual` | residual adds and gated residual |
| `recurrent` | gated delta-net / Mamba state updates |
| `expert-gemm` | grouped MoE GEMM (new) |
| `expert-gemm-unpack` | the unpack arithmetic inside it (new) |
| `router` | routing and top-k (new) |
| `kv-read` / `kv-write` | KV page access (new) |
| `kv-quant` | online KV quantisation (new) |
| `transfer` | H2D, D2H, NVMe I/O |
| `spec` | drafting and verification (new) |
| `head+sample` | lm head and sampler |
| `bias` | routing bias and other small fixups |

Splitting `expert-gemm` into `expert-gemm` and `expert-gemm-unpack` is
deliberate: on gfx1031, where unpack is real work rather than a footnote, a
single fused number would hide the one thing that needs optimising. If unpack is
more than a few percent of expert GEMM time, the pack format is wrong — the
profiler should make that a visible fact rather than a suspicion.

---

## 6. The instrumentation floor

Every measurement is wrong by the cost of measuring it, so the engine measures
its own overhead and prints it, in the same format, every run.

```
   instrumentation floor, 256 empty ops timed through the same begin/end path: 0.70 us host, 19.54 us device each.
```

* `--profile-floor N` (default 256) runs N **empty** ops through the identical
  begin/end path at startup: same RAII scope, same event pair, same ring slot,
  same accumulator.
* The measured per-op host and device cost is the floor.
* By default the floor is **subtracted** from every component's numbers, so the
  `dev us` column reads as kernel time rather than as measurement time. The
  subtraction is reported, and `--profiling=no-subtract` shows the raw numbers.
* A component whose total is at or below the floor is reported but marked, so a
  0.48 µs `embed` is not silently rounded into a false claim.
* If the floor itself changes (different driver, different clock state, different
  profile mode), the report changes with it — it is measured per run, never
  cached.

The floor line is part of the contract. A profiler that reports numbers without
reporting its own overhead is making a claim it cannot support.

---

## 7. Overhead modes

| mode | what it costs | what you get |
|---|---|---|
| `off` | nothing | nothing |
| `counters` | host timers only, ~0.05 µs/op | `ops`, `host us`, all telemetry counters |
| `full` | host timers + event pairs | the complete table |
| `table` | `full`, printed at exit | the reference output |

`--profile-warmup N` discards the first N steps before accumulating, because
first-call kernel loading, allocator warm-up and the first NVMe extents are not
representative of steady state.

**Rule:** if `full` mode changes throughput by more than 3%, that is a profiler
bug to fix, not a cost to document. The overhead is bounded by the event ring
being preallocated and by `poll()` doing one pass.

---

## 8. Telemetry — always on, low cost

Counters are collected in `counters` mode and are cheap enough to leave on:

```
expert hit / miss        RAM hit        NVMe read count
H2D bytes                D2H bytes       NVMe read / write bytes
prefetch hit rate        prefetch waste bytes
KV hit rate by tier      KV recompute rate
GPU slot occupancy       RAM occupancy
CPU fallback count       CPU fallback time
GPU expert exec time     queue idle time
```

Plus the residency and transfer blocks shown in §2, which are the numbers that
answer the question the whole engine exists to answer: *is this run
VRAM-resident, RAM-resident, storage-assisted, or storage-bound?*

---

## 9. Acceptance criteria

The profiler is done when:

1. `--profiling` prints the reference table with the exact column layout and the
   floor line, and `awk`/`grep` over it is stable enough to diff between runs.
2. The floor is measured every run, and subtracting it changes small components
   measurably — proving the subtraction is real and not decorative.
3. An intentionally injected stall (a deliberately cold cache, `--force-cold`)
   shows up as `transfer` device time plus `idle us` on the consumer plus a
   non-zero `STALL_TIME_MS`. **This is the test that proves the profiler can see
   the failure it exists to find.**
4. `--profiling=json` and `=csv` agree with the table to within rounding.
5. Overhead of `full` mode is under 3% on a resident decode step.
6. The same table appears on Windows and on Linux, for both targets.

---

## 10. Why this shape

The table has three properties worth defending, because they are the ones that
make it useful in practice:

* **It separates GPU busy from GPU idle from CPU time.** Almost every
  architectural mistake in a tiered engine is a *scheduling* mistake, and
  scheduling mistakes are invisible in a device-time-only measurement.
* **It prints its own floor.** Otherwise the smallest components are fiction.
* **It names the components, not the kernels.** `attention` at 95% of device time
  is a finding; `v_wmma_f32_16x16x16_f16_w32` at 95% of device time is not,
  because it does not tell you whether attention is supposed to be there.

The example table's shape — attention dominant, everything else rounding noise,
transfer with a large `idle us` and a larger `host us` — is the signature of a
correctly diagnosed run. The plan's success condition is that the same table on
an engine with good tiering shows the same attention dominance **and** a transfer
line that is small, with a non-zero `storage-bound` label on the runs where it
is honestly bound.
## 11. The regression gate

A table you read is not a gate. `tools/c21/profile_diff.py` makes the comparison
the artifact:

```sh
python tools/c21/profile_diff.py --self-test                 # grades the gate itself
python tools/c21/profile_diff.py --ref ref.json --cur new.json
python tools/c21/profile_diff.py --ref ref.json --cur new.json --repin \
    --reason "why the new number is the right number"
```

Three decisions worth stating, because each one is a way the gate could have been
decorative:

1. **A component in the reference but absent from the current profile is a
   regression**, not a skip — the same rule `run_bench.sh` already applies to a
   pinned metric it cannot read. Renaming a scope therefore fails the gate, which
   is the point: a rename silently breaks every comparison built on the old name.
2. **Rows below the noise floor are not judged**, and the report says `NOT JUDGED`
   with the threshold. The profiler publishes its own floor; a 40% "regression" on
   0.4 µs is noise wearing a percentage.
3. **Clock domains must match.** Comparing a host interval against device event
   time is meaningless rather than merely imprecise, so the tool refuses (exit 3)
   instead of printing a number.

Exit codes: `0` pass, `1` regression or vanished component, `2` usage error
(including `--repin` without `--reason`), `3` inputs not comparable.

A re-pin records `reason`, `pinned_at`, `metric`, `tol_pct`, `clock_domain`,
`detail`, and the reason it replaced. `--repin` without `--reason` is refused,
because a pin that does not say why it moved is indistinguishable from a pin that
was moved to silence a gate. A reference with no pin block is reported as such
rather than silently trusted.

**One artifact to read carefully:** container rows (`prefill`, `decode`, a test
suite's `pass`) are self time, so when their leaves get *slower* the container's
self time gets *smaller* and the diff reports `improved`. Measured: a 1-thread run
against a 32-thread reference reports `decode` −53.0% and `prefill` −48.8% while
`head` regresses +972%. The rows partition the run, so this is arithmetic and not
a bug — but it means an `improved` container is never evidence that anything got
better.

## 12. Sub-component instrumentation of `moe`

`moe` was 57% of device time on the reference model and a single opaque row. It is
now instrumented at the stage level, so the dominant component is attributed
instead of restated:

| row | what it covers |
|---|---|
| `moe-router` | the router GEMM, the softmax, the exact top-k, the renormalise |
| `moe-gather` | gathering the rows an expert actually sees |
| `moe-gate` / `moe-up` / `moe-down` | the three expert GEMMs, one row each |
| `moe-act` | the silu×up activation |
| `moe-scatter` | the weighted scatter-add back into the residual |
| `moe` | what is left: the output fill and the loop itself |

Measured on Qwen3-MoE-4x0.6B (`--bench 16 8 --profiling --profile-warmup 2`): the
`moe` row falls from **57.2% to 0.2%**, and the branch is `moe-down` 17.2% +
`moe-up` 16.7% + `moe-gate` 16.4% + `moe-act` 6.8% + router/gather/scatter ≈ 0.1%
— i.e. the three expert GEMMs are ~50% of device time and **the router is 0.1%**.
The scope names deliberately avoid a bare `router`/`gate` so they read as parts of
`moe` rather than as top-level components. With `--profile-detail=layer` the same
instrumentation yields `moe-gate:12` and friends, unchanged.

## 13. The baseline, and the reference profile this checkout pins

**MEASURED 2026-10-08** on the only model that runs end-to-end here
(`Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf`). The cell-level record, with every
`run.log`, menu, table and JSON, is
[`records/c21-baseline-2026-10-08/`](../records/c21-baseline-2026-10-08/README.md);
this section is the design-facing statement of it.

```sh
export PATH="/c/Strawberry/c/bin:$PATH"   # MinGW runtime, else the exe exits 127
kanjoos-run.exe --model <moe.gguf> --bench 64 32 --threads 8 --ctx 2048 \
    --profiling=json --profile-dir <dir> --profile-warmup 4
```

| axis | measured |
|---|---|
| threads (ctx 2048) | 1 → 2 → 4 → 8 → **16 → 32**: speedup 1.00× → 1.91× → 3.33× → 5.47× → **6.82×** → 5.32× |
| context (8 threads) | ctx 512 / 1024 / 2048 / 4096 / 8192 → profile total 9.48 / 9.30 / 9.26 / 9.16 / 9.21 s, **3.4% spread** |
| repeatability, pinned cell ×3 | prefill 6407.28 / 6315.91 / 6344.78 ms, decode 3288.98 / 3267.44 / 3267.16 ms → **≤1.5%** |
| shares at the pin | expert GEMMs (down+gate+up) **50.9%**, whole `moe` path 58.9%, attention block 23.3%, `head` alone **17.8%**, everything else **0.6%** |

Three things this baseline settles:

1. **The ceiling is 16 threads (6.82×, 43% of linear on 16 physical cores).** 32
   threads is *worse than 16* on both axes — SMT oversubscription, measured, not
   assumed. So a "more threads is better" default is wrong here by 22%.
2. **A thread increase has two opposite effects, and only the component table
   shows both.** 16 threads vs 8: `head` −32.0%, `moe-down` −24.2%,
   `moe-gate` −23.8%, `moe-up` −23.7%, `attn-o` −18.0%, `qkv` −11.6% — while
   `moe-gather` +44.4%, `moe-scatter` +33.2%, `prefill` +30.4%, `decode` +26.6%,
   `norm/-ffn` +12.5/+12.9%. The parallel regions gain, the serial glue pays the
   pool wake-up. Overall it is 22% faster, which is exactly why the wall clock
   alone is not enough evidence.
3. **`--ctx` is an allocation knob on this path, not arithmetic.** Prefill is 64
   tokens and decode attends over `n_past` (96), so the KV budget never binds;
   the 16× change in `--ctx` moves the total by less than the run-to-run noise of
   the *machine* would in a noisy cell. Reported because it was measured; a future
   path with a real KV budget must not inherit the claim.

**The pin.** [`tools/c21/baseline/ref-8t-ctx2048.json`](../tools/c21/baseline/ref-8t-ctx2048.json)
holds the 8-thread cell with its `pin` block, so the gate can be run without
re-deriving a reference:

```sh
python tools/c21/profile_diff.py --ref tools/c21/baseline/ref-8t-ctx2048.json \
    --cur records/c21-baseline-2026-10-08/<cell>/profile.json
```

Run against the matrix it passes the two repeat cells (`rc 0`, 18 judged, 0
failed) and fails the slower operating points as it should: `t16` 7 failed, `t32`
11 failed, `t1` 10 failed. The cell chosen is 8 threads, **not** the fastest one
(16): a reference that encodes an over-subscribed optimum would make the next
person's `16 → 8` change look like a regression. The reason is recorded in the
file itself, as `--repin` requires.

**A threshold finding, left alone on purpose:** the two cells that differ only in
`--ctx` each failed on one *small* row (`moe-gather` +13.6% on 2.2 ms,
`norm-ffn` +15.2% on 3.9 ms) — 0.02% of the run each. The default `--min-us 50`
is too low to be meaningful for this model, where the honest floor is ~5 000 µs.
It is left at 50 because the threshold is global and retuning it to make one
model's noise disappear is the tolerance-fitting AGENTS.md §4 rule 8 forbids;
the fix is to pass `--min-us 5000` when running this baseline, and to say so.
