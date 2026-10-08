# 06 — Profiling and telemetry

> **STATUS: spec, not yet built.** No `kanjoos` binary exists; nothing in this
> document has ever run. Every number printed in a block below is a
> **format example**, not a measurement. Real, measured numbers live in
> `docs/00-verified-facts.md`, `docs/03-kernels.md` and `tools/bench/` output,
> and are labelled MEASURED where they appear. (Corollary for readers
> extracting figures: `12.22 tok/s` in section 2 is a placeholder.)

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