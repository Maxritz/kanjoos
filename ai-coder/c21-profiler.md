# Worksheet — C21 Profiler and telemetry

**What it owns:** the `--profiling` subsystem. Full specification is in `06-profiling.md`
(because it is a first-class deliverable with its own output contract, its own
instrumentation-floor measurement, and its own acceptance tests). This worksheet is the
bridge from that spec to an implementation.

**Why it exists (from `02-components.md` — C21 section; `08` acceptance criteria):**
every later phase is verified by it. The profiler is built first, before there is
anything to profile. Every number it reports must be interpretable — including the
instrumentation floor, reported alongside, so a "0 ms" column is read as "at or below
the floor", not as "zero".

**Spec lines that must hold:**
* `02-components.md` — C21 section (verbatim: per-component `ops / %dev / dev us /
  idle us / host us`, fixed column layout of the reference example, instrumentation
  floor measured and reported alongside).
* `06-profiling.md` — full specification (output contract, instrumentation floor,
  acceptance tests). Read this file first; the worksheet borrows its headings.
* `08-roadmap.md` — Phase 1 exit gate: "an empty forward pass over a small dense model
  produces the reference table with a non-zero measured floor, and the same binary runs
  on both targets and both platforms."
* `08-roadmap.md` — Performance acceptance: "no synchronous NVMe read in the attention
  or expert path; H2D/D2H overlap compute — visible as idle < wall in the profile table."

**Interface (from `02-components.md` C1/C21 contracts):**
* C1 timestamps every op into an `OpRecord` (`dev_start_ns`, `dev_end_ns` from the event
  ring; `host_begin_ns`, `host_end_ns` CPU-side). C21 consumes those records.
* C9/C10/C11/C12 emit telemetry events on residency/transfer/eviction transitions — C21
  accumulates them into the per-component columns.
* Output: per-component `ops / %dev / dev us / idle us / host us` in the fixed column
  layout of the reference example.

**Output contract (read `06-profiling.md` for the full thing; the worksheet states the
parts every implementation must satisfy):**
1. Fixed column layout — the same columns in the same order in every run, so a diff of
   two profile tables is meaningful.
2. Instrumentation floor — measured and reported alongside the numbers, so a "0" is read
   as "at or below the floor", not as "zero device time".
3. Every number is attributable to a component (C1–C24) — no anonymous "other" column
   that hides where time went.

**Acceptance test (verbatim + from `06-profiling.md`):**
* The profile table has the fixed column layout in the reference example.
* The instrumentation floor is measured (not assumed) and reported alongside.
* On a small dense model with `--profiling`, the table is non-empty and the floor is
  non-zero (Phase 1 exit gate).
* On a MoE decode step on gfx1201, `--profiling` shows a non-zero overlap factor
  (device busy time < wall time) — and that overlap is attributed to the right
  components (C1/C11), not to an anonymous bucket.
* "no synchronous NVMe read on the critical path" must be reconstructable from the
  profile: no op in the attention/expert path has NVMe-on-the-critical-path as its
  source of device time. If one does, that is a `STALL_TIME_MS` telemetry event, not a
  profile column.

**Gates:**
* Depends on C1 (OpRecord timestamps), C9/C11/C12 (transition telemetry events), C2
  (arch, caps), C17/C16 (the kernels being profiled).
* The instrumentation floor is a real measurement: it is the smallest non-zero device
  time the profiler can attribute to a single op. It is not a constant and not assumed —
  it is measured by running a no-op or near-no-op op and reading the smallest
  attributable time. Read `06-profiling.md` for the exact method.
* "no synchronous NVMe read on the critical path" (I4) is enforced by a debug assert
  **plus** the profiler — the profiler is the persistent, non-debug proof. A component
  that violates I4 during a profiled run must produce a detectable signal in the profile
  (a stall event or an NVMe-on-critical-path column), not just an assert that is off in
  release builds.

**Coder notes / pitfalls:**
* Do not implement the floor as a constant like "0.5 µs". The floor is measured on the
  real target by running the cheapest possible attributed op and reading the smallest non-
  zero time. It differs per arch, per kernel, per stream topology. A constant floor is
  the same class of error as assuming the ISA matrix.
* Do not merge profiler output into a single "GPU time" column. The point of the fixed
  layout is that you can tell whether time is in attention, expert GEMM, transfer, or
  idle — and that is the difference between "the engine is slow" and "the engine is
  transfer-bound because X% of wall time is idle waiting on a PCIe read". The latter is
  the answer the whole plan is built around; the profiler must be able to say it.
* The profile table must survive a kernel that is "fast but wrong" — the correctness sweep
  (full output vs host oracle) is a separate check, and the profiler must not be the thing
  that catches a wrong answer. If the profiler is the only thing catching a wrong answer,
  the profiler is being asked to do a correctness sweep's job.
* Do not hide "idle" inside "device time". Idle (waiting on a transfer/event) is a
  distinct column from device busy time, because idle is the transfer-bound signal. A
  profile that reports "device time = wall time" with no idle column is uninterpretable.

**Worked micro-example (host, sanity-check before device):**
Take a fake stream of OpRecords (a few compute ops, a transfer op with an event
dependency, a compute op that depends on that transfer). Walk the records, attribute
each to a component, and emit a one-row-per-component table with columns `ops / %dev /
dev us / idle us / host us` plus a floor row. Assert: (1) the transfer-dependent compute
op's `dev_start_ns` is not attributed to the transfer component's "device busy" time
(event dependency respected); (2) the interval the compute op spends waiting on the
transfer is attributed to "idle" for the compute component, not to "device busy"; (3) the
floor row is present and is the smallest attributable non-zero time in the table; (4) the
column order matches the reference example's fixed layout.

**Files a coder should read before starting:**
* `docs/06-profiling.md` (the full spec — read this first).
* `docs/02-components.md` — C21, C1 (the OpRecord contract), C11 (transfer telemetry).
* `docs/08-roadmap.md` — Phase 1 exit gate, Performance acceptance criteria.
* `docs/00-verified-facts.md` §8 (PCIe noise 2–4% — sets the floor/tolerance mindset).
