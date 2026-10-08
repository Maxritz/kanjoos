# Worksheet — C1 Runtime core and stream scheduler

**What it owns:** the op graph, stream set, event ring, submission cadence.
A decode step is not a function call — it is a queue of ~500 tiny ops. Whether
they overlap is worth more than most kernel micro-optimisation, so this component
exists to make overlap expressible and measurable.

**Why it exists (from `01` §1, §6):** the central line of the control flow is
"attention and expert GEMM only ever see VRAM; everything else is a prefetcher's
job." C1 is the substrate that makes that possible without blocking: transfer ops
enqueued now must complete before the layer that needs them, tracked by events, and
the scheduler must be able to express that dependency without a `hipDeviceSynchronize`
inside the step loop.

**Spec lines that must hold:**
* `02-components.md` — C1 section (verbatim ownership, OpRecord, Runtime API, rule, invariants, failure mode, done-when).
* `01-architecture.md` §6 — one-token control flow: C11 poll completions at the end of the step; C9 ready() is the only thing kernels ask.
* `04-memory-tiering.md` §5.1 — "the attention path never issues a storage read"; the planner guarantees a RAM page has landed before attention, or the CPU path is taken.

**Interface (implement to this; other components depend on it):**
```cpp
enum class OpClass { Embed, Norm, Attention, AttnMix, Proj, FFNAct,
                     Residual, Recurrent, MoERouter, MoEGEMM, KVWrite,
                     KVRead, KVQuant, Spec, Transfer, Head, Bias, Sample };

struct OpRecord {
    OpClass     cls;
    uint64_t    dev_start_ns, dev_end_ns;   // from the event ring
    uint64_t    host_begin_ns, host_end_ns; // CPU-side, around the launch
    uint32_t    stream;
};

class Runtime {
public:
    StreamId compute_stream();          // where math goes
    StreamId transfer_stream();         // where copies go, may share compute
    StreamId aux_stream();              // norms/quant/rope, overlappable

    // Non-blocking. Returns immediately; the op is timed by C21.
    void submit(Op op);

    // Walks the event ring, updates C21 accumulators, retires transfer ops,
    // unblocks anything waiting on residency (C9).
    void poll();

    void submit_all();                  // one fence point, then poll
};
```

**Rule (verbatim):** every op is submitted to a stream and timestamped. Nothing
calls `hipDeviceSynchronize()` inside the step loop; a full sync exists only at
`submit_all()` for the sampler, and C21 counts how often it is hit.

**Invariants (verbatim):** op order on one stream is program order; transfer ops
never share a stream with the compute that reads their destination without an event
dependency.

**Fails by (verbatim):** falling back to a single stream when queue-family topology
does not permit more (measured at startup, P0-1), which costs overlap but not
correctness.

**Acceptance test (verbatim "done when"):**
* `--profiling` shows a non-zero overlap factor (device busy time < wall time on a
  MoE decode step) on gfx1201.
* The same test on gfx1031 documents the achieved overlap rather than assuming zero.

**Gates:**
* P0-1 (platform census, `00` §5 ASSUMPTION) — queue-family topology, number of
  separable stream types — is an ASSUMPTION until measured on the real box. C1's
  "fall back to a single stream" path is the safety net, but the overlap claim is
  not substantiated until P0-1 is measured.
* Depends on C2 (DeviceCaps: stream count, queue groups), C11 (TransferEngine::submit_*),
  C21 (OpRecord timestamps feed the profiler's device/us/idle/us columns).

**Done is not claimed until:** a profiled MoE decode step on gfx1201 shows device
busy time strictly less than wall time, and the profiler table attributes the
non-overlapping portion to the right OpClass.

**Coder notes / pitfalls:**
* Do not implement "overlap" by launching two kernels with no dependency and calling
  it overlapping. Overlap must be dependency-correct: a transfer whose destination
  is read by a compute op must have an event dependency, and the profiler must be
  able to tell the difference between "overlapped" and "unrelated kernels on two
  streams".
* The event ring is the source of truth for C21's `dev_start_ns`/`dev_end_ns`. If a
  launch is not timestamped from the event ring, C21 cannot report it correctly.
* The single-stream fallback must be a real measured path, not a compile-time
  `#ifdef`. P0-1 is an ASSUMPTION; the binary must discover the topology at runtime
  and degrade gracefully.
* Do not put a full sync in the step loop "just in case". The only full sync is at
  `submit_all()` for the sampler, and C21 counts how often it is hit — that count
  is a KPI, not a hygiene detail.

**Worked micro-example (CPU reference, host-only, sanity-check before device):**
Model a step as a list of ops with a `stream` and a `depends_on` event. Walk the
event ring in timestamp order; a transfer op is "complete" when its event is retired;
a compute op is submitted only when its stream is free and its transfer dependencies
are retired. Emit a trace of (op, stream, submit_ns, dev_start_ns, dev_end_ns,
overlap_with_prev). The assertion: on a trace with at least one transfer whose
destination is read by a later compute op, the compute's `dev_start_ns` is >= the
transfer's `dev_end_ns` (event dependency respected), and there exists at least one
pair of ops on different streams whose `[dev_start, dev_end]` intervals overlap
(overlap is real, not assumed).

**Files a coder should read before starting:**
* `docs/02-components.md` — C1 section.
* `docs/01-architecture.md` §6 (control flow).
* `docs/01-architecture.md` §5 (invariants I1–I7 — C1 must not violate I4: no
  synchronous NVMe read on the critical path; overlap is how that is achieved).
* `docs/02-components.md` — C11, C21 (the interfaces C1 calls and feeds).

**Open question for this component alone:** none — the interface and invariants are
specified. The only open thing is the measured overlap factor on gfx1201, which is a
"done when" gate, not a design question.
