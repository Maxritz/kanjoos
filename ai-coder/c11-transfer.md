# Worksheet — C11 Transfer engine

**What it owns:** all bytes that move, and the schedule they move on.

**Why it exists (from `02-components.md` — C11 section; `01` §6):** the transfer engine is the thing that moves bytes between NVMe, RAM, and VRAM, asynchronously, coalesced, aligned, prioritized,
with demand reads getting a deadline and prefetch reads not getting to starve them. It is what makes "prefetch → RAM → VRAM → attention" possible without a blocking read anywhere in the critical path.
The logical/physical split (C6: residency decides per expert; I/O moves multi-expert extents) is implemented here — fine-grained decisions, coarse-grained transfer.

**Spec lines that must hold:**
* `02-components.md` — C11 section (verbatim: the transfer paths NVMe→RAM→VRAM→attention and VRAM→D2H→RAM→NVMe(admission-controlled); the Extent struct; the TransferEngine API; the rules; the done-when).
* `04-memory-tiering.md` §4.1 (the I/O shape per platform: Linux `io_uring` + `O_DIRECT` where it wins, `RWF_HIPRI` for demand reads; Windows DirectStorage with IOCP/overlapped fallback; 4 KiB aligned,
  sequential, coalesced, 64 KiB–8 MiB extents, pre-registered buffers, many in flight; never a 4 KiB application-level random read of a payload; staging into GPU-addressable memory is still ours).
* `01-architecture.md` §6 (control flow: C11 enqueues the NVMe read → RAM → H2D for a missing expert; the attention path never issues a storage read; a hard stall is a last resort and a STALL_TIME_MS telemetry
  event, never the mechanism).

**Interface (verbatim `02-components.md` C11):**
```cpp
struct Extent { void* dst; const void* src; size_t bytes; uint64_t key; };
class TransferEngine {
public:
    TransferId submit_nvme_read (const Extent&, int priority);
    TransferId submit_h2d       (const Extent&);
    TransferId submit_d2h       (const Extent&);
    TransferId submit_nvme_write(const Extent&);
    void poll();  void flush();
};
```

**Rules (verbatim):**
* Coalesce adjacent logical objects into one physical transfer; never issue a 4 KiB random application-level read for a payload.
* 4 KiB alignment, large sequential extents, pre-registered host buffers, many outstanding requests. Extent size is tuned per host (64 KiB … 8 MiB).
* **Reads beat speculative writes**, always.
* Linux: `io_uring` with `O_DIRECT` where it wins, `RWF_HIPRI` priority for demand reads, `fallocate`d files for the cold store. Windows: DirectStorage with an IOCP/overlapped fallback; DirectStorage gives
  many-small-reads and low CPU overhead but does *not* make the SSD a VRAM extension, so the staging into GPU-addressable memory is still our job.
* Demand reads get a deadline; prefetch reads do not get to starve them.

**Acceptance test (verbatim "done when"):**
* NVMe read bandwidth reaches a stated fraction of the device's measured sequential rate with extents in the tuned range.
* H2D/D2H overlap compute — both visible in the C21 table as `transfer` device-time vs idle-time.

**Gates:**
* Depends on C6 (the directory — C11's `Extent` src/dst and key come from C6's logical→physical mapping; the coalesced multi-expert extent is what C6 provides), C7 (host tier — the pinned bounce buffer
  is what C11's H2D/D2H reads from/writes to; the warm pool is where NVMe reads land before H2D; the pinned pool cap is what bounds C11's in-flight DMA), C9 (residency — C11's transfers are what C9 enqueues
  to satisfy a `request()`; C9's `ready()` is the proof the transfer completed), C10 (prefetch — C11's prefetch transfers are what C10 enqueues; C11's deadline-aware prioritization is what makes C10's
  self-throttling real), C21 (telemetry — C11's transfer device-time vs idle-time is a C21 column; the "H2D/D2H overlap compute" acceptance is a C21 number), C22 (budget — the NVMe quota and the pinned pool cap
  come from C22's profile), C24 (platform — the platform-specific mechanism (Linux `io_uring`/`O_DIRECT`/`RWF_HIPRI`, Windows DirectStorage/IOCP) lives in C24; C11 calls C24's platform transfer primitives).
* P0-1 (platform census — `00` §5, ASSUMPTION) measures the host copy path and the NVMe sequential rate. Until it is measured on the real box, the "stated fraction of the device's measured sequential rate" and the
  extent-size defaults (64 KiB–8 MiB, tuned per host) are starting points, not measured facts. C11 must be able to report its achieved NVMe read bandwidth and its H2D/D2H overlap, so P0-1 can tune the extent size and the
  in-flight count.
* The "H2D/D2H overlap compute — visible as idle < wall in the profile table" acceptance is a C21 number. A coder must be able to point at the C21 table and show that transfer device-time < wall time (i.e. some compute ran while a
  transfer was in flight) on a MoE decode step on gfx1201. This is the same overlap factor that C1's acceptance test requires (device busy time < wall time). C11's transfers are a big part of where that overlap comes from (or
  where the idle comes from, if the transfers are not overlapped with compute).

**Coder notes / pitfalls:**
* The logical/physical split is implemented here. C6 decides per expert (which experts are resident, which are cold, which are being prefetched); C11 moves coalesced multi-expert extents. A coder must not implement "one
  transfer per expert" — that is the 4 KiB random-read failure mode. The Extent's `bytes` is a coalesced extent (64 KiB–8 MiB, tuned per host), and the `key` is the logical object(s) it contains (so C6 can map the transfer
  back to the experts it satisfies). The `src`/`dst` are the pinned bounce buffer (C7) for H2D/D2H, or the NVMe file descriptor mapping (C24) for NVMe reads/writes.
* Reads beat speculative writes, always. A coder must not let a speculative write (e.g. persisting a KV page to NVMe on a hunch) starve a demand read (a transfer that a layer is waiting on). The priority system
  (demand reads highest priority, prefetch reads lower, speculative writes lower still) must be enforceable in C11's scheduling, and the deadline-aware behavior (demand reads get a deadline; prefetch reads do not starve them)
  is what C10's self-throttling depends on.
* DirectStorage on Windows "does not make the SSD a VRAM extension, so the staging into GPU-addressable memory is still our job" (`04` §4.1). A coder must not treat DirectStorage as a direct-GPU-read mechanism. DirectStorage reads
  into host memory (the warm pool / pinned bounce buffer); the H2D from there into VRAM is still C11's job (and still PCIe latency). The value of DirectStorage is many-small-reads and low CPU overhead, not "SSD → GPU".
* 4 KiB alignment, large sequential extents, pre-registered host buffers, many in flight — these are the things that make the NVMe read bandwidth reach "a stated fraction of the device's measured sequential rate". A coder must not issue
  4 KiB random application-level reads (that is the failure mode to avoid), must align to 4 KiB, must use large sequential extents (64 KiB–8 MiB, tuned per host), must pre-register the host buffers (pinned, C7), and must keep many
  requests in flight (so the SSD's queue depth is used). The "stated fraction" is the acceptance — a coder must be able to report the achieved NVMe read bandwidth as a fraction of the measured sequential rate (P0-1's number), with the
  extent size and in-flight count that produced it.
* The deadline for demand reads is what makes C10's self-throttling real and what makes C9's "never a blocking read" real. A coder must implement the deadline: a demand read (a transfer a layer is waiting on) has a deadline; if it looks like it
  will not land in time, C10/C11 deprioritise or cancel speculative prefetch transfers so the demand transfer gets the bandwidth — and if it still will not land in time, C9 has already returned "not ready" and the scheduler has fallen back to C19.
  The deadline is the thing that makes "never a blocking read" enforceable, not a comment.

**Worked micro-example (host, sanity-check before device):**
Implement a fake TransferEngine backed by a fake NVMe (a file or in-memory store), a fake pinned bounce buffer pool (C7), and a fake C6 (logical→physical mapping with coalesced multi-expert extents). Then: (1) implement the coalesce:
  adjacent logical objects (same layer, contiguous packing) are submitted as one Extent with one `bytes` and one `key` range, not N submits of 4 KiB each — the "never a 4 KiB random read" gate; (2) implement 4 KiB alignment, large
  sequential extents (pick an extent size in 64 KiB–8 MiB), pre-registered pinned buffers, and many in flight — and measure the achieved NVMe read bandwidth as a fraction of the measured sequential rate (P0-1's number) — the "stated
  fraction of sequential rate" gate; (3) implement the priority system: demand reads (a transfer a layer is waiting on, with a deadline) are higher priority than prefetch reads, which are higher than speculative writes — and assert a demand read
  is never starved by speculative writes — the "reads beat speculative writes" gate; (4) implement the deadline for demand reads: if a demand read will not land by its deadline, speculative prefetches are deprioritised/cancelled so the demand read wins —
  and assert that if it still will not land in time, C9 has already returned "not ready" and the scheduler fell back to C19 — never a blocking read (the "never a blocking read" gate); (5) implement H2D/D2H that overlaps compute: a
  transfer's completion is an event, and a compute op can run on a different stream while the transfer is in flight — and assert the C21 table shows transfer device-time < wall time (overlap) on a MoE decode step — the "H2D/D2H overlap compute"
  gate; (6) on Windows, assert DirectStorage (or the IOCP fallback) reads into host memory (the pinned bounce buffer), and that the H2D from there into VRAM is still a separate C11 step — DirectStorage is not "SSD → GPU" (the "DirectStorage is not a VRAM
  extension" gate); (7) assert a 4 KiB random application-level read of a payload is never issued — any payload read is a coalesced extent, 4 KiB aligned — the "never a 4 KiB random read of a payload" gate.

**Files a coder should read before starting:**
* `docs/02-components.md` — C11 section.
* `docs/04-memory-tiering.md` §4.1 (I/O shape per platform; the logical/physical split; never a 4 KiB random read of a payload; DirectStorage is not a VRAM extension).
* `docs/01-architecture.md` §6 (control flow — C11 enqueues NVMe→RAM→H2D; the attention path never issues a storage read; hard stall is a last resort and a STALL_TIME_MS event).
* `docs/00-verified-facts.md` §5 (P0-1, ASSUMPTION — the host copy path and NVMe sequential rate are not yet measured on the real box; C11 must report its achieved numbers).
* `docs/08-roadmap.md` — Performance acceptance criteria (no synchronous NVMe read in the attention or expert path; H2D/D2H overlap compute — visible as idle < wall in the profile table).
* `ai-coder/c6-directory.md` (C6 — the logical→physical mapping; the coalesced multi-expert extent is what C11 moves; the `key` on the Extent).
* `ai-coder/c7-host-tier.md` (C7 — the pinned bounce buffer is what C11's H2D/D2H reads from/writes to; the warm pool is where NVMe reads land; the pinned pool cap bounds in-flight DMA).
* `ai-coder/c9-residency.md` (C9 — C11's transfers are what C9 enqueues; C9's `ready()` is the proof the transfer completed; the deadline is what makes "never a blocking read" real).
* `ai-coder/c10-prefetch.md` (C10 — C11's prefetch transfers are what C10 enqueues; C11's deadline-aware prioritization is what makes C10's self-throttling real).
* `ai-coder/c22-budget.md` (C22 — the NVMe quota and pinned pool cap come from C22's profile).
* `ai-coder/c24-platform.md` (C24 — the platform-specific transfer primitives: Linux `io_uring`/`O_DIRECT`/`RWF_HIPRI`, Windows DirectStorage/IOCP; the staging-into-GPU-addressable-memory-is-ours rule).
* `ai-coder/c21-profiler.md` (C21 — the transfer device-time vs idle-time column; the "H2D/D2H overlap compute" acceptance is a C21 number).
