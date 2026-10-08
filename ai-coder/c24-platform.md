# Worksheet — C24 Platform layer

**What it owns:** Windows 11 and Linux differences, and the ROCm packaging.

**Why it exists (from `02-components.md` — C24 section):** every platform difference
goes behind an interface in C24. No `#ifdef _WIN32` above this layer, ever.

**Spec lines that must hold:**
* `02-components.md` — C24 section (verbatim Windows and Linux bullets, rule, done-when).
* `01-architecture.md` §3.2 (RAM: WARM_KV_LIMIT, PINNED_POOL, dynamic shrink order).
* `04-memory-tiering.md` §1 (ReBAR prerequisite), §4.1 (I/O shape per platform),
  §3 (RAM profiles, pinned pool caps).
* `00-verified-facts.md` §5 (P0-1: platform census — ReBAR aperture, PCIe gen/width,
  NVMe model + sequential + random read, host copy path, RAM profile, ReBAR state,
  queue topology — currently ASSUMPTION, not readable over non-interactive SSH).

**Windows 11 (verbatim):**
* DirectStorage preferred, IOCP/overlapped fallback.
* `VirtualLock` for the pinned pool; `FILE_FLAG_SEQUENTIAL_SCAN` / `SetFileInformationByHandle`
  for read-ahead.
* hipcc.bat on PATH; `--offload-arch` per target; no `HSA_OVERRIDE_GFX_VERSION` needed
  (both targets are natively supported in the Linux channel; Windows gfx1031 is best-effort
  and must say so).
* ReBAR verified through HIP device properties, not assumed.

**Linux (verbatim):**
* `io_uring` + `O_DIRECT`, 4 KiB aligned, large extents, pre-registered buffers.
* `madvise(MADV_WILLNEED | MADV_HUGEPAGE)`, `mlock` bounded by RLIMIT.
* ReBAR via `/sys/class/drm/card*/device/resource` and the `amdgpu` resize state.
* The gfx1031 **kernel packs** (`blas/fft/rand` kpack) are installed and used as the
  microbenchmark baseline; the `torch`/`torchvision`/`rccl` packs are irrelevant to the
  engine and are documented as such so nobody tries to link them.

**Rule (verbatim):** every platform difference goes behind an interface in C24. No
`#ifdef _WIN32` above this layer, ever.

**Acceptance test (verbatim "done when"):**
* The same binary and the same test suite pass on Windows and Linux for both targets.
* A single `kanjoos doctor` command prints the resolved platform facts (ReBAR state,
  pinned pool, extents, queue topology, capability table) that C21 then uses in its report.

**Gates:**
* P0-1 is an ASSUMPTION (`00` §5) — ReBAR aperture, PCIe gen/width, NVMe model +
  sequential + random read, host copy path, RAM profile, ReBAR state, queue topology.
  C24's `kanjoos doctor` is the thing that resolves these; until it runs on the real box
  over a capable channel, the platform-specific numbers in `04` (pinned pool caps, extents,
  read-ahead mechanism) are starting points, not measured facts.
* Windows gfx1031 is best-effort and must say so (`02-components.md` C24 bullet). Do not
  claim equal support.

**Coder notes / pitfalls:**
* ReBAR is a prerequisite, not an optimisation (`04` §1). If C24 reports ReBAR off and the
  model does not fit in the reported aperture, `kanjoos doctor` refuses to start a decode
  session — one line, with the BIOS setting. Silent thrashing here is the worst possible
  outcome (looks like a performance bug). The "allocate more than the reported aperture and
  observe it fail cleanly, not corrupt" functional check is part of this.
* The pinned pool cap is a hard bound, not a hint: 2 GiB (24/32 GiB hosts), 3 GiB (48), 6 GiB
  (96) (`04` §3). The platform layer must enforce this cap and fail loudly if the host cannot
  provide it (Windows `VirtualLock` on a pre-committed reserve; Linux `mlock` + `MAP_POPULATE`
  with an RLIMIT check up front so a missing capability fails at init, not mid-transfer).
* Do not pin the whole machine. The pinned pool exists to make DMA efficient in bounded chunks,
  not to make the whole machine fast. A 96 GiB host pinned to "simplify transfers" is a bug, not
  a configuration.
* The "host copy path" is part of P0-1 and is an ASSUMPTION until measured. The transfer engine
  (C11) and host tier (C7) depend on it; C24's doctor is what resolves it. Until then, the
  "~19 GB/s cached file reads" note in `01` §3.2 and the "~20 GB/s PCIe pinned H2D" assumption
  in `01` §4 are labelled assumptions and must stay labelled.
* gfx1031's `hipcc` is broken on this machine (`00` §2, §8.7: `/g/ROCM10RT-gfx1031/bin/hipcc.exe`
  execs `G:\ROCM10RT-gfx1201\lib\llvm\bin\clang.exe` — a different, live ROCm tree). So gfx1031 has
  a working `clang` for compiling and **no working `hipcc` for linking**. Tier C on gfx1031 is a
  real gap on this machine, and `kanjoos doctor` should keep reporting it as one rather than as
  "absent". The gfx1031 **kernel packs** (`blas/fft/rand` kpack) are the microbenchmark baseline;
  the `torch`/`torchvision`/`rccl` packs are irrelevant.

**Worked micro-example (host, sanity-check before device):**
Implement a fake platform with a configurable ReBAR state, pinned-pool cap, and queue-topology
description. Run `kanjoos doctor` against it and assert: (1) ReBAR-on + model-fits-aperture →
  doctor reports "ready" with the resolved facts; (2) ReBAR-off + model-does-not-fit-aperture →
  doctor refuses with one line naming the BIOS setting; (3) allocated-more-than-aperture → fails
  cleanly (not corrupt); (4) pinned-pool request above the cap → fails loudly at init, not
  mid-transfer; (5) the doctor output prints the same categories (ReBAR state, pinned pool, extents,
  queue topology, capability table) that C21's report consumes.

**Files a coder should read before starting:**
* `docs/02-components.md` — C24 section.
* `docs/04-memory-tiering.md` §1, §3, §4.1.
* `docs/01-architecture.md` §3.2 (RAM budget arithmetic).
* `docs/00-verified-facts.md` §5 (P0-1, marked ASSUMPTION).
* `docs/00-verified-facts.md` §2, §8.7 (gfx1031 toolchain state on this machine).
