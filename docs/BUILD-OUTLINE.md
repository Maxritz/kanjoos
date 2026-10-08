# Build outline — MoE GGUF streaming engine

**Status:** draft, reasoned from `docs/00…09`, `ai-coder/c1…c24`, and the
measured 2026-10-07 rocWMMA results.
**Authoritative companions:** [AGENTS.md](../AGENTS.md), [CLAUDE.md](../CLAUDE.md).
**Contract source:** the master spec (`AI_AGENT_MOE_STREAMING_ENGINE_SPEC.md`).

---

## 0. How to read this

The master spec names **Phase 0…9**. This outline keeps that numbering, splits
what the spec bundles (attention and KV are separate economics and separate
worksheets here), and inserts **one phase the spec assumes but this repository
does not have**: *Phase 0.5 — the build substrate*. There is no way to compile
the engine today. That is the first real blocker, and it is not a design one.

Every phase below states four things:

| field | meaning |
|---|---|
| **worksheet** | the `ai-coder/c*.md` contract that owns the work |
| **deliverable** | the artifact that must exist |
| **exit** | a *measurable* condition — a number, a bit-identity, or a refusal |
| **measurement** | the number that goes in `CODING-LOG.md` with provenance |

Nothing is "done" because it compiles.

---

## 1. The framing that governs every phase

Three decisions, owned by three different components
(spec §2, `docs/02-components.md`):

| decision | granularity | owner |
|---|---|---|
| **logical residency** — which expert is needed? `ExpertId{layer, expert}` | **fine** | C18 router → C23 scheduler |
| **physical residency** — where is it? `NVME / RAM / VRAM / IN_FLIGHT / ABSENT` | **fine** | C9 residency manager |
| **physical I/O** — what bytes move? a **coalesced multi-expert extent** (2–32 MiB, normally a layer slab) | **coarse** | C11 transfer engine |

> residency decision = fine grained · I/O decision = coarse grained

**The single most expensive fact in the project** (measured, `00` §3.1):

```
one W4 g128 expert = 2.47 MB
its arithmetic at peak                    =   7.68 us
its PCIe transfer at measured small-obj BW = 181 us
```

Expert movement is ~23× its own compute. Every phase below is therefore ordered
by *how much unnecessary movement it removes*, not by how fast it makes a GEMM.

---

## 2. What exists today (honest gap table)

| layer | state | evidence |
|---|---|---|
| device caps / backend | **compiled** — `src/device/` (C2), the only built C++ | `src/device/CMakeLists.txt` |
| W4 g128 pack format + stride | **exists** | `kernels/knj_pack.h` |
| W4 expert GEMM (SIMT, tier A) | **passes** | tier A `7 checks, 0 failures` |
| gfx1201 matrix backend | **chosen and probe-verified** — rocWMMA 2.2.1 | `records/2026-10-07_rocwmma_model_ops.txt` |
| expert GEMM on rocWMMA | **correct in a bench driver**, not a production kernel | `tools/bench/rocwmma_moe.hip` |
| attention on rocWMMA (prefill + decode) | **correct in a bench driver**, not a production kernel | `tools/bench/rocwmma_attn.hip` |
| host oracle harness | **exists** | `tests/host_oracle.h`, tier A |
| arch guard | **exists and enforced** | `tools/bench/knj_wmma_guard.h` |
| machine audit tools | **exist** | `tools/{check_docs,kvroof,isa_probe,i7,route,doctor,ci}` |
| **GGUF reader** | **absent** | — |
| **runtime above C2** (C1, C4–C24) | **absent** | `find src -type f` → 4 files |
| **device-code build target** | **absent** — no CMake target compiles `kernels/` | `grep add_subdirectory CMakeLists.txt` |
| **host/device shared runtime** | **absent** | — |
| **model dir with a real GGUF** | **absent** — only a `config.json` | `models/qwen3-30b-a3b/` |

So the project is: a *very* well-measured substrate with real kernels-in-drivers,
and no engine. The outline below builds the engine around the substrate.

---

## 3. Phase 0 — repository and hardware audit

**Worksheet:** —
**Deliverable:** `kanjoos doctor` emits the `docs/07-build-platforms.md` contract
block, every line measured, on gfx1201/Win11.
**Exit:** `tools/doctor/kanjoos_doctor.sh` (or its compiled successor) prints
VRAM, usable VRAM, RAM, free RAM, pinned pool, ReBAR, PCIe, NVMe model,
NVMe bandwidth, GPU copy bandwidth, queue topology, arch, kernel capabilities,
selected profile — **all measured, none defaulted silently**, with a provenance
label per line (spec §29 `MEASURED / CONFIGURED / AUTOTUNED / DEFAULTED`).
**Measurement:** NVMe sequential + 2.47 MB random-read bandwidth; PCIe large +
small-object bandwidth; the existing 13.4–14.7 GB/s small-object figure is
re-verified or corrected here, because the whole design rests on it.

Work already done for this phase: `tools/doctor/`, `tools/isa_probe/`,
`tools/kvroof/`. What is missing is compiling them into one report.

---

## 4. Phase 0.5 — the build substrate  ·  **the real first blocker**

**Worksheet:** C24 (`ai-coder/c24-platform.md`), `docs/07-build-platforms.md`
**Deliverable:** a CMake build that produces, from one configuration:

```
libkanjoos_core        C1, C4–C24 as pure host C++ (no arch, no #ifdef)
libkanjoos_kernels     device code for the selected KNJ_ARCH
kanjoos                the CLI
kanjoos_doctor         the diagnostic
tests_unit / tests_integration
```

**Exit:** on Windows 11 / VS + `KNJ_ARCH=gfx1201`:
`cmake --build` succeeds; `ctest` runs the host-oracle unit tests; `kanjoos
doctor` exits 0; and a deliberate arch mismatch makes a device test exit 6.

**Why this is hard here specifically:**

* `cmake/rocm.cmake` deliberately does **not** call `enable_language(HIP)` —
  CMake's HIP module is incompatible with this install. The build must invoke
  `hipcc` through a custom command, the same way `tools/bench/run_bench.sh`
  does, with `-nogpulib` and the `+wavefrontsize32` flag.
* `--rocm-path` must be `G:/ROCM10RT-gfx1201/lib/llvm` (not the root) for any
  translation unit that includes rocWMMA, and `-std=c++17` is mandatory for
  those same units. Two settings that look like preferences and are not.
* The arch must reach the device code through the **environment**, not `-D`.
* rocWMMA sources must be a *separate* compile rule from non-rocWMMA sources,
  or the flags leak and change what the other kernels mean.

**Measurement:** build wall time for a clean `gfx1201` configure+build; the
list of sources that needed the rocWMMA rule (should be exactly the ones that
include `<rocwmma/rocwmma.hpp>`).

---

## 5. Phase 1 — GGUF model / index substrate

**Worksheet:** C4 (`ai-coder/c4-loader.md`), C6 (`c6-directory.md`), C13 (`c13-radix.md`)
**Deliverable:** a GGUF reader that opens a real MoE GGUF and produces:

```
ModelFingerprint   immutable; covers tensor metadata, quant formats, tokenizer,
                   RoPE, and every version that can change kernel output (I5)
TensorDirectory    name -> {type, dims, offset, size}, honouring general.alignment
                   (32 / 64 / 128)
ExpertDirectory    ExpertId{layer, expert} -> ExpertObject (C6)
ColdExtentIndex    coalesced multi-expert extents, ascending offset
ModelGeometry      read from metadata, never hardcoded (spec §46)
```

**Entry point:** `engine --model model.gguf` (spec §5) — the user must not need
to know an internal store exists.

**Exit / startup invariant (spec §5):** a 40–100+ GiB MoE GGUF opens **without**
reading its expert payload, and without a matching VRAM or RAM allocation.
Load time is proportional to metadata/trunk/index work, not model size.

**Exit / correctness:** the expert directory's byte ranges, summed, account for
the expert tensors exactly — no gaps, no overlaps, no double-counting.

**Measurement:** cold-start time on the measured disk for a stated model size;
resident RSS at open; bytes read at open. This is the number `c4-loader.md`'s
"done when" asks for and does not yet have.

**Note:** `_PROVENANCE.txt` + truthful `tensor_count` matter here — the repo
currently has **one** model config and no GGUF. Getting a real Qwen3-30B-A3B
GGUF onto this machine is a Phase-1 prerequisite, not a detail.

---

## 6. Phase 2 — expert store

**Worksheet:** C5 (`c5-compiler.md`), C6 (`c6-directory.md`)
**Deliverable:** `model.gguf.kanjoos/` — manifest, expert index, packed expert
objects, checksums, tuning data — with the crash-safe publication protocol of
spec §32:

```
write temp object → checksum → fsync/commit → atomic publish → journal update
```

and on restart: replay journal, discard incomplete objects, quarantine corrupt
objects, rebuild missing index entries, preserve committed objects, **never
expose a partially written expert as valid**.

**Exit:** kill the writer mid-publish (all of: before checksum, during
checksum, between fsync and publish, during journal update) → next open
recovers, quarantines, or rebuilds; **no partially-written expert is ever
readable**.

**Exit / invalidation (spec §6):** change any of model fingerprint, tensor
metadata, quantisation format, compiler version, packing version, kernel
ABI, arch-specific pack, or relevant runtime config → the stale cache is
rejected, never used silently.

**Format:** W4 g128 first (`docs/00-verified-facts.md`, spec §9). Architect the
pack interface for W3 but do not claim W3 quality until its 81.5 dB SNR gate
passes. Metadata is exactly **3 B/group**.

**Measurement:** pack throughput; bytes/weight including metadata for W4 g128
(0.5234) and W3 g128 (0.3984); cache-invalidation test results.

---

## 7. Phase 3 — RAM tier

**Worksheet:** C7 (`c7-host-tier.md`), C22 (`c22-budget.md`)
**Deliverable:** two pools with different natures.

| pool | pageable? | purpose | W4 g128 starting cap |
|---|---|---|---|
| `WarmPool` | yes | retain cold/recent experts; reduce repeated NVMe reads | remainder |
| `PinnedPool` | no (bounded) | DMA staging only | 2 GiB @24–32 GiB RAM · 3 GiB @48 · 6 GiB @96 |

2 MiB aligned, 2–32 MiB chunks.

**Exit / hard invariant:** **never pin the entire warm pool**, and **never pin
the entire model**. RSS and pinned bytes stay inside the reported budget under
pressure. Dynamic shrink is mandatory, not optional (`c7-host-tier.md`).

**Exit:** under injected memory pressure the warm pool shrinks and the run
stays correct; the pinned cap is never exceeded.

**Measurement:** RSS high-water mark; pinned bytes high-water mark; warm-pool
hit rate; bytes saved per NVMe read avoided.

---

## 8. Phase 4 — VRAM slot allocator

**Worksheet:** C8 (`c8-slot-allocator.md`)
**Deliverable:** **one** VA reservation, sub-allocated into fixed-size
`ExpertSlot`s. States: `FREE, LOADING, RESIDENT, IN_USE, EVICT_PENDING`.
`ExpertSlot{layer, expert, vram_offset, byte_size, generation, use_count, state}`.

**Exit / hard rule (spec §10):** a slot is recycled **only after the last GPU
operation that used it has completed**, proven by a HIP event — not a
wall-clock delay:

```
submit kernel → record event → retire slot → event completes → slot reusable
```

The prohibited shape is `free(slot); launch_kernel_using(slot);`.

**Exit:** an eviction stress test produces **zero** use-after-free (Linux:
ASAN/UBSAN in CI, `docs/07-build-platforms.md` CI matrix); occupancy ≥ 80%;
no per-expert `hipMalloc`.

**Measurement:** slot allocation latency; eviction-stress pass/fail; achieved
tile occupancy; `hipMalloc` call count over a full run (**must be a small
constant**).

---

## 9. Phase 5 — transfer engine

**Worksheet:** C11 (`c11-transfer.md`), C24 (`c24-platform.md`)
**Deliverable:** one abstraction for all movement (spec §14):

```cpp
class TransferEngine {
  TransferId submit_nvme_read (const Extent&, int priority);
  TransferId submit_h2d        (const Extent&);
  TransferId submit_d2h        (const Extent&);
  TransferId submit_nvme_write (const Extent&, int priority);
  void poll();  void flush();
};
```

Priorities: `DEMAND > RECOVERY > PREFETCH > WRITEBACK`.
Windows: DirectStorage preferred, IOCP/overlapped fallback.
Linux: `io_uring` + `O_DIRECT`.

**Exit / the two things that matter (spec §14):**
1. transfers **overlap** compute — measured as non-zero GPU idle reduction with
   STEADY `STALL_TIME_MS`;
2. **a prefetch never starves a demand transfer** — provable by construction
   from the priority queue *and* by an injected saturation test.

**Exit / I/O shape:** no application-level random 4 KiB expert reads. All reads
aligned, contiguous extents, multiple outstanding, bounded pinned buffers.

**Measurement:** NVMe bytes/token; RAM→VRAM bytes/token; achieved small-object
bandwidth; demand-vs-prefetch completion ordering under saturation; in-flight
depth actually used.

---

## 10. Phase 6 — residency manager

**Worksheet:** C9 (`c9-residency.md`), C10 (`c10-prefetch.md`), C12 (`c12-admission.md`)
**Deliverable:** the single source of truth for where an expert is. States:
`ABSENT, NVME_RESIDENT, LOADING_NVME, RAM_RESIDENT, LOADING_RAM, VRAM_RESIDENT,
PREFETCHED, EVICTING_VRAM, EVICTING_RAM, PINNED, INVALID`. Kernels only ever
ask `ready()`. Every transition emits telemetry.

The compute path must **never** contain:

```cpp
read_nvme(); wait();
```

It contains:

```cpp
request(expert);
if (ready(expert)) execute();
else                schedule_fallback_or_wave();
```

Admission score (spec §17, weights tunable **only through measurement**):

```
0.30*recency + 0.25*frequency + 0.20*reuse_prediction
+ 0.15*shared_prefix_value - 0.10*transfer_cost
```

**Exit (spec §50, functional):** **zero synchronous NVMe reads in the inference
critical path.** Provable by an assertion that the critical path never blocks on
an NVMe completion.

**Exit:** zero synchronous NVMe reads on the critical path; a resident expert is
never evicted while a submitted GPU operation can still read it.

**Measurement:** expert hit rate; residency-state histogram per token; miss
count and reason (spec §49).

---

## 11. Phase 7 — router

**Worksheet:** C18 (`c18-router.md`)
**Deliverable:** the model's own router, exactly: correct weights, correct bias,
**both known metadata spellings for `exp_probs_b.bias`**, exact top-k,
deterministic tie handling.

Record `top-1 agreement`, `top-k agreement`, `Jaccard`, `logit RMSE`,
`router margin`.

**Exit:** routing matches a trusted reference on a real model, on both bias
spellings, with a stated tie-break rule. Quantisation must not silently change
routing (I3).

**Measurement:** agreement metrics above; **new experts per token** (the
existing 59.1% / 80.5% figures) re-measured on the real GGUF so prefetch design
rests on current numbers, not on the earlier probe.

---

## 12. Phase 8 — grouped expert execution  ·  rocWMMA

**Worksheet:** C17 (`c17-expert-gemm.md`), C3 (`c3-autotuner.md`)
**Deliverable:** the grouped executor (spec §19):

```
hidden → router → exact top-k → group tokens by expert
      → determine resident/missing → prefetch/transfer missing
      → form waves → grouped gate/up GEMM → activation
      → grouped down GEMM → weighted accumulation → residual
```

**Prohibited:** one GPU kernel launch per token per expert.
**Required:** waves. All selected contributions accumulated (I1).

**rocWMMA's role (decided, measured):** rocWMMA is the gfx1201 matrix backend.
`tools/bench/rocwmma_moe.hip` already proves the shape:

* grouped gate/up/down as rocWMMA GEMMs over W4 g128 unpacked to f16 fragments;
* 16×16×16 wave32 fragments are the only shape the gfx12 builtins have;
* the zero point is applied **once** at fragment materialisation, so the WMMA
  path carries **no per-group correction term** — a structural advantage over
  the int8-dot path (no de-interleaved activation layout, no accumulator
  headroom question);
* `layout_t` is per-operand: this is where the attention-side lesson bites.

Do **not** hand-roll `_gfx12` operands and do **not** inline the builtins.

**Exit:** full-output correctness against the host oracle for a real model
forward pass — every output, max abs error, relative RMSE, zero/unwritten
outputs, mismatching indices. Tolerance: expert GEMM fp32 `1e-4`, unpack
bit-exact.

**Exit:** a real number. The current `0.0413 ms` is a tiny-shape correctness
datapoint, **not** a throughput claim.

**Measurement:** expert GEMM µs for the real per-layer shapes; achieved
FLOP/s vs the derived 39.3 TFLOP/s packed-f16 peak; wave count per token; cells
over tolerance (must be 0).

---

## 13. Phase 9 — prefetch

**Worksheet:** C10 (`c10-prefetch.md`)
**Deliverable:** predictors, shipped incrementally. **L1 first** (previous-token
routing for the same layer). L0 current router, L2 per-layer history, L3
cross-layer, L4 activation matrix, L5 adaptive horizon.

Every prediction carries
`Prediction{candidates, confidence, horizon_layers}`.

**Exit / hard rule:** **prefetch is never a correctness dependency.** A wrong
prediction may waste bandwidth; it must never change model output. Test: force
all predictions wrong → output bit-identical to no-prefetch.

**Exit:** L1 alone measurably reduces demand misses on a real model, at a stated
horizon and extent. Implement L3/L4/L5 only if measurement justifies them.

**Measurement:** prefetch hit rate; bytes wasted per misprediction; demand
misses before/after; the `late/ontime` telemetry split (spec §49).

---

## 14. Phase 10 — KV engine and attention  ·  rocWMMA

**Worksheet:** C14 (`c14-kv-cache.md`), C15 (`c15-kv-codecs.md`),
C16 (`c16-attention.md`), `docs/09-kv-engine-architecture.md`

This is where **the product feature** lives. KV is not another expert cache: it
is unbounded, has 100% miss every token forever, and cannot be cached.

**Deliverable:** the four tiers `T0 VGPR/LDS · T1 VRAM · T2 pinned RAM · T3 NVMe`,
the three session classes `A resident / B spilled (priced) / C suspended`, the
codec ladder (0–2K FP8 → 2K–32K INT8 per-group → 32K–end INT4 per-group +
coarser RoPE buckets), and the attention data path with **pipelined DMA** —
chunk *i+1* in flight while chunk *i* is consumed. Running the regimes as
separate phases costs 2–3×.

**Index:** chunked-hash-tree radix (C13).

**rocWMMA's role:** `tools/bench/rocwmma_attn.hip` already carries both
regimes:
* prefill `Q·Kᵀ` (B `col_major`, K stored `[n][d]`) and `P·V` (B `row_major`) —
  **two different `layout_t` inside one kernel, both required**;
* decode, with output rows the launch *covers but does not own* preserved in
  place (measured: 15360 such cells, 0 clobbered).

**Exit / I7 (spec §22, `docs/09`):** tier movement is **bit-for-bit identical**.
A lossy codec is a **declared precision reduction**, never called a tier move.
Canonical reduction order, exactly:

```
ascending pages → ascending keys within page → per-page partial
→ left-to-right merge → global fp32 max first
```

**Exit:** prefill and decode reported **separately** (spec §23) — they are
different workloads and must never share a headline number.

**Exit / honesty:** a session that cannot pay for its KV **refuses or prices**;
it never silently demotes. No "unlimited context" claim.

**Measurement:** KV read/write bytes per token; µs per regime; bytes_per_token
at each codec step; the reboot/reload bit-identity result (`RadixKV reload
bit-identical`).

---

## 15. Phase 11 — speculation

**Worksheet:** C20 (`c20-speculation.md`), `docs/05-speculation.md`
**Deliverable:** **MTP first** — the head is already resident via
`nextn_predict_layers`, so it costs ~0 extra I/O and is on whenever the model
has it. Then one loader serving DFlash / DFlash2 / DSpark
(`arch = "dflash"`, `dflash.*` metadata, `blk.N.*`).

The `decoder_arch` compatibility check is **mandatory**: a mismatched drafter is
**refused at load**, not run.

**Exit / I6:** bit-identical output under a fixed seed, with speculation on and
off. Reports `experts touched / accepted token` and reduces draft width when the
union grows (DSpark's confidence head gates drafting width against cold-expert
stalls).

**Measurement:** acceptance rate; extra experts touched per accepted token; I6
bit-identity result.

---

## 16. Phase 12 — profiling and observability

**Worksheet:** C21 (`c21-profiler.md`), `docs/06-profiling.md`
**Deliverable:** the fixed output contract —

```
component | ops | %dev | dev us | idle us | host us
```

plus the floor line, residency block, transfer block, cold-store block.
`--profiling=off|counters|full|table|json|csv|no-subtract`, `--profile-dir`,
`--profile-floor 256`, `--profile-warmup 8`, `--profile-detail=layer`.

Component taxonomy adds `expert-gemm`, `expert-gemm-unpack`, `router`,
`kv-read`, `kv-write`, `kv-quant`, `spec`.

**Exit / the profiler's own acceptance test (`docs/06` §9 #3):** an injected
cold run must show **transfer dev time + consumer idle + non-zero
`STALL_TIME_MS`** — the proof the profiler can see the failure it exists to
find.

**Exit / overhead budget:** if `full` changes throughput by more than **3%**,
that is a profiler bug.

**Exit / the question it must answer:** *why is this token slow?* — with
measured evidence, distinguishing compute-bound / PCIe-bound / NVMe-bound /
KV-bound / scheduler-bound.

---

## 17. Phase 13 — session scheduler and server

**Worksheet:** C23 (`c23-session.md`), `docs/02-components.md` C23
**Deliverable:** a **thin** OpenAPI-compatible server over the runtime:

```
HTTP → request parser → session scheduler → runtime → model/session state
```

The server contains **no inference logic**, and the runtime works without it.

**Exit:** prefix homogeneity beats batch size (`batch_policy =
"prefix_homogeneous"`, `max_batch = 8`) — measured, using **actual model/session
state**, not prompt similarity. Batching must never change model semantics.

**Multi-session rules (spec §33):** expert payloads are immutable and shared;
slot ownership is event/reference safe; KV blocks have explicit ownership and
refcount; cancellation cannot free data another session still references;
eviction cannot invalidate a shared block; session destruction releases
references deterministically. **No global LRU without session-aware refs.**

---

## 18. Phase 14 — portability, and gfx1031

**Worksheet:** C24 (`c24-platform.md`), `docs/03-kernels.md`, `docs/07-build-platforms.md`

Order: **gfx1201/Win11** → **gfx1201/Linux** → **gfx1031/Win11 + Linux**.

| target | matrix path | notes |
|---|---|---|
| gfx1201 Win11 | rocWMMA 16×16×16 wave32 | DirectStorage / IOCP; `VirtualLock` |
| gfx1201 Linux | rocWMMA | `io_uring` + `O_DIRECT`; `mlock`; ReBAR via sysfs |
| gfx1031 both | **no WMMA** (`v_pk_add_f16` / `v_pk_fma_f16`, `sdot4` → `v_dot4c_i32_i8`) | W4 is `v_dot8_i32_i4` consuming packed nibbles **directly — no unpack step** |

rocWMMA refuses gfx1031 at compile with
`static assertion failed: Unsupported architecture` — correct, and the gate
already reports it as a **refusal, not a pass**.

**Exit:** every arch difference lives in `src/device/`, `kernels/`,
`src/platform/`, or `cmake/arch.cmake`. No `#ifdef __gfx*__` / `#ifdef _WIN32`
above those four. CI matrix (build Win/Linux × 1201/1031, warnings-as-errors,
isa-probe expectation match, unit, integration, bench no-regression >10%, Linux
stress under ASAN/UBSAN).

**Exit / ReBAR (spec §30):** detected, never assumed. If unavailable: do not
claim direct host-memory access, do not select a path that depends on it, fall
back to staged transfers, and **print the capability and the selected path**.

---

## 19. Benchmark ladder — always run all six

Attribution requires controlling residency conditions (spec §44). Never compare
only before/after.

| step | configuration | establishes |
|---|---|---|
| **A** | all required experts resident | the compute-only ceiling |
| **B** | VRAM slots + **synchronous** expert transfer | the raw streaming penalty |
| **C** | + RAM warm pool + asynchronous transfer | the value of asynchrony |
| **D** | + prefetch | the value of prediction |
| **E** | + advanced admission | the value of the score function |
| **F** | + KV optimisation | the value of the KV engine |

Every recorded state must include: model · quantisation · context · batch · GPU ·
arch · driver/toolchain · VRAM · RAM · ReBAR · NVMe · configuration · revision.

Reported as a decomposition (spec §43): TTFT, prefill tok/s, decode tok/s,
p50/p95/p99 token latency, expert hit rate, prefetch hit rate, NVMe bytes/token,
RAM→VRAM bytes/token, VRAM bytes/token, and µs for NVMe read / H2D / expert GEMM
/ attention / router / scheduler idle / CPU fallback.

**Never report a single aggregate "tokens/s" as the only metric.**

---

## 20. Failure policy (spec §31) — no failure is unlabelled

| class | meaning | example |
|---|---|---|
| `RETRY` | transient I/O or device event | a dropped completion |
| `DEGRADE` | slower but correct | prefetch miss → demand transfer; GPU expert unavailable → CPU fallback |
| `REFUSE` | the request cannot satisfy hard requirements | unsupported geometry, no usable GPU, too little memory for minimum execution, corrupt model |
| `INVALIDATE` | cache / index / packed object stale or corrupt | fingerprint mismatch |
| `FATAL` | internal invariant violation or unrecoverable device failure | I1 violated |

**Never hide a fatal correctness failure behind a performance fallback.**

---

## 21. Decisions owed before the phases that need them

| # | decision | blocks | why it is not obvious |
|---|---|---|---|
| 1 | **the build substrate** — how `kernels/` gets compiled (custom command vs CMake HIP language) | Phase 0.5, everything | `cmake/rocm.cmake` explicitly refuses `enable_language(HIP)` |
| 2 | **`KNJ_ARCH` default** — `CMakeLists.txt` says gfx1031 primary, the stated target order is gfx1201 first | Phase 0.5 | one-line change, but it silently decides which arch the CI baseline means |
| 3 | **a real GGUF on this machine** | Phase 1 | there is only a `config.json`; several `00-verified-facts` numbers are from a probe, not a model |
| 4 | **tier B repair** — `--rocm-path=<root>/lib/llvm -nogpulib` | Phase 0.5 | verified by hand; still refuses everywhere in `run_bench.sh` |
| 5 | **the doc corrections** — `07` §3.2, `00` §7.4a, `c2`'s gate, `README`'s "five configs" | nothing, but it misleads every reader | they are *written down* and therefore cited |
| 6 | **W3's SNR gate owner** | Phase 2 | capacity claim vs bandwidth claim; the distinction is I7's cousin |

---

## 22. Definition of done

Mapped to spec §50. The project is complete only when **all** hold:

**Functional** — a real GGUF MoE model loads; metadata discovered dynamically;
the full expert bank is not required in VRAM; experts can live on NVMe, warm
experts in RAM, hot experts in VRAM; router exact; grouped execution correct;
KV correct; attention correct; multi-token generation works; **a missing expert
never disappears**.

**Performance** — transfers asynchronous; demand outranks prefetch; transfer
and compute overlap; hit rate, prefetch accuracy, NVMe and H2D throughput all
measurable; token latency **decomposed**; low-VRAM profiles tested.

**Reliability** — corruption detected; cache invalidation works; crash recovery
works; slot lifetime event-safe; multi-session references safe; cancellation
safe.

**Portability** — platform differences isolated; Windows and Linux paths
explicit; capability detection real; unsupported kernels rejected cleanly.

**Correctness** — every GPU output has a reference comparison; tier movement
obeys I7; negative controls fail; arch mismatch refuses to run; **no silent
fallback changes semantics**.
