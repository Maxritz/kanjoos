# Component reference — what each piece is, and which doc describes it

One page. Updated as docs are added. If a component is listed here and its doc
is missing, that is the kind of gap §7.2 of `08-roadmap.md` is about.

---

## The 24 components (C1–C24)

Each row: **id — name**, one-line what-it-is, the doc that specifies it, and the
one artifact or measurement that settles it.

| id | name | what it is | spec doc | what settles it |
|---|---|---|---|---|
| C1 | Runtime core and stream scheduler | op graph, stream set, event ring, submission cadence; the thing a decode step is routed through | `02-components.md` §C1 | `--profiling` shows a non-zero overlap factor on gfx1201 |
| C2 | Device abstraction layer | arch split; `DeviceCaps`, one `Backend` per arch, capability probing | `02-components.md` §C2 | same binary on either box selects the right backend and prints the capability table |
| C3 | ISA probe and kernel autotuner | compile-time probe (`tools/isa_probe/`) plus a runtime benchmark sweep writing a per-arch cache | `02-components.md` §C3 + `tools/isa_probe/` | `tools/isa_probe/run_isa_probe.sh` (exit 0 = measured) |
| C4 | Model loader | GGUF on disk → resident buffers + cold index; mmap, alignment, read hints, no expert reads at load | `02-components.md` §C4 | load time bounded by trunk size, not model size |
| C5 | Expert compiler (offline) | canonical checkpoint → kernel-native packed expert store; quantisation pipeline + manifest | `02-components.md` §C5 | re-running on same checkpoint produces identical manifest hash |
| C6 | Expert directory and NVMe object store | addressable unit the residency manager moves; per-expert logical, multi-expert physical I/O extents | `02-components.md` §C6 | crash during write leaves committed objects intact |
| C7 | Host tier: warm pool + pinned DMA staging | RAM tier; pageable warm pool + bounded pinned DMA pool | `02-components.md` §C7 + `04-memory-tiering.md` §2/§3 | RSS inside declared budget under 96 GiB-profile synthetic load |
| C8 | GPU hot expert slot allocator | VRAM slots for experts; slab allocation, event-bounded lifetimes | `02-components.md` §C8 | no use-after-free under eviction stress with sanitiser build |
| C9 | Residency manager | where every expert and KV page currently is; state machine + async transitions | `02-components.md` §C9 + `04-memory-tiering.md` §5 | zero synchronous NVMe reads in attention path on a full trace |
| C10 | Prefetch predictor | layered predictor (L0–L5) that makes the RAM tier pay for itself | `02-components.md` §C10 + `05-speculation.md` §1.3 | measured prediction accuracy per model; adaptive horizon beats any fixed horizon |
| C11 | Transfer engine | all bytes that move: NVMe↔RAM, RAM↔VRAM, async + coalesced | `02-components.md` §C11 + `04-memory-tiering.md` §4 | NVMe read bandwidth at a stated fraction of device sequential rate; H2D/D2H overlap compute |
| C12 | Admission and eviction | placement score + NVMe admission control; what gets demoted and what is written to NVMe | `02-components.md` §C12 + `04-memory-tiering.md` §5.3 + `09-kv-engine-architecture.md` §8 | SSD write bytes/token under bound on prefix-heavy workload; eviction never touches in-flight page |
| C13 | Radix KV index | prefix reuse across requests/turns; chunked-hash-tree leaves, shared immutable blocks | `02-components.md` §C13 + `09-kv-engine-architecture.md` §8 | repeated-prefix benchmark shows TTFT reduction proportional to shared prefix length |
| C14 | KV cache engine | paged KV attention kernels read; layer slabs, block tables, geometry from config | `02-components.md` §C14 + `09-kv-engine-architecture.md` §7 | a KV page survives evict+reload bit-identical (I7) |
| C15 | KV codecs | how KV bytes are represented per tier; FP8 on gfx1201, FP16 on gfx1031, TurboQuant 4bit-nc behind a gate | `02-components.md` §C15 + `09-kv-engine-architecture.md` §6 | FP8 KV within tolerance of BF16 on deterministic prompts |
| C16 | Attention kernels | prefill + decode, GQA/MQA, paged; two-tile ping-pong, selective rescale, SGPR/VGPR split | `02-components.md` §C16 + `03-kernels.md` | decode kernel matches CPU reference within tolerance for GQA/MQA/MLA |
| C17 | MoE expert GEMM | grouped expert GEMM over the slot-resident set; per-token launches prohibited, contributions never dropped | `02-components.md` §C17 + `03-kernels.md` | grouped GEMM matches materialised reference within tolerance |
| C18 | Router | choosing the experts; exactly the model’s router, top-k exact, bias checked under both spellings | `02-components.md` §C18 | unbiased execution impossible to reach by config; routing-agreement metrics reported |
| C19 | CPU / direct-memory expert fallback | escape hatch when an expert will not arrive in time; transfer-vs-compute cost model | `02-components.md` §C19 + `04-memory-tiering.md` §1 | cold-expert requests complete within latency SLO on a 6 GiB profile |
| C20 | Speculative decoding | MTP first, one loader for DFlash/DFlash2/DSpark, DSpark confidence head wired into prefetcher | `02-components.md` §C20 + `05-speculation.md` | bit-identical to non-speculative under fixed seed |
| C21 | Profiler and telemetry | `--profiling` subsystem; output contract, instrumentation floor, per-component table | `02-components.md` §C21 + `06-profiling.md` | `--profiling` prints the reference table with the exact column layout and the floor line |
| C22 | Memory budget manager | detected hardware → policy numbers; profile selection across VRAM × RAM matrix | `02-components.md` §C22 + `01-architecture.md` §3 + `04-memory-tiering.md` §2 | a run on each VRAM class reports the profile it selected and the measurement that justified it |
| C23 | Session scheduler and server | multiple requests, prefix-homogeneity batching, OAI/Anthropic-compatible HTTP surface | `02-components.md` §C23 | shared-prefix multi-request workload beats FIFO batching on TTFT |
| C24 | Platform layer | Windows 11 and Linux differences, ROCm packaging, ReBAR, DirectStorage / io_uring | `02-components.md` §C24 + `07-build-platforms.md` | same binary and test suite pass on Windows and Linux for both targets |

---

## The docs, in one line each

| doc | what it settles | current state |
|---|---|---|
| `00-verified-facts.md` | what was measured on this machine, with commands; corrections to earlier project notes | the source of truth; §9.8 carries the W3 operating point |
| `01-architecture.md` | tier model, budget arithmetic, the transfer budget, invariants, per-token control flow | subsance present; structural gap — outline promises §1/§2/§9 but the file opens at §3. See note below. |
| `02-components.md` | C1–C24, each with interface, state machine, invariants, failure modes, acceptance tests | complete through the component list; acceptance tests named in `08`, not in `02` |
| `03-kernels.md` | gfx1201 WMMA path, gfx1031 SIMT path, tile geometry, validation matrix | complete |
| `04-memory-tiering.md` | ReBAR, VRAM/RAM profile tables, NVMe I/O shape, prefetch, eviction, KV residency, context-class ladder | §6 extended by `09`; recovery + session checkpoint still missing |
| `05-speculation.md` | MTP first, one loader for DFlash/DFlash2/DSpark, lossless verification, confidence coupling | complete at the design level |
| `06-profiling.md` | the `--profiling` output contract, how dev/idle/host time is measured, the instrumentation floor | complete; tail/SLO and output schema version still missing |
| `07-build-platforms.md` | ROCm 10.1, per-target flags, Windows/Linux specifics, kernel packs, CI, `kanjoos doctor` | complete |
| `08-roadmap.md` | phases P0–P7 with gates, whole-engine acceptance criteria, risk register, and the missing-items list | complete; see §7.2 for the explicit missing-items list |
| `09-kv-engine-architecture.md` | why KV residency is the product and weights are the optimisation, the context-class ladder, the component flowchart, the gap check against the tree that exists | the most complete doc; KV ladder decision and session checkpoint still missing |

---

## The tooling, in one line each

| file | what it is | runs | settles |
|---|---|---|---|
| `tools/isa_probe/isa_probe.hip` + `run_isa_probe.sh` | compile-only capability probe; decides what any kernel is allowed to use | `tools/isa_probe/run_isa_probe.sh` | the ISA matrix; exit 0 = measured, exit 3 = toolchain broken |
| `tools/kvroof/kv_roofline.py` + `kvroof.py` | derives the KV roofline, crossover and Class A/B ceilings from a model’s own `config.json` | `python tools/kvroof/kv_roofline.py models/qwen3-30b-a3b/config.json --check` | exit 0 = doc agrees, exit 5 = doc contradicts the config |
| `tools/check_docs.py` | audits the figures quoted in `00` §9 against the tools | `python tools/check_docs.py` | exit 0 = every documented figure matches the tool that produced it |
| `tools/route/route_locality.py` | P0-7; measures how much of a token’s expert selection survives to the next token from real router weights | `python tools/route/route_locality.py ...` | the miss cost in ms/token at the measured PCIe bandwidth |
| `tools/i7/i7_bit_identity.py` + `i7_hip.hip` | the I7 test; resident, movement-codec and tier-split paths must agree bitwise; three negative controls must fail | `python tools/i7/i7_bit_identity.py` | exit 0 = I7 holds, exit 10 = a path diverged |
| `tools/bench/run_bench.sh` | microbenchmark runner; tier A host correctness, tier B device compile census, tier C device run | `tools/bench/run_bench.sh` | exit 0 = tier A passed; tier C failures are non-zero |
| `tools/doctor/kanjoos_doctor.sh` | audit the machine before any measurement is believed; ROCm installs, packs, ReBAR, toolchain | `tools/doctor/kanjoos_doctor.sh` | exit 0 = audit clean, exit 1 = blocking problem, exit 2 = no ROCm found |
| `tools/kpack_report.md` | what each `.kpack` on this machine provides, and which the engine uses | n/a (reference) | blas as the measured reference; the rest deliberately not used |
| `models/README.md` + `models/*/config.json` | five real `config.json` files with sha256 provenance, including the target | n/a (reference) | no KV number in the repo rests on an assumed head configuration |

---

## Configuration

| file | what it is | notes |
|---|---|---|
| `kanjoos.toml` | shipped default configuration described by C22; every value is a starting point the autotuner overrides | C22 + `01` §3; `kanjoos doctor` prints which values were measured and which were taken from this file |

---

## The headline numbers, where they live

These are the numbers the plan is sized against. Each one is in a doc or a tool,
and each one is either measured or derived-from-measured, never assumed.

| number | where | status |
|---|---|---|
| ISA matrix (14 probes, 4 cell values: OK / REFUSED / CRASH / EMPTY) | `00` §1 + `tools/isa_probe/` | measured |
| VRAM bandwidth 589–598 GB/s; PCIe 27.9–28.0 GB/s streaming, 13.4–14.7 GB/s for a 2.47 MB expert copy | `00` §8.1 + `tools/bench/` | measured |
| Derived FP16 peak 39.3 TFLOP/s; machine balance 65.7–66.7 FLOP/byte | `00` §8.2 + `tools/kvroof/kvroof.py` | derived from measured |
| Expert arithmetic at 100% of peak = 7.68 µs; transfer = 181.3 µs; transfer-bound by ~23× | `00` §8.3 | measured + derived |
| Weight format is 0.5234 B/weight at W4 g128 (0.5 payload + 3/128 metadata) | `00` §8.10 + `tools/bench/gemm_w4.hip` | measured/settled |
| Routing locality weak: 59–81% of a token’s experts are new; 70% of bank buys ~95% hit | `00` §9.6 + `tools/route/` | measured, two independent sources |
| KV bytes/token by codec (FP16 98,304; FP8 49,152; INT4 g128 25,728) | `00` §9.2 + `tools/kvroof/` | derived from config |
| Crossover at 9.6 K (expert-only) / 16.1 K (with LM head) | `00` §9.3 + `tools/kvroof/` | derived from config |
| W3 g128 operating point: 11.79 GiB resident, 3.46 GiB KV budget, 37.8 K FP16 context | `00` §9.8 + `01` §4.2 + `tools/kvroof/` | derived; decision recorded |
| I7 tier movement is bit-identical; negative controls detected | `00` §9.7 + `tools/i7/` | tested, host + device |
| Profiler contract: per-component ops / %dev / dev us / idle us / host us + instrumentation floor | `06` + `02` §C21 | specified |

---

## What is NOT yet in any doc (explicit list)

See `08-roadmap.md` §7 and `MISSING-ITEMS.md` for the full list with ordering.
The short version:

1. Error model and failure taxonomy (retriable / drop / degrade / user-facing / escape-hatch / telemetry).
2. Security boundary for the HTTP surface and the cold store.
3. Recovery walk-back on dirty shutdown (journal replay, in-flight discard, partial-extent quarantine).
4. Multi-session correctness invariants (shared-block refcounts, evict-while-shared, cancel semantics).
5. Latency SLO and tail definition (p95/p99, per-step vs per-token, Class A/B/cold-start).
6. Testing contract for the driver/launch abstraction and the CPU-reference path.
7. Load/save/checkpoint/reload of a live session (format, what is captured, reload path, fingerprint mismatch).
8. Fused-kernel contract (guaranteed vs opportunistic, regression shape in the profiler, fallback).
9. Tokenizer and detokenization boundary (where it lives, unknown-token behaviour, embedding table loading/versioning).
10. Mixed-precision KV ladder decision (when it is built, promotion/demotion triggers, Class A vs Class B).

And these are intentionally not missing: training/fine-tuning, multi-GPU, default trunk/router quantisation, production BF16→ternary, NVMe-as-VRAM, a general tiering engine, WMMA tiling, and CPU fallback as a performance path. All recorded as non-goals or deferred in `01` §8, `09` §10.2, and `MISSING-ITEMS.md` §4.

---

*This page is a map, not a spec. The specs are the numbered docs. When a doc
and this page disagree, the doc wins and this page is stale.*
