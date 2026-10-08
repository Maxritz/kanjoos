# CLAUDE.md — Kanjoos

Concise agent companion. **Read [AGENTS.md](AGENTS.md) for the full contract** —
this file is the quick version: what to run, what is forbidden, what is already
measured, and where the truth lives.

---

## The project in one sentence

An out-of-core **MoE GGUF inference engine** that runs models larger than VRAM
by keeping hot experts in VRAM, warm experts in RAM, cold experts on NVMe — with
routing driving demand and prefetch, and telemetry plus full-output correctness
proving it works.

**KV residency is the product feature. Weight residency is an optimisation.**
([docs/09-kv-engine-architecture.md](docs/09-kv-engine-architecture.md))

Target order: **gfx1201 Windows 11 first** → gfx1201 Linux → gfx1031 Win11 +
Linux. Reference model: **Qwen3-30B-A3B**
(`models/qwen3-30b-a3b/config.json`).

---

## Ground truth about this machine

```
GPU   AMD Radeon RX 9070 XT, gfx1201, 32 CU, wave32, 2400 MHz     OS  Windows 11
ROCm  G:/ROCM10RT-gfx1201   (complete: device bitcode + rocWMMA 2.2.1)
      G:/ROCM10RT-gfx1031   /c/ROCm72
hipcc HIP 7.16.26323, AMD clang 23.0.0git (the `clang` on PATH is Swift's — not it)
```

**Not a git repository.** No branches, commits, or stashes. Don't plan around them.

The only compiled C++ in the repo is `src/device/` (component C2). Everything
else is design docs, worksheets, and bench drivers.

---

## Commands

```sh
python tools/check_docs.py                                     # exit 0, 24 figures
python tools/kvroof/kv_roofline.py models/qwen3-30b-a3b/config.json --check
bash tools/bench/run_bench.sh                                  # exit 0
KNJ_BUILD_ARCH=gfx1201 <driver>.exe                            # arch-guarded run
```

| tier | what it is |
|---|---|
| A | host correctness — CPU reference vs kernel mirror. Runs. |
| B | device compile census — instruction + resource, **not** throughput. Currently refuses everything: `-nogpuinc` hides the HIP headers. |
| C | device run — timed, with the CPU reference as oracle. Runs on gfx1201. |

---

## Build lines that work

Device compile (replaces the broken tier-B form):

```sh
"G:/ROCM10RT-gfx1201/lib/llvm/bin/clang.exe" \
  --offload-arch=gfx1201 --rocm-path="G:/ROCM10RT-gfx1201" \
  -nogpulib --cuda-device-only -x hip -std=c++17 \
  -DKNJ_DEVICE_TIER=1 -Itools/bench -S tools/bench/<driver>.hip -o out.s
```

Device run:

```sh
hipcc -nogpulib -O2 --offload-arch=gfx1201 \
  -Xclang -target-feature -Xclang +wavefrontsize32 \
  -I tools/bench -I kernels -L"G:/ROCM10RT-gfx1201/lib" -lamdhip64 \
  tools/bench/<driver>.hip -o <tmp>/<driver>.exe
```

**rocWMMA drivers additionally need `--rocm-path="G:/ROCM10RT-gfx1201/lib/llvm"`
and `-std=c++17` — both mandatory.** `run_bench.sh` adds them automatically to
any source containing `#include <rocwmma/rocwmma.hpp>`, so no list to update.

---

## Forbidden — each of these shipped here at least once

* **`gridDim` in any kernel on gfx1201.** `-nogpulib` → `__ockl_get_num_groups`
  is unresolved. Pass the grid extent as a kernel argument.
* **Hand-rolling the `_gfx12` WMMA operand layout**, or writing inline
  `__builtin_amdgcn_wmma_*_gfx12`. Use **rocWMMA**; it owns the mapping.
* **`hipDeviceSynchronize()` in the step loop.**
* **A synchronous NVMe read on the critical path.**
* **Skipping a routed expert because its slot is unavailable.** Degrade or
  refuse explicitly; never change model semantics silently.
* Per-expert `hipMalloc`; random 4 KiB expert payload reads.
* Weakening an assertion, swallowing an error, or adding a suppression to make
  a check pass.
* `-DKNJ_BUILD_ARCH=gfx1201` — pass the arch through the **environment**;
  hipcc eats the quotes and you get `use of undeclared identifier`.
* `HSA_OVERRIDE_GFX_VERSION`.
* `#ifdef __gfx*__` / `#ifdef _WIN32` above `src/device/`, `kernels/`,
  `src/platform/`, `cmake/arch.cmake`.

---

## Already measured — do not re-derive

| fact | value |
|---|---|
| rocWMMA on gfx1201 | 2.2.1; 16×16×16 f16→f32 matched the host oracle **bit-exactly**, 0/256 mismatches |
| rocWMMA tile shape | 16×16×16, wave32 only — no other variant exists |
| rocWMMA on gfx1031 | refused at compile: `static assertion failed: Unsupported architecture` |
| W4 g128 expert | 2.47 MB; 0.5234 B/weight; 14.13 GiB full bank |
| one expert, arithmetic at peak | **7.68 µs** |
| one expert, PCIe transfer at measured small-transfer bandwidth | **181 µs** |
| PCIe large stream / 2.47 MB transfer | 27.9–28.0 GB/s / 13.4–14.7 GB/s |
| VRAM stream / pinned RAM stream | 589–598 GB/s / ~13.6 GB/s |
| metadata | **3 bytes/group** (fp16 scale + u8 zero-point), any bit width |
| KV read overtakes expert weight traffic at | ~9.6 K tokens FP16 (16.1 K with LM head) |
| new experts per token | 59.1% / 80.5%; top-8 survives to next token 41% |
| W4 g128 KV budget | **zero** — W4 is a capacity decision, not a bandwidth one |
| rocWMMA expert GEMM (256 pairs/16 experts/K256/N512) | worst row-rel err `9.018e-07`, 131072/131072 checked, 0 unwritten |
| rocWMMA attention prefill / decode | `3.950e-04` / `2.869e-04` worst row-rel err, 0 unwritten, 0 clobbered |

Raw evidence: [records/2026-10-07_rocwmma_model_ops.txt](records/2026-10-07_rocwmma_model_ops.txt),
[records/2026-10-07_wmma_9070xt_run.txt](records/2026-10-07_wmma_9070xt_run.txt).

The unexplained **2× MAC factor** in the raw `_gfx12` builtin is a recorded
curiosity, not an open blocker — rocWMMA is the path. The old hand probes
(`wmma_layout.hip`, `wmma_tile_probe.hip`, `wmma_run.hip`) still run in tier C
and are kept as the evidence that the published mapping is wrong.

---

## Invariants — never violate

| | |
|---|---|
| **I1** | No expert is lost. |
| **I2** | No dequantisation on a miss path. Bytes moved = bytes consumed. |
| **I3** | The router is authoritative — exact top-k, deterministic ties. |
| **I4** | No synchronous NVMe read on the critical path. |
| **I5** | KV identity includes model fingerprint, arch, dtype, layout, RoPE, tokenizer, attention-impl version. |
| **I6** | Speculation is lossless — bit-identical under a fixed seed. |
| **I7** | Tier movement is invisible — bit-for-bit. |

I7 canonical reduction order: **ascending pages → ascending keys within page →
per-page partial → left-to-right merge → global fp32 max first.** A codec that
changes values is a **declared precision reduction**, not a tier move.

Tolerances: unpack bit-exact · expert GEMM fp32 `1e-4` · attention fp32 `2e-3` ·
RadixKV reload bit-identical.

---

## Correctness doctrine

* Every GPU kernel has a **host oracle**. Full-output comparison, always.
* `exit 0` is not proof. One plausible sampled value is not proof.
* Report max abs error, relative RMSE, **zero/unwritten outputs**, mismatching
  indices.
* **An arch-mismatched binary runs and lies.** Every tier-C driver checks
  `KNJ_BUILD_ARCH` against `gcnArchName`: unset → exit 7, mismatch → exit 6
  (`SKIPPED -- not run, not measured, not a result`), match → run.
* Capability chain: `compile → emit → link → load → dispatch → execute → verify`.
  States: `OK`, `REFUSED`, `CRASH`, `EMPTY`, `LINK_FAIL`, `RUN_FAIL`.
* **A tier that did not run is reported, never omitted.**
* If a measurement contradicts the design: **STOP**, record it, find the invalid
  assumption, update the design, rerun — then continue coding. Do not bend the
  measurement to fit the architecture.

---

## Documentation

`docs/CODING-LOG.md` is **append-only**. Format per `.agents/skills/dox/SKILL.md`:
`Believed at the time` / `Decision (chose · because · rejected · falsified by)` /
`Changed` / `Verified (command → exit code → observed)` / `Measurements` table /
`Still open`. Label every number `MEASURED` / `DERIVED` / `ASSUMPTION` / `REFUSED`.

Update status in `ai-coder/` worksheets when a component's state changes;
worksheets are **contracts**, not suggestions.

---

## Where the truth lives

When docs disagree: measured result > `docs/00-verified-facts.md` > the tool
that produced the number > `docs/09` (KV) > `docs/02` (interfaces) >
`docs/01` > the rest > old comments.

**Never preserve an old statement merely because it is already written.**

### Known-stale (verified wrong)

* `docs/07-build-platforms.md` §3.2 — claims the ROCm tree and the `.kpack`
  files are gone. Both exist.
* `docs/00-verified-facts.md` §7.4a — the `_gfx12` layout "unknown" is closed.
* `ai-coder/c2-device.md` — its WMMA gate is superseded by the rocWMMA decision.
* `README.md` — claims five model configs; one exists.
* `CMakeLists.txt` header — calls gfx1031 primary while the target order is
  gfx1201 first. **Pending an explicit decision.**

---

## Non-goals

Training/fine-tuning · multi-GPU · default trunk/router quantisation ·
production BF16→ternary · NVMe-as-VRAM · a generic tiering engine before the MoE
case works · WMMA tiling as a first blocker · CPU fallback as a performance path
· "unlimited context" claims the KV budget cannot pay for.
