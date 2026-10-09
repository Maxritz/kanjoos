# AGENTS.md — Kanjoos

Orientation and hard rules for any coding agent (or human) working in this
repository. Read this first, then the documents it points at. Sources of truth
are listed in [§8](#8-sources-of-truth-and-the-order-to-read-them).

---

## 1. What this project is

**Kanjoos** is an out-of-core **MoE (mixture-of-experts) GGUF inference engine**.
The product is not "load a huge model somehow". The product is:

> A bounded-working-set MoE inference engine in which the model can be much
> larger than VRAM, cold expert weights can stay on NVMe, useful experts are
> promoted through RAM into VRAM, routing drives demand and prefetch, and the
> runtime proves through telemetry and full-output correctness that the
> resulting system is correct and faster than naive streaming.

The governing reframe, from [docs/09-kv-engine-architecture.md](docs/09-kv-engine-architecture.md):

> **KV residency is the product feature. Weight residency is an optimisation.**

Two resources contend for VRAM, and they have different economics:

| resource | size | cacheable? |
|---|---|---|
| expert weights | ~14.13 GiB at W4 g128, **fixed** | yes — locality is exploitable |
| KV | `tokens × bytes_per_token`, **unbounded** | no — 100% miss every token, forever |

Reference model: **Qwen3-30B-A3B** (`models/qwen3-30b-a3b/config.json`) —
48 layers, 128 experts/layer, top-8, hidden 2048, moe_intermediate 768, 32 heads
/ 4 KV heads × head_dim 128, vocab 151936, bf16, `tie_word_embeddings false`.

**Target order — gfx1201 is the primary card.** gfx1201 on Windows 11 first →
gfx1201 Linux → gfx1031 Windows 11 + Linux. **gfx1031 is future work.** It stays
a *declared* first-class target — the tier B census still compiles it, no RDNA2
design may depend on hipBLASLt, and a gfx1031 refusal or skip is reported rather
than omitted — but it is never the default, never the first thing measured, and
**no current result is gated on it**. Anything that reads "gfx1031 first" is
stale as of 2026-10-08.

---

## 2. Repository map

```
CMakeLists.txt        top-level build (KNJ_ARCH selects the target arch)
kanjoos.toml          runtime configuration — starting values only, never truth
docs/                 the design evidence: 00…09, COMPONENT-REFERENCE, MISSING-ITEMS
ai-coder/             implementation worksheets c1…c24 — CONTRACTS, not suggestions
kernels/              device code headers + knj_kernels.hip
src/                  the runtime; src/device/ is the only DEVICE C++, and the
                      host side (loader, tokenizer, model, profiler, platform,
                      util, cli) is built into knj_runtime + kanjoos-run
tools/bench/          run_bench.sh (tier A/B/C gate) + device bench drivers
tools/ref_qwen3moe.py  the numpy oracle for the reference model's forward pass
tools/ref_qwen35.py    the numpy oracle + dequant cross-check for the qwen35 front end
tools/{ci,doctor,i7,isa_probe,kvroof,route}/   the machine-audit tools
tools/tok_crosscheck.py  tokenizer parity vs llama.cpp: the hand set, --corpus,
                       --fuzz, and --separator-sweep (the 14-codepoint whitespace
                       sweep). tools/tok_pre_rules.py re-derives the pre-tokeniser
                       dispatch table from llama.cpp's actual regexes and pins it
                       in tests/fixtures/tok/pre_rules.expected;
                       tools/smoke_tools.py smokes both tools
tools/c21/             profile_diff.py — the C21 regression gate (exit 1 on a
                       regression; a re-pin needs a stated reason)
tools/ggufmeta/        gguf_meta.py — read-only GGUF metadata/tensor dump; the
                       tool behind docs/10's measured draft-model layouts
models/                model config + provenance
records/              raw measured output, kept verbatim
tests/                host-side tests
cmake/                arch.cmake and friends
build/                build output (ignore)
```

The only *device* C++ in the repository today is `src/device/` (`backend.cpp`,
`backend.h`, `device_caps.h`) — component **C2**. The root CMake build also
produces a host-only `knj_runtime` + `kanjoos-run` from `src/loader`,
`src/tokenizer`, `src/model` (the `qwen3moe` forward pass and the `qwen35` front
end), `src/profiler`, `src/platform` and `src/util`. Everything else is design,
worksheets, and bench drivers. Do not assume a component exists because it is
documented; check `src/` and `kernels/`.

---

## 3. The machine, as measured

```
GPU      AMD Radeon RX 9070 XT, gfx1201, RDNA4, 32 CU, wave32, 2400 MHz
RAM/OS   Windows 11
```

### Toolchain trees (all verified present)

| tree | contents |
|---|---|
| `G:/ROCM10RT-gfx1201` | ROCm headers, `lib/llvm`, **device bitcode** (`lib/llvm/amdgcn/bitcode/*.bc`), **rocWMMA 2.2.1**, `.kpack/rocwmma_test_gfx1201.kpack` |
| `G:/ROCM10RT-gfx1031` | the gfx1031 counterpart |
| `/c/ROCm72` | a further `hip/hip_runtime.h` + `amdhip64` |

Env in use: `HIP_PATH`, `HIP_CLANG_PATH`, `HIP_LIB_PATH`, `ROCM_PATH`,
`HIP_PATH_72` → `G:\ROCM10RT-gfx1201`; `HIP_VISIBLE_DEVICES=0`,
`HIP_PLATFORM=amd`, `HIP_FORCE_DEV_KERNARG=1`, `HIP_VMEM_MANAGE_SUPPORT=1`,
`HIP_MEM_POOL_SUPPORT=1`, `AMD_LOG_LEVEL=0`.

`hipcc` on `PATH` is HIP **7.16.26323**, AMD clang 23.0.0git. The `clang` on
`PATH` is the *Swift* toolchain (6.3.3) and is **not** the ROCm clang — always
pass `G:/ROCM10RT-gfx1201/lib/llvm/bin/clang.exe` explicitly.

### The three build lines that work

Device compile census (this is the spelling that works; `run_bench.sh` tier B does
**not** use it yet and still refuses everything with `'hip/hip_runtime.h' file not
found`, because it passes `-nogpuinc`. Verified by hand, rc=0):

```sh
"G:/ROCM10RT-gfx1201/lib/llvm/bin/clang.exe" \
  --offload-arch=gfx1201 --rocm-path="G:/ROCM10RT-gfx1201" \
  -nogpulib --cuda-device-only -x hip -std=c++17 \
  -DKNJ_DEVICE_TIER=1 -Itools/bench -S tools/bench/<driver>.hip -o out.s
```

Tier C (device run) with hipcc:

```sh
hipcc -nogpulib -O2 --offload-arch=gfx1201 \
  -Xclang -target-feature -Xclang +wavefrontsize32 \
  -I tools/bench -I kernels \
  -L"G:/ROCM10RT-gfx1201/lib" -lamdhip64 \
  tools/bench/<driver>.hip -o <tmp>/<driver>.exe
```

rocWMMA drivers need **two extra flags, both mandatory**:

```sh
hipcc --rocm-path="G:/ROCM10RT-gfx1201/lib/llvm" -std=c++17 -O2 \
  --offload-arch=gfx1201 \
  -Xclang -target-feature -Xclang +wavefrontsize32 \
  -I tools/bench -I kernels \
  tools/bench/<driver>.hip -o <tmp>/<driver>.exe
```

* `--rocm-path` must be **`.../lib/llvm`**, not the ROCm root. Pointing at the
  root answers `cannot find ROCm device library`. Both spellings look like
  "the ROCm path"; only one works.
* `-std=c++17` is a **header requirement**, not a preference. Without it,
  rocWMMA fails with `no member named 'apply' in namespace 'std'`
  (`utility/apply.hpp`) plus a cascade of constexpr-init errors in
  `io_bearer_base.hpp` / `register_layout_traits_impl.hpp`.

`tools/bench/run_bench.sh` applies these flags automatically to any driver whose
source contains `#include <rocwmma/rocwmma.hpp>` (see `is_rocwmma_driver`), so
you do not add a driver to a list — you add a file.

---

## 4. The rules that have already cost time

These are not style preferences. Each one exists because the opposite shipped
here at least once.

1. **`-nogpulib` is the default build mode.** No device-side libm, and — the
   consequence that bites — **no `gridDim` in any kernel on gfx1201**:
   `__ockl_get_num_groups` comes from the runtime `-nogpulib` skips. Pass the
   grid extent as a kernel argument.
2. **`__expf` is the native quarter-rate instruction**, not a fast approximation
   of a correctly-rounded `expf`. There is no device `exp` to compare against.
3. **`uint32_t` indexing suffices.** 64-bit indexing costs registers and buys
   nothing at these sizes.
4. **`+wavefrontsize32` is inert on gfx1201** (the hardware is natively
   wave32, and clang warns `not a recognized feature (ignoring)`). Keep it for
   explicitness; do not expect it to change codegen.
5. **The arch must pass through the environment, not `-D`.** `hipcc` re-spawns
   clang through a command string that eats quotes, so
   `-DKNJ_BUILD_ARCH="gfx1201"` fails with `use of undeclared identifier`.
6. **Every tier-C driver carries the arch guard** (`tools/bench/knj_wmma_guard.h`).
   `KNJ_BUILD_ARCH` must equal `hipGetDeviceProperties().gcnArchName`:
   * unset → **exit 7** + `ARCH GUARD NOT CONFIGURED`
   * mismatch → **exit 6** + `SKIPPED -- not run, not measured, not a result`
   * match → continue
   ***An arch-mismatched binary runs and lies.*** This is a correctness
   requirement, not a diagnostic nicety.
7. **Check every output.** Full-output correctness against a host oracle is the
   default. A strided sample plus a passing exit code is consistent with
   nonsense — that has happened here. Report: max abs error, relative RMSE,
   **zero/unwritten outputs**, and mismatching indices.
8. **Never weaken an assertion, swallow an error, or add a suppression to make
   a check pass.** If a check must change, record why and verify the new
   behaviour.
9. **A tier that did not run is reported, never omitted.** "Not run" and
   "passed" are different results and must never be written the same way.
10. **Preserve exit status through any filter.** `cmd | head` reports `head`'s
    status. Use `set -o pipefail` or capture `${PIPESTATUS[0]}`.
11. **Both GPU targets must be accounted for in any device-code change.** If one
    cannot execute on this machine, mark it explicitly `compile-only` or
    `declined`; never let an unrun target read as a passing one.
12. **No `hipDeviceSynchronize()` inside the step loop.** Use streams, events,
    and `poll()`.
13. **Never turn every expert request into a tiny random disk read.** The
    residency decision is fine-grained; the I/O decision is **coarse-grained**
    (coalesced multi-expert extents, normally a layer slab).
14. **A missing expert must never silently disappear.** Every routed
    contribution is accounted for: resident → execute; arriving → wait at an
    explicit dependency point while useful work continues; else CPU fallback;
    else **explicitly degrade or refuse**. `expert unavailable → skip` changes
    model semantics and is prohibited.

---

## 5. Correctness doctrine

### Invariants I1–I7

| id | invariant |
|---|---|
| **I1** | No expert is lost. Every routed contribution is executed. |
| **I2** | No dequantisation on a cache-miss path. Bytes moved = bytes the GEMM consumes. |
| **I3** | The router is authoritative. Exact top-k, deterministic tie handling. |
| **I4** | No synchronous NVMe read on the critical path. |
| **I5** | KV identity includes model fingerprint, arch, dtype, layout, RoPE, tokenizer, and attention-implementation version. |
| **I6** | Speculation is lossless: bit-identical output under a fixed seed. |
| **I7** | Tier movement is invisible: bit-for-bit. |

### I7's canonical reduction order (normative, do not improvise)

```
ascending pages
→ ascending keys within a page
→ per-page partial
→ left-to-right merge
→ global fp32 max first
```

A codec that changes values is a **declared precision reduction**, never
conflated with a tier move. That distinction is the whole of I7.

### Validation tolerances (`docs/03-kernels.md`)

| path | tolerance |
|---|---|
| unpack | bit-exact |
| expert GEMM (fp32) | rtol `1e-4` |
| attention (fp32) | rtol `2e-3` |
| KV codec | format-specific |
| RadixKV reload | **bit-identical** |

### Capability chain

Compiling an instruction is **not** proof the target can execute it. Treat
these as separate gates, and report the exact state:

```
compile → emit → link → load → dispatch → execute → verify output
```

States: `OK`, `REFUSED`, `CRASH`, `EMPTY`, `LINK_FAIL`, `RUN_FAIL`.

---

## 6. Architecture rules

**The layering rule** ([docs/07-build-platforms.md](docs/07-build-platforms.md)):

> Every architecture difference lives in `src/device/`, `kernels/`,
> `src/platform/`, or `cmake/arch.cmake`. Nothing above those four contains
> `#ifdef __gfx*__` or `#ifdef _WIN32`.

`cmake/arch.cmake` sets `--offload-arch=gfx1201 -mno-wavefrontsize64
-D__gfx1201__` (and the gfx1031 equivalent) from `KNJ_ARCH`.
`HSA_OVERRIDE_GFX_VERSION` is deliberately **not** set.

Platform split (all behind C24):

| | Windows | Linux |
|---|---|---|
| I/O | DirectStorage preferred, IOCP/overlapped fallback | `io_uring` + `O_DIRECT` |
| pin | `VirtualLock` | `mlock` + `MAP_POPULATE` |
| hints | `FILE_FLAG_SEQUENTIAL_SCAN` | `RWF_HIPRI`, `posix_fadvise(WILLNEED)`, `madvise(MADV_WILLNEED\|MADV_HUGEPAGE)` |
| ReBAR | must be detected | `resource` in `/sys/class/drm/card*/device/` |

### gfx1201 matrix backend: rocWMMA

**Decision (measured, 2026-10-07):** rocWMMA 2.2.1 is the gfx1201 matrix
backend. Its `internal/wmma_impl.hpp` instantiates the gfx12 path for
`AMDGCN_ARCH_ID_GFX1200/GFX1201` and calls exactly the builtins this project
needs. The per-lane register interleave lives inside `load_matrix_sync` /
`store_matrix_sync`.

Consequently:

* **Do not hand-roll the `_gfx12` operand layout.**
* **Do not write inline `__builtin_amdgcn_wmma_*_gfx12` in new code.**
* Express model operations (expert GEMM, attention prefill, attention decode)
  as rocWMMA fragments. See [kernels/knj_rocwmma.h](kernels/knj_rocwmma.h).
* The only tile shape the gfx12 builtins support is **16×16×16, wave32**.

Two rocWMMA facts that are easy to get wrong, both measured:

* **`layout_t` is per-operand and both spellings are needed in one attention
  kernel.** For `S = Q · Kᵀ` where K is stored `[n][d]` (K×N column-major), B
  must be `col_major`; declaring it `row_major` reads K transposed and every
  value is wrong (measured 4096/4096 wrong vs 0/4096). For `P · V` where V is a
  true K×N row-major matrix, B is `row_major`.
* rocWMMA refuses gfx1031 at compile time with `static assertion failed:
  Unsupported architecture`. That is the correct behaviour — gfx1031 has no
  WMMA. The gfx1031 compute basis is `v_pk_add_f16` / `v_pk_fma_f16`, with
  `sdot4` lowering to native `v_dot4c_i32_i8`, and W4 as `v_dot8_i32_i4`
  consuming packed nibbles **directly** (no unpack step).

### Operating point (W3 g128)

| quantity | value |
|---|---|
| bytes/weight | 0.3984 |
| expert bank | 10.76 GiB |
| resident | 11.79 GiB |
| KV budget | 3.46 GiB |
| FP16 context | 37.8 K tokens (75.6 K FP8) |
| experts resident | 126 / 128 |

W4 g128 = 0.5234 B/w, 14.13 GiB bank, **zero** KV budget, 127/128 resident.
**W4 is the first implemented pack; W3 is a one-constant repack of the same
kernel and format** — but only after its quality/SNR gate passes. Do not
silently claim W3 quality, and do not turn a capacity decision into a
bandwidth claim.

Metadata is **3 bytes/group** (fp16 scale + uint8 zero-point) for expert packs
*and* KV group codecs, whatever the bit width.

---

## 7. Commands

```sh
# the machine audit — 24 documented figures vs the tools
python tools/check_docs.py                    # expect: exit 0

# the KV roofline check against the model config
python tools/kvroof/kv_roofline.py models/qwen3-30b-a3b/config.json --check

# the full bench gate: tier A (host), tier B (device compile census), tier C (device run)
bash tools/bench/run_bench.sh                 # expect: exit 0; pins baselines in tools/bench/baselines.txt

# bit-identity check for the I7 reduction order
python tools/i7/i7_bit_identity.py

# routing locality
python tools/route/route_locality.py

# the host suite, including the python tool gates
ctest --test-dir build/cmake-host --output-on-failure
#   c21_profiler, tok_pre_dispatch, tok_whitespace_run -- C++ unit gates
#   tools_smoke          the tokenizer tools' behaviour, asserted not exit-coded
#   tok_separator_sweep  every separator in the engine's Space class vs llama.cpp
#   tok_pre_rules        the pre-tokeniser table re-derived from llama.cpp's
#                        actual regexes (name map + pre-type regex table) and the
#                        oracle binary's own strings
# A python gate whose input is missing prints NOT RUN and exits 3; ctest shows it
# as SKIPPED with the reason, so an unrun gate never reads as a pass. Configure
# with -DKNJ_PYTHON=<python.exe> if the first interpreter cmake finds lacks the
# `gguf` package the fixture generator imports.

# the same tools by hand, when the full report is wanted
python tools/tok_crosscheck.py --model <file.gguf> --separator-sweep   # env: KNJ_TOK_MODEL
python tools/tok_pre_rules.py                       # env: KNJ_LLAMA_CPP, --reference
python tools/smoke_tools.py    # env: KNJ_TOK_MODEL, KNJ_ENGINE, KNJ_LLAMA_TOKENIZE
```

Tier C drivers must be run arch-guarded:

```sh
KNJ_BUILD_ARCH=gfx1201 /tmp/rocwmma_moe.exe
```

**Note:** this checkout is **not a git repository** (`git status` fails).
Do not plan work around branches, commits, or stashes.

---

## 8. Sources of truth and the order to read them

When documents disagree:

1. a measured runtime/test result;
2. [docs/00-verified-facts.md](docs/00-verified-facts.md);
3. the tool output that generated or validated the number;
4. [docs/09-kv-engine-architecture.md](docs/09-kv-engine-architecture.md) for KV;
5. [docs/02-components.md](docs/02-components.md) for interfaces;
6. [docs/01-architecture.md](docs/01-architecture.md);
7. the other design documents;
8. comments or old notes.

> **Never preserve an old statement merely because it is already written.**

Reading order: `00` → `01` → `02` → `04` → `09` → `03` → `07` → `06` → `08` →
`10` (the draft models — read it before `c20-speculation.md`, because it is the
measured input that worksheet assumes) → `CODING-LOG` → `MISSING-ITEMS` →
`COMPONENT-REFERENCE` → the applicable `ai-coder/c*.md` worksheets → the existing
tests, tools and build files.

### Known-stale documentation (verified wrong, 2026-10-07)

| location | the wrong claim | the fact |
|---|---|---|
| `docs/07-build-platforms.md` §3.2 | `G:\ROCM10RT-gfx1201` deleted; "no gfx1201 `.kpack` files on this machine" | the tree is complete, with device bitcode and `rocwmma_test_gfx1201.kpack` |
| `docs/00-verified-facts.md` §7.4a | the `_gfx12` operand layout is a blocking unknown | **closed** — rocWMMA owns the mapping |
| `ai-coder/c2-device.md` | the WMMA path is gated behind the layout question | that gate is **superseded**; the rocWMMA gate replaces it |
| `README.md` | "five real `config.json` files" | one model directory exists (`models/qwen3-30b-a3b/`) |

> Fixed and removed from this table on **2026-10-08**: `CMakeLists.txt` no longer
> says gfx1031 is primary — `KNJ_ARCH` defaults to `gfx1201`, and
> `tools/bench/run_bench.sh` now runs its census and its tier C list
> `gfx1201`-first. The row was deleted rather than struck through, because this
> table lists what is wrong *now*, and a fixed entry is not.

---

## 9. Documentation conventions

**`docs/CODING-LOG.md` is append-only.** New work appends a phase; it never
rewrites an old one. Corrections are appended as a new entry that names the
entry it corrects. The format (see `.agents/skills/dox/SKILL.md`):

```markdown
### Phase N — <name>  ·  <DONE | PARTIAL | BLOCKED | ABANDONED>

**Believed at the time**   constraint (cite the id) + baseline
**Decision**               chose / because / rejected / falsified by
**Changed**                `path` — what and why
**Verified**               `$ <command>` → exit `<n>` → what was observed
**Measurements**           | quantity | value | provenance |
**Still open**             item — unblocked by: what
```

Every number carries a provenance label:

| label | meaning |
|---|---|
| `MEASURED` | ran it on hardware; command and run recorded |
| `DERIVED` | computed from measured inputs; the inputs are named |
| `ASSUMPTION` | not verified; the task that would verify it is named |
| `REFUSED` | deliberately not computed; the reason is recorded |

> Never promote a `DERIVED` number to `MEASURED` because it looks exact.

Other skills: `.agents/skills/sherlock-it/SKILL.md`.

Every `kanjoos doctor` line in `docs/07-build-platforms.md` must be a measured
value.

---

## 10. Non-goals

Do not build these. They are deliberately out of scope
([docs/MISSING-ITEMS.md](docs/MISSING-ITEMS.md) §4 records the reasoning):

* training or fine-tuning;
* multi-GPU;
* default trunk/router quantisation;
* production BF16→ternary;
* NVMe-as-VRAM;
* a general-purpose tiering engine before the MoE case works;
* WMMA tiling as a first blocker;
* CPU fallback as a performance path (it is an escape hatch that prevents
  catastrophic stalls);
* "unlimited context" promises when the KV cannot fit the transfer budget.

---

## 11. How to work in this repo

```
UNDERSTAND
   ↓
LOCATE COMPONENT       (which c1…c24 worksheet owns this?)
   ↓
CHECK DEPENDENCIES
   ↓
CHECK EXISTING IMPLEMENTATION
   ↓
CHECK MEASUREMENTS
   ↓
WRITE TEST
   ↓
IMPLEMENT
   ↓
BUILD
   ↓
RUN
   ↓
VERIFY FULL OUTPUT
   ↓
PROFILE
   ↓
COMPARE AGAINST BASELINE
   ↓
UPDATE STATUS
```

If a measurement contradicts the design:

```
STOP → record the measurement → identify which assumption is invalid
     → update the design → rerun the affected tests → only then continue coding
```

**Do not bend the measurement to fit the architecture.**

The optimisation chain is always:

```
less unnecessary expert movement → better prediction → better residency
→ fewer PCIe misses → more overlap → less idle GPU time
→ lower token latency → larger usable models on smaller GPUs
```

Everything else is secondary. Do not spend the primary optimisation effort on
tiny GEMM improvements while expert movement dominates: one W4 g128 expert is
**2.47 MB**, its arithmetic lower bound at peak is **7.68 µs**, and its PCIe
transfer at measured small-transfer bandwidth is **181 µs**.
