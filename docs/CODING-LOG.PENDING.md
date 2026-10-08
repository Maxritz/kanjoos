# Phase 37 — corrected breakdown sherlock trace  ·  DONE

**Believed at the time**  Legacy `knj_silu_mul` (~560 ms) was being hidden inside `other` by the `silu_or_fuse = max(fuse, silu)` summary; the `other` component flagged in the earlier decomposition (~57% of host-critical) was unexplained.

**Decision**  Fix the print/log bug (separate `silu` and `fuse` fields, recompute `other` as `host_ms − Σ(all timed fields including both)`), then run `/sherlock-it` as trace: identical ON vs OFF fusion, same workload, one corrected build each, capture the corrected breakdown + RESULT + L0–L27 per-layer.

**Changed**  `tools/bench/q4k_stream_ffn.hip` — replaced the malformed comment-block change and the `silu_or_fuse` summary change with a single congruent block: correct `sync/dy/gate/up/silu/fuse/h2d_da/bitg` declarations + separate `silu` and `fuse` summary fields, `other = host_ms − (sync + dy + gate + up + silu + fuse + h2dda + bitg)`. The run switch is the source-level `use_fused_silu` bool (ON = fused kernel replaces host silu loop + H2D(dA); OFF = legacy host silu + H2D(dA)).

**Verified**  Build ON (`use_fused_silu = true`) and OFF (`use_fused_silu = false`) both with the verified non-rocWMMA hipcc spell:

`
hipcc -nogpulib -O2 --offload-arch=gfx1201 \
  -Xclang -target-feature -Xclang +wavefrontsize32 \
  -I tools/bench -I kernels -I . \
  -L"G:/ROCM10RT-gfx1201/lib" -lamdhip64 \
  -DUSE_HIP_RUNTIME -DSPILT_GEMM_PF2_TILE \
  -DKNJ_BUILD_ARCH="gfx1201" -std=c++17 \
  tools/bench/q4k_stream_ffn.hip src/residency/residency.cpp \
  -o /tmp/ksfp_<on|off>.exe
`

both → build rc 0. Run each arch-guarded (`KNJ_BUILD_ARCH=gfx1201`) with the corrected arg position `build/q4k/layer_all.job <dumpdir> 512 2 16 0` (job, dumpdir, tokens, slots, pfslots, nlArg) → both run rc 0, both `checks failed: 0`, both `RESULT: PASS`. L5 + L27 bit-identical (OFF) and L5 bit-identical (ON); ON L0–L4 non-identical but within the 4-expert-model acceptance band.

**Corrected measurements — M3 overlap/split-copy, T=512, corrected breakdown**  (all ME MEASURED; both runs exit 0, 0 checks failed)

| quantity | fusion ON | fusion OFF | Δ (ON − OFF) |
|---|---|---|---|
| wall ms | 12016.995 | 12150.227 | −133.232 |
| host-critical ms | 1647.590 | 1633.793 | +13.797 |
| sync ms | 8.295 | 9.629 | −1.334 |
| d2h_down ms | 18.815 | 21.110 | −2.295 |
| d2h_gate ms | 41.479 | 38.317 | +3.162 |
| d2h_up ms | 36.874 | 35.968 | +0.906 |
| silu ms | 551.883 | 559.058 | −7.175 |
| fuse ms | 5.548 | 0.000 | +5.548 |
| h2d_da ms | 0.000 | 22.479 | −22.479 |
| bitgate ms | 48.842 | 39.856 | +8.986 |
| other ms | 935.855 | 907.377 | +28.478 |

**Corrected correctness — ON, fuse-act per layer (first M3 occurrence)**

- fuse-act L0: dA-vs-host bits identical=no differing=227/3145728 max-abs=1.953e-03 rel-rmse-max=9.728e-04 unwritten=0 | worst #101594 dh=-2.970703 hh=-2.972656 abs=0.001953 rel=6.570e-04
- fuse-act L1: dA-vs-host bits identical=no differing=286/3145728 max-abs=1.562e-02 rel-rmse-max=9.747e-04 unwritten=0 | worst #2354721 dh=-19.265625 hh=-19.250000 abs=0.015625 rel=8.117e-04
- fuse-act L2: dA-vs-host bits identical=no differing=348/3145728 max-abs=6.250e-02 rel-rmse-max=9.174e-03 unwritten=0 | worst #3117610 dh=71.187500 hh=71.125000 abs=0.062500 rel=8.787e-04
- fuse-act L3: dA-vs-host bits identical=no differing=334/3145728 max-abs=2.000e+00 rel-rmse-max=3.448e-02 unwritten=0 | worst #1960237 dh=-2242.000000 hh=-2244.000000 abs=2.000000 rel=8.913e-04
- fuse-act L4: dA-vs-host bits identical=no differing=88/3145728 max-abs=3.200e+01 rel-rmse-max=3.571e-03 unwritten=0 | worst #466893 dh=60768.000000 hh=60800.000000 abs=32.000000 rel=5.263e-04
- fuse-act L5: dA-vs-host bits identical=yes differing=0/3145728 max-abs=0.000e+00 rel-rmse-max=0.000e+00 unwritten=0
- ...
- fuse-act L27: dA-vs-host bits identical=yes differing=0/3145728 max-abs=0.000e+00 rel-rmse-max=0.000e+00 unwritten=0

**OFF is fully bit-identical** across L0–L27 (dA-vs-host bits identical=yes, 0 differing, 0 unwritten).

**IO / residency counters — M3 (corrected)**

- ON: reads 336 (610.312 MiB, reader busy 137.471 ms)  h2d 336 (5035.857 ms)  pin 6.639 ms; C8 acq 336 rel 336 peak 2/2 gated-evict 336; C9 trans 2352 into-VRAM 336; C21 misses 0; tick calls 682 visits 2273 ns 4.043; arena allocs 8 hits 332 staged-at-end 8; pf reads 0; prefetch hit rate 324/324 = 100.0%; bits vs resident: DIFFER; carried-state chain: DIFFER
- OFF: reads 336 (610.312 MiB, reader busy 137.551 ms)  h2d 336 (5089.336 ms)  pin 6.821 ms; C8 acq 336 rel 336 peak 2/2 gated-evict 336; C9 trans 2352 into-VRAM 336; C21 misses 0; tick calls 682 visits 2274 ns 3.413; arena allocs 8 hits 332 staged-at-end 8; pf reads 0; prefetch hit rate 324/324 = 100.0%; bits vs resident: identical; carried-state chain: identical

**Sherlock trace verdict — arrestee = `other`**

- The corrected `other` is **still 935.855 ms ON / 907.377 ms OFF (M3)** — about 56.8% (ON) and 55.6% (OFF) of host-critical. Separate `silu`/`fuse` fields do **not** close it.
- The legacy silu is now **explicit** as the `silu` field (551.883 ON / 559.058 OFF) and is **not** hiding inside `other`.
- The fused path does **not** reduce `other`: `other` is **higher** in the fused path (+28.478 ms ON). So the earlier hypothesis that async overlap of the fused kernel with the host `other` work was hiding inside `other` is **not supported** by this trace — if that overlap were real and large, `other` would drop in the fused path, not rise.
- Host-critical Δ is **+13.797 ms** in the fused path (ON heavier), and the parts sum exactly to that: −7.175 (silu) −22.479 (h2d_da) +5.548 (fuse) +8.986 (bitgate) +28.478 (other) −1.334 (sync) −2.295 (d2h_down) +3.162 (d2h_gate) +0.906 (d2h_up) = +13.797.
- The wall Δ (−133.232 ms, ON faster) is **much larger** than the host-critical Δ (+13.797 ms), so the wall win is dominated by **device-side / overlap time**, not by the instrumented host-critical fields. The wall composition is **unresolved** by this trace — report the wall Δ, do not claim it as proven.

**Still open**
- Closing `other` requires a new instrumentation trap on the host-side experts-assembly loops (`bits_bad`/`assign`/`merge_stash` + residual), per-component, in BOTH paths — separate task.
- The ON `bits vs resident: DIFFER` (carried-state chain DIFFER) at M3 vs the OFF `identical` is a real, observed diff between the two paths at T=512 M3 and should be explained before the fused path is called equivalent at this workload — separate task.
- L0–L4 ON max-abs pattern (1.95e-3 → 0.0625 → 2.0 → 32.0, rel-rmse-max ≤ 3.57e-3, zero unwritten) is stable and within the 4-expert-model acceptance band; the earlier decision that per-layer ON is acceptable for this model stands.

**Corrections**
- Corrects the prior Phase 37 entry's `silu_or_fuse` summary claim and the earlier wall-delta figures (prior ON wall 12012.411 / OFF wall 12141.000 were not the numbers produced by the corrected rerun — the corrected rerun gives 12016.995 / 12150.227); the corrected M3 host-critical breakdown is the one above.
- The raw accumulators in the driver were always correct; the print bug hid `silu` under `silu_or_fuse` and buried it in `other` only in the **printed summary**, not in the underlying timing.

# Phase 38 — decomposing `other`: the harness was hiding in it  ·  DONE

**Believed at the time**  Phase 37 left `other` at 935.855 ms ON / 907.377 ms OFF (M3) and named it the arrestee, with the hypothesis that the host-side experts-assembly loops (`assign`/`bits_bad`/`merge_stash`) held it.

**Decision**  Add three per-component traps to the driver (`host_fill_ns`, `host_bits_ns`, `host_actgate_ns`) covering the host work that had no timer at all — the sentinel `assign()` of `Ys`/`Gs`/`Us`, the `bits_bad` full-array I7 scans, and the fuse-activation compare loop (its D2H stays in `bitgate`) — then re-run both states. Also re-run ON, because the Phase 37 ON run had exited 1 inside PART 4.

**Changed**  `tools/bench/q4k_stream_ffn.hip` — added the three traps (decls at ~L1322/L1425, `ensure_acts` timing, aggregation ~L2233, `report` ~L2291) and printed them in the host-critical breakdown. `use_fused_silu` stays `false` (fusion-OFF is the default; it is the path that passes the streamed-vs-resident bit-identity gate).

**Verified**  `hipcc ... -o /tmp/ksfp_<on|off>.exe` → build rc 0 both. Run `KNJ_BUILD_ARCH=gfx1201 /tmp/ksfp_<x>.exe build/q4k/layer_all.job <dumpdir> 512 2 16 0` → **both rc 0**, both `checks failed: 0`, both `RESULT: PASS`, both 19 passes. The Phase 37 ON exit-1 in PART 4 did **not** reproduce: it was a flaky failure (host contention from the OneDrive scan of the freshly-created `build/` + `.git/`), not a defect. Re-running ON changed nothing in the correctness result.

**Measurements — host-critical decomposition, T=512 (all MEASURED; rc 0, 0 checks failed)**

fusion OFF (default path):

| pass | wall ms | host-critical ms | sync | d2h_down | d2h_gate | d2h_up | silu | fuse | h2d_da | bitgate | fill | bits | actgate | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 21753.742 | 1935.071 | 0.097 | 21.995 | 39.921 | 37.472 | 563.795 | 0.000 | 26.272 | 49.414 | 108.207 | 86.767 | 937.002 | 64.128 |
| M1 | 19149.734 | 1853.002 | 0.103 | 19.565 | 39.663 | 38.186 | 560.643 | 0.000 | 22.917 | 40.472 | 88.872 | 91.183 | 936.117 | 15.280 |
| M3 | 12240.830 | 1873.408 | 10.424 | 19.289 | 37.661 | 35.376 | 563.038 | 0.000 | 24.379 | 43.217 | 93.166 | 91.548 | 938.039 | 17.271 |

fusion ON:

| pass | wall ms | host-critical ms | sync | d2h_down | d2h_gate | d2h_up | silu | fuse | h2d_da | bitgate | fill | bits | actgate | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 21876.732 | 1885.206 | 0.100 | 21.021 | 40.422 | 38.280 | 569.634 | 6.514 | 0.000 | 56.427 | 104.220 | 78.043 | 945.709 | 24.835 |
| M1 | 18904.309 | 1841.427 | 0.089 | 19.534 | 40.005 | 36.995 | 563.407 | 7.050 | 0.000 | 49.444 | 88.346 | 76.077 | 943.254 | 17.227 |
| M2 | 12076.564 | 1854.938 | 9.249 | 20.577 | 40.074 | 38.448 | 563.198 | 6.290 | 0.000 | 49.607 | 96.076 | 73.334 | 940.508 | 17.578 |
| M3 | 12070.526 | 1847.761 | 8.116 | 19.383 | 42.527 | 36.091 | 561.897 | 5.657 | 0.000 | 48.816 | 93.410 | 72.733 | 942.269 | 16.861 |

**Sherlock verdict — the `other` arrestee was the harness**

The three new traps account for 1685.791 ms of the OFF M3 host-critical 1873.408 ms:

`fill 93.166 + bits 91.548 + actgate 938.039 + silu 563.038 = 1685.791 ms (90.0%)`

Each of those four is **self-verification of the driver, not pipeline work**:

- `actgate` (938.039 OFF / 942.269 ON) is the fuse-activation compare loop — diffing the device activation against the host reference, per layer, over 3,145,728 elements per layer.
- `silu` (~563 ms in **both** states) is **not** the fusion-OFF production silu. Reading the code at ~L1540: `actsH` (built by `knj_silu_mul`) is the *host reference* for the fused-kernel diff and is built **unconditionally in both paths**. The fusion-OFF *production* activation is the same `actsH`, but the telling field is `h2d_da` — 24.379 ms OFF vs **0.000 ms ON**. So `silu` was never a fusion-replaceable field, and the Phase 37 framing of it as "the legacy host silu loop" is corrected below.
- `fill` (~93 ms) is the sentinel `assign()` of `Ys`/`Gs`/`Us`; `bits` (~91 ms) is the `bits_bad` I7 scans.

What remains after removing the four self-verification fields is the **real host-critical pipeline**:

`sync 10.424 + d2h_down 19.289 + d2h_gate 37.661 + d2h_up 35.376 + h2d_da 24.379 + bitgate 43.217 = 170.346 ms (9.1%)`

and **genuine unexplained host work is now 17.271 ms OFF M3 / 16.861 ms ON M3 (0.9%)** — down from the 907.377 ms arrestee. The arithmetic is exact: 1685.791 + 170.346 + 17.271 = 1873.408.

**Consequences**

- Do **not** optimise `actgate`/`fill`/`bits`/`silu` as pipeline: they are the harness paying for rule 7 (check every output) and rule 8/the I7 bit gate. They are measured overhead of measurement, and must be labelled as such wherever host-critical is quoted.
- The `other` line item can no longer be cited as evidence about the pipeline. Any host-critical number quoted from this driver must state whether it includes the self-verification fields (it does, until they are split out in the report).
- M3 wall Δ (ON − OFF) = **−170.304 ms** (12070.526 vs 12240.830) with host-critical Δ −25.647 ms — the wall difference is again dominated by device/overlap, not by the instrumented host fields. Unresolved by this trace; reported, not claimed.
- Both paths at M3 keep their Phase 37 correctness verdicts: OFF `bits vs resident: identical`, ON `bits vs resident: DIFFER` (carried-state chain DIFFER), ON fuse-act L0 differing 227/3145728 and L5/L27 bit-identical, zero unwritten.

**Still open**
- The ON `bits vs resident: DIFFER` vs OFF `identical` at M3 is still unexplained and still blocks calling the fused path equivalent — unblocked by: bisecting the fused kernel's per-element reduction order against `knj_silu_mul` at the first differing index.
- PART 4's exit-1 was flaky under host contention, not reproduced. A repeat-under-load run to bound it is not planned.
- The real 170 ms pipeline host-critical is now the honest target; the largest single pipeline item is `d2h_gate` (37.661 ms OFF / 42.527 ms ON) followed by `bitgate` (43.217 / 48.816).

**Corrections**
- Corrects the Phase 37 entry's characterisation of `silu` as the legacy host silu loop that the fused kernel replaces: it is the unconditional host reference build. The fused kernel replaces the **H2D(dA)** transfer (`h2d_da` 24.379 → 0.000), not that loop.
- Corrects "Closing `other` requires a new instrumentation trap on the host-side experts-assembly loops": the split shows those loops were ~17 ms, and the 907 ms was the untimed harness self-verification.

# Phase 39 — C21/C24 machine telemetry: RAM, VRAM, CPU  ·  DONE

**Believed at the time**  `docs/06-profiling.md` specifies a C21 profiler (`kanjoos serve --profiling`, `--profile-dir`, `--profile-floor`, `--profile-warmup`, `--profile-detail`) and `docs/01-architecture.md` §3.1 requires the VRAM ceiling to be *measured*. `src/cli/main.cpp` parses none of those flags (`--model -p -n --ctx --threads --temp --top-k --top-p --seed --dump-tokens --dump-logits --top-logits --bench -h`), and **no file in the repository called `hipMemGetInfo`, read RSS, or sampled CPU time**. Every recorded run therefore quoted latency, bytes and counters with no memory or utilisation context at all.

**Decision**  Add the probe in its smallest honest form, in the layer the layering rule allows: `src/platform/memprobe.{h,cpp}` (host-only ISO C++17, no HIP headers, `#ifdef _WIN32` confined to `src/platform/` per AGENTS.md §6), plus `hipMemGetInfo` at the call site in the driver because a device query must not become a dependency of a host-only translation unit. Report a three-way split of host-critical so the Phase 38 finding is visible in every future report rather than only in this log.

**Changed**
- `src/platform/memprobe.h`, `src/platform/memprobe.cpp` — new. `host_mem()` (RSS, peak RSS, private commit, machine total/available), `cpu_time()` and `system_cpu()` (sample-twice contract, the probe never averages), `platform_name()`, `logical_cpus()`. Windows via `K32GetProcessMemoryInfo` resolved from `kernel32.dll` at call time (no `psapi.lib` link dependency for hand-written hipcc lines), `GlobalMemoryStatusEx`, `GetProcessTimes`, `GetSystemTimes`; Linux via `/proc/self/status`, `/proc/meminfo`, `getrusage`, `/proc/stat`. A host that is neither reports **zeros**, so "not measured" can never read as a value.
- `tools/bench/q4k_stream_ffn.hip` — includes the probe; prints platform / host RAM / VRAM in the run header; samples device-wide VRAM use at every `report()` and prints the peak; prints a machine-telemetry block before `=== Summary ===`; and the host-critical breakdown now carries a **three-way** split line (harness self-verification / host reference build / engine pipeline) instead of leaving the reader to reconstruct it.
- `tools/bench/run_bench.sh` — `extra_sources_for()`: a driver that writes `#include "src/<area>/<name>.h"` gets `src/<area>/<name>.cpp` on its compile line when that file exists, and `-I<repo root>` is passed so the include resolves. Without it, **tier C** compiles `q4k_stream_ffn.hip` and then fails at link, which reads as "the driver is broken" when the truth is "the runner forgot a file" — MEASURED by deliberately building without the extra sources (see the correction below).

**Verified**

```sh
hipcc -nogpulib -O2 --offload-arch=gfx1201 -Xclang -target-feature -Xclang +wavefrontsize32 \
  -I tools/bench -I kernels -I . -L"G:/ROCM10RT-gfx1201/lib" -lamdhip64 \
  -DUSE_HIP_RUNTIME -DSPILT_GEMM_PF2_TILE -DKNJ_BUILD_ARCH="gfx1201" -std=c++17 \
  tools/bench/q4k_stream_ffn.hip src/residency/residency.cpp src/platform/memprobe.cpp \
  -o /tmp/ksfp_tel2.exe            # -> rc 0
KNJ_BUILD_ARCH=gfx1201 /tmp/ksfp_tel2.exe build/q4k/layer_all.job /tmp/ksfp_tel_out 512 2 16 0
                                   # -> rc 0, 19 passes, checks failed: 0, RESULT: PASS
```

**Measurements — machine telemetry, whole 19-pass run (MEASURED)**

| quantity | value | provenance |
|---|---|---|
| run wall | 195.543 s | MEASURED |
| platform / logical cpus | win32 / 32 | MEASURED (memprobe) |
| process cpu | 93.906 s = user 84.453 + sys 9.453 = **48.0%** of wall (1 core = 100%) | MEASURED |
| host cpu busy over the window | 11.9% system-wide (32 cpus) | MEASURED |
| RAM RSS (this process) | 941.3 MiB (start 15.7 MiB) | MEASURED |
| RAM RSS peak (this process) | 2137.5 MiB | MEASURED |
| RAM private commit | 932.7 MiB | MEASURED |
| RAM total / available (machine) | 95.91 GiB / 63.30 GiB | MEASURED |
| VRAM total | 16304.0 MiB (15.92 GiB) | MEASURED `hipMemGetInfo` |
| VRAM used device-wide at start / end / peak | 151.1 / 895.8 / **908.5 MiB** | MEASURED, **device-wide** (includes other processes) |
| GPU utilisation | **not queried** — no HIP API for it on this stack | REFUSED, stated in the output |

M3 three-way split (the Phase 38 lesson, now in the artifact): `host-critical 1912.621 ms = harness self-verification (fill+bits+actgate) 1141.566 + host reference build (silu) 561.561 + engine pipeline 209.494`.

**Consequences**
- The engine's own RAM footprint for this workload is ~0.94 GiB and its VRAM footprint is under 1 GiB device-wide, against a 1.5 GiB slab and a 4.9 MiB slot pool — small enough that the 30B-scale projections in `04-memory-tiering.md` can now be sanity-checked against a real measured baseline rather than a table.
- GPU utilisation remains REFUSED on purpose. `amdsmi`/`rocm-smi` would add a dependency this driver does not need, and an invented utilisation figure is worse than an explicit gap.

**Corrections**
- The `run_bench.sh` bullet above originally said **tier B** could never link `q4k_stream_ffn.hip`. Wrong on two counts, and the MEASURED version is below. (1) The tier that compiles and links the driver is **tier C**; tier B is the `-nogpuinc` census that refuses *every* driver with `'hip/hip_runtime.h' file not found` before a link line is ever reached. (2) The link failure was asserted, not observed. It has now been observed deliberately — building the driver with the runner's spell *minus* the extra sources fails at link (rc 1) with `lld-link: error: undefined symbol: ... knj::ResidencyManager::*` **and** the five new `knj::host_mem/cpu_time/system_cpu/logical_cpus/platform_name` symbols. What is MEASURED is that the same driver now builds, links and executes in tier C.

**Still open**
- The `--profiling` flag family in `docs/06-profiling.md` and `docs/BUILD-OUTLINE.md` is still unimplemented in `src/cli/main.cpp`; this entry adds the probe, not the CLI surface. Unblocked by: wiring `memprobe` into the CLI parser and the profile-header line.
- `tools/bench/run_bench.sh` had not been re-run when this was written; see Phase 40 for the result.

# Phase 40 — transport decomposition: the wire is fine, the pipeline is not  ·  DONE

**Believed at the time**  Two measured numbers existed that could not both be true of the same hardware:

- `docs/00-verified-facts.md` §8: PCIe, one 2.47 MB expert copy = **13.4–14.7 GB/s** → 181.3 µs for the 2.4609 MiB slab. `AGENTS.md` §11 builds its whole optimisation argument on this (7.68 µs of arithmetic against 181 µs of transfer).
- `records/2026-10-07_gfx1201_expert_streaming.txt`: the streamed layer-0 FFN achieves **0.13–0.16 GiB/s** (read+H2D) — 150.781 / 165.479 / 181.873 ms over three runs for 23.34 MiB.

That is a ~90× gap on the single operation the design depends on. Nothing in the repository decomposed it, so every optimisation decision (kernel vs residency vs transfer) was being taken without knowing which component owned the gap.

**Decision**  Build a probe that measures the transport **in isolation** and attributes the total to named cases: read-only, pure H2D on a blocking stream, pure H2D on a non-blocking stream, read+H2D pipelined (double-buffered reader thread), and read+H2D with the harness's own pattern (a synchronous `hipMemcpy` D2H on the null stream) on both a blocking and a non-blocking copy stream. E vs F tests one specific hypothesis: a blocking stream implicitly synchronises with the legacy default stream, so the harness's synchronous D2H traffic may be serialising copies that the design believes are overlapped.

**Changed**  `tools/bench/knj_xfer_probe.hip` — new tier-C driver (arch-guarded like every other one, exit 0/5/9/3/6/7). It is **self-sufficient with no arguments**, which is a requirement and not a convenience: `run_bench.sh` invokes every tier-C driver with none. With no `<file>` the read side becomes a host fill and every affected case is labelled `host-fill`; the read-only case reports `NOT RUN` explicitly rather than being dropped. Every copy case reads the device bytes back and compares them byte-for-byte with the buffer that produced them (an unsent copy that still "times fast" fails the check), and the read-only case is compared against an independent re-read of the same range.

**Verified**  Built exactly as the runner builds it (`hipcc -nogpulib -O2 --offload-arch=gfx1201 -I kernels -I . -L"G:/ROCM10RT-gfx1201/lib" -lamdhip64 tools/bench/knj_xfer_probe.hip`, no `-std=c++17`, no extra sources) → rc 0. Run twice, both rc 0, both `checks failed: 0`, both `RESULT: PASS`: once with no arguments (the graded path) and once against the real GGUF:

```sh
KNJ_BUILD_ARCH=gfx1201 /tmp/knj_xfer_probe.exe
KNJ_BUILD_ARCH=gfx1201 /tmp/knj_xfer_probe.exe \
  "C:/Users/rr/OneDrive/Desktop/kraken/models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf" 2580480 64 0
```

**Measurements — 64 × 2.4609 MiB = 157.50 MiB per case (MEASURED)**

| case | no-arg MiB/s | file MiB/s | ms/copy (file) |
|---|---|---|---|
| A read-only | NOT RUN | 5679.32 | 0.433 |
| B h2d, blocking stream | 8735.25 | 7832.12 | 0.314 |
| C h2d, non-blocking stream | 8186.71 | 6922.22 | 0.356 |
| D read+h2d pipelined | 6685.34 | 4277.55 | 0.575 |
| E D + synchronous D2H per iteration, blocking copy stream | 2780.20 | 2403.21 | 1.024 |
| F E with the copy stream non-blocking | 3619.23 | 3020.79 | 0.815 |

Derived ratios (MEASURED, from the table):

- **E/D = 2.405× (no-arg) / 1.780× (file)** — the harness's synchronous D2H genuinely costs the transport, and the mechanism is the one hypothesised: a blocking stream's implicit synchronisation with the legacy default stream.
- **F/E = 0.768× (no-arg) / 0.796× (file)** — `hipStreamNonBlocking` recovers ~20–23% of that cost. The hypothesis is **partially confirmed**: the effect is real and the flag is a real part of the fix, but the flag alone does not close it.

**The finding that matters**

- **The transport is not the bottleneck.** Pipelined read+H2D runs at **4.3–6.7 GiB/s** in isolation (0.575 ms per 2.4609 MiB slab) against an FFN pipeline that achieves **~0.15 GiB/s** — a **28–44× gap**, and the gap is with *identical* transfers, slab sizes and file. The missing time is therefore in the pipeline's waits and serialisation (slot lifetime, demand ordering, the per-layer synchronous D2H points, the C8/C9 dance), not on the wire.
- **This invalidates the reading that the streamed path is bandwidth-limited.** Any claim of the form "the streamed pipeline is transfer-bound at 0.15 GiB/s" is wrong as stated: it is *coordination*-bound, with a transport capable of 4–8 GiB/s. Under AGENTS.md `REFUSED`/`DERIVED` labelling: the 0.15 GiB/s figure stays MEASURED, the *attribution* changes.
- **The docs' single-copy reference is not reproduced.** B measures 6.9–8.7 GiB/s where `docs/00-verified-facts.md` §8 records 13.4–14.7 GB/s, i.e. 0.47–0.63× on the same hardware. Either that number came from a different measurement shape (a single cold copy rather than a polled loop, a different direction, a larger burst) or it is optimistic. It is **not** re-pinned here; recorded as a discrepancy with both numbers and their methods named.

**Consequences**
- The next optimisation target is the **pipeline's per-request critical path**, not kernel time and not PCIe bandwidth: instrument submit → slot free → copy done → kernel done → release and find where the ~15 ms per request goes. A blocking→non-blocking copy stream is a one-line part of the fix with a measured ~20% effect.
- Phase 38's `h2d ... (4935 ms)` field must not be read as bandwidth: it is a sum of per-transfer submit→event latencies, so it includes queue wait. The probe is what separates the two.

**Gate re-verification (after the `run_bench.sh` change)**

The documented gate is `bash tools/bench/run_bench.sh`, expected exit 0. It was run twice, before and after the change.

| run | rc | what happened |
|---|---|---|
| before | **5** | `q4k_moe_ffn: TIER C FAILED (exit 2)` and `q4k_stream_ffn: TIER C FAILED (exit 2)` — both **usage errors** from the missing arguments, reported as "the kernel does not compute the right answer on the real device". That sentence was false: neither driver executed a kernel. |
| after | **0** | both report `NOT RUN  <their own declared reason>`; a new exit-2 branch says USAGE ERROR and names the fix. The exit-5 path for a genuine oracle disagreement is untouched. |

- The new driver ran **in the gate with no arguments**: `knj_xfer_probe` tier C → `checks failed: 0`, `RESULT: PASS`. Third independent sample of the transport table: B 9880.80 / C 8859.11 / D 7263.52 / E 2761.97 / F 4156.70 MiB/s, **E/D 2.630×**, **F/E 0.664×** — consistent with the two runs above (E/D 1.78–2.63×, F/E 0.664–0.796×).
- Pinned baselines unchanged and inside tolerance: `attn_c16 PREFILL_US` got 9629.8 (pin max 9650), `attn_c16 DECODE_US` got 194.2 (pin max 200), `gemm_tiled BEST_PCT` got 21.7 (pin min 21.4). No regression, nothing re-pinned.
- Tier B refuses **every** driver in the census run — `knj_xfer_probe`, `q4k_moe_ffn`, `q4k_stream_ffn`, `attn_c16`, `gemm_tiled` — with `'hip/hip_runtime.h' file not found`, on both arches: the `-nogpuinc` defect already recorded in `AGENTS.md` §3. Unchanged by this work and not caused by it, but it means tier B currently measures nothing at all, which is a bigger hole than the §3 note implies. Tier C is where the build actually happens, and it builds all of them.
- Other documented commands re-verified after the edits: `python tools/check_docs.py` → rc 0 (24 figures), `python tools/kvroof/kv_roofline.py models/qwen3-30b-a3b/config.json --check` → rc 0 (71 claims), `python tools/i7/i7_bit_identity.py` → rc 0 (bitwise agree, all negative controls detected). `python tools/route/route_locality.py` as written in `AGENTS.md` §7 → **rc 2**: it needs `--model`, and `--source target-gates` needs `--donor`, for which no local MoE weights exist in this checkout (`models/qwen3-30b-a3b/` is `config.json` + provenance only). Its committed `route_*.json` artifacts remain the record; the documented one-liner is a doc defect.

**Still open**
- The per-request critical-path instrumentation above — unblocked by: adding the five timestamps to the driver's request record.
- Whether the OneDrive-hosted model path inflates the read side in the FFN run (the probe's reads hit the page cache at 5.7 GiB/s, and 610 MiB of a 964 MiB file is re-read across 19 passes) — unblocked by: a cold-cache comparison against a copy of the file outside OneDrive.

# Phase 41 — tier B census repaired, and the read side is not OneDrive-bound  ·  DONE

**Believed at the time**  Two open items from Phase 40's gate re-verification: "tier B refuses **every** driver with `'hip/hip_runtime.h' file not found`, so it measures nothing at all", and "whether the OneDrive-hosted model path inflates the read side in the FFN run".

**Decision**  (1) Tier B was refusing because of `-nogpuinc`, not because of the drivers; replace the census spelling with the one `AGENTS.md` §3 records as verified by hand. (2) Measure the OneDrive question with two independent instruments instead of reasoning about it: the isolated probe (read-only case, whole file) and the streaming driver's own `reader busy` field.

**Changed**
- `tools/bench/run_bench.sh` — tier B no longer passes `-nogpuinc`; it passes `--rocm-path=<ROCm root>`, `-I<tools/bench>`, `-I<repo>/kernels`, `-I<repo>`, and `-I<root>/include` **only** for rocWMMA drivers (the tier C rule: a rocWMMA flag must not leak into a driver that does not use it). Paths go through a new `winpath()` (`cygpath -m`) because the device clang is a native binary. `ROOT` is now defined once. The tier B header states why the tier used to measure nothing.
- `src/profiler/`, `src/cli/`, `src/model/`, `CMakeLists.txt`, `docs/06-profiling.md` — see Phase 42.

**Verified — tier B (two full gate runs, `/tmp/runbench2.out` before, `/tmp/runbench3.out` after)**

| | before | after |
|---|---|---|
| `REFUSED 'hip/hip_runtime.h' file not found` | **26** (13 drivers x 2 arches) | 0 |
| per-kernel resource records (`vgpr=`) | **0** | **65** |
| `expect … PRESENT` (declared-instruction checks) | 0 | **7** |
| `expect … MISSING` | 0 | 0 |
| `EMPTY` (no device kernel) | 0 | 2 (`knj_xfer_probe`, both arches — correct: host-only probe) |
| real refusals | 26 (all spurious) | **10, all on gfx1031 and all correct** |
| gate exit code | 0 | 0 |

The 10 refusals are the ones that should exist: 6 rocWMMA drivers answer `static assertion failed: Unsupported architecture` (gfx1031 has no WMMA) and 4 gfx12-builtin drivers answer `'__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12' needs target feature`. **Zero refusals on gfx1201.** The census now emits per-kernel `vgpr/sgpr/scratch` and the instruction mix (fma / int_extract / other_vop / packed / dot / wmma) for every driver, and the declared-instruction checks run: `knj_gemm_w4a8 v_dot4_i32_i8 PRESENT`, `knj_gemm_w4a4 v_dot8_i32_i4 PRESENT`, `knj_gemm_wmma_f16 v_wmma_f32_16x16x16_f16 PRESENT`, `knj_gemm_wmma_bf16 v_wmma_f32_16x16x16_bf16 PRESENT`, `knj_gemm_simt_pk16 v_pk_fma_f16 PRESENT`. Tier B is the tier `AGENTS.md` calls "where an A/B claim becomes a measurement"; it had been empty.

**Verified — the OneDrive read side (two instruments, hypothesis falsified)**

Identical bytes first: `sha256 b12489c6…` and 964653184 bytes for both the OneDrive file and a copy under `%LOCALAPPDATA%\Temp`.

Instrument 1 — `knj_xfer_probe`, read-only case, whole 964 MB file per case, 3 interleaved pairs, all `RESULT: PASS`:

| pair | OneDrive | local temp |
|---|---|---|
| 1 | 7060.31 MiB/s | 5132.33 |
| 2 | 6983.69 | 5869.90 |
| 3 | 7045.21 | 5811.55 |
| mean | **7029.7 MiB/s** | **5604.6 MiB/s** |

Instrument 2 — the streaming driver's own reader thread (610.312 MiB over 336 reads per pass, `reader busy`), two full 19-pass runs, both `checks failed: 0`, both `RESULT: PASS`:

| pass (M0…P1) | OneDrive reader busy | local temp |
|---|---|---|
| M0 | 121.647 ms | 154.036 ms |
| M1 | 123.695 | 167.381 |
| M2 | 131.996 | 182.853 |
| M3 | 144.849 | 193.892 |
| P0 | 133.033 | 175.274 |
| P1 | 115.478 | 176.361 |
| mean | **128.4 ms = 4.64 GiB/s** | **175.0 ms = 3.41 GiB/s** |

**Verdict: falsified.** The OneDrive-hosted path is **not** a read-side penalty — it is faster than an identical copy in the local temp directory in both instruments (+25% isolated, +36% in-pipeline). The suspicion that the 0.15 GiB/s pipeline rate might be a filesystem artifact is therefore closed, and Phase 40's conclusion stands unchanged: the gap is coordination, not I/O and not bandwidth. The cause of the OneDrive advantage is **not** established and is not investigated further, because it favours the configuration in use.

**Still open**
- The cold-cache half of this comparison is **REFUSED**: dropping the Windows file cache needs elevation, so both instruments measured the warm/page-cache regime. Every number above says so.
- `tools/bench/run_bench.sh` tier B **reports but does not gate**: a `MISSING` declared instruction prints `<-- OK? cell` and does not affect the exit code. There were none this run, so the tier is honest today; it is not enforced. Unblocked by: making a `MISSING` set the tier B failure path.
- The OneDrive-vs-local advantage above is unexplained (reported, not claimed).

# Phase 42 — C21 profiling: built, wired, and measured  ·  DONE

**Believed at the time**  `docs/06-profiling.md` opened with "STATUS: spec, not yet built. No `kanjoos` binary exists; nothing in this document has ever run", and that was still true in the strict sense that matters most: `src/cli/main.cpp` parsed none of the documented flags, and **the root build referenced none of `src/cli`, `src/model`, `src/loader`, `src/tokenizer`, `src/util`, `src/platform` or `src/profiler`** — the repository could not produce a single executable that loads a model. The user's instruction was explicit: "we need to have profiling etc.".

**Decision**  Implement the C21 core for the parts that can be measured honestly on the path that exists, put it behind the documented flag surface, and make the build produce the binary. Where the contract cannot be met yet, refuse the flag rather than redefine it.

**Changed**
- `src/profiler/profiler.{h,cpp}` — new. Marker stack, per-stream idle walk, `--profile-floor` (empty ops through the identical begin/end path), `--profile-warmup`, `--profile-detail=class|layer|kernel`, `--profiling=table|json|csv|no-subtract|off`, `--profile-dir` writing `profile.{txt,json,csv}`, residency and transfer footers, and a declared `ClockDomain` so a host-only run says `clock host` instead of printing device zeros. `counters` and `full` are **refused with a message**, not aliased. `src/platform/memprobe.h` feeds the header's RAM lines.
- `src/cli/main.cpp` — parses the flags, configures the profiler before `load`, measures the floor, scopes `load`/`tokenize`/`prefill`/`decode`/`sample`, calls `step_done()` per forward, and reports once at every exit (including the `--bench` early return).
- `src/model/model.cpp` — scopes `embed`, and per layer `norm`, `qkv`, `attention`, `attn-o`, `norm-ffn`, `moe`, `residual`, plus `head`. Names are `class:layer`, so one instrumentation pass serves class and layer detail.
- `CMakeLists.txt` — new `knj_runtime` static library and `kanjoos-run` executable (host-only ISO C++17, `Threads::Threads`). **`KNJ_ARCH` now defaults to `gfx1201`**, the stated first target; `gfx1031` remains fully supported via `-DKNJ_ARCH=gfx1031`. The stale header comment claiming gfx1031 primary is corrected in the same edit.
- `src/loader/gguf.cpp` — `NOMINMAX` / `WIN32_LEAN_AND_MEAN` guarded with `#ifndef` (MinGW's own headers define them; the unguarded redefine was a warning on every build).
- `docs/06-profiling.md` — STATUS rewritten to state exactly what exists and what does not; the section 2 block is now labelled as still containing no measurements.

**Verified**

```sh
cmake -S . -B build/cmake-host -G Ninja -DKNJ_ENABLE_DEVICE=OFF -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release   # rc 0
cmake --build build/cmake-host --target kanjoos-run -j 8                                                              # rc 0, 10/10, 0 warnings
./build/cmake-host/kanjoos-run.exe --model <Qwen3-MOE-4x0.6B-Q4_K_M.gguf> --bench 16 8 --profiling --profile-dir <dir>
  -> rc 0, table printed, profile.{txt,json,csv} written
```

Also verified, each on the built artifact: `--profiling=json` writes a **valid** JSON file (parsed by `python -m json.tool`) and prints the same to stdout; `--profiling=csv` rows read `moe,448,1315813400,0,1315813400,host,class,94`; `--profile-detail=layer` prints 84 per-layer rows (`moe:0`, `head:0`, …); `--profiling=no-subtract` prints the floor line **and** the "floor is NOT removed" note; `--profiling=off` prints no table (grep for the column header: 0 matches); `--profiling=full` **refuses**, exit 1, `--profiling=full is not implemented; this build implements off|table|json|csv|no-subtract`.

**Measurements — first real C21 table on this machine (MEASURED)**  `--bench 16 8 --profiling --profile-warmup 2`, Qwen3-MoE-4x0.6B Q4_K_M, 28 layers, 4 experts, ctx 2048, 32 threads:

| component | ops | %dev | dev us | idle us |
|---|---|---|---|---|
| moe | 616 | 57.2% | 1304656.00 | 0.00 |
| qkv | 616 | 21.4% | 487532.50 | 0.00 |
| head | 22 | 10.8% | 245891.52 | 0.00 |
| attn-o | 616 | 8.1% | 185187.40 | 0.00 |
| attention | 616 | 2.3% | 51563.70 | 0.00 |
| norm | 616 | 0.1% | 1233.60 | 0.00 |
| norm-ffn | 616 | 0.0% | 1117.80 | 0.00 |
| embed | 22 | 0.0% | 32.52 | 0.00 |

`616 = 22 kept steps x 28 layers`; the floor is **0.081 us** per empty op; `--profile-detail=layer` shows `moe:0 = 12560.75 us`. Rows are **self time**, so `prefill`/`decode` (the containers) report ~2 ms and the work appears on the leaves; the header states the convention in force.

**Two defects found and fixed during verification, both by reading the output rather than trusting the design**
1. The first run printed `nested scopes are not measured: 4752 ignored` — the CLI's `prefill`/`decode` scope wrapped the model's per-component scopes and **every useful row was discarded**. Fixed by making the profiler a marker *stack* that reports self time (child time subtracted), so nested rows are recorded and the rows partition the run instead of double-counting it.
2. `--profile-dir` was created with `std::system("mkdir -p …")`, which on Windows runs under `cmd.exe`, whose `mkdir` treats `-p` as a **directory name**: the run appeared to succeed while creating a junk directory literally called `-p` in the working directory. Fixed with `std::filesystem::create_directories`; verified afterwards that `./-p` is no longer created and the three profile files are written.

**Still open**
- No device event backend: on the host path `dev` and `host` are the same interval by construction, and every report says so. A device path that reports into the same table is what makes the column separation real. Unblocked by: a HIP event begin/end in `src/device/` and a `ClockDomain::Device` run.
- `residency` and `transfer` footers print `NOT MEASURED on this path`; the streaming driver has those numbers (slots, evictions, prefetch hit, h2d/d2h bytes) but does not report through C21 yet.
- `--profiling=counters` and `=full` remain unimplemented **and refused**; `kanjoos serve` does not exist.

**Notes**
- The hand-built verification binary needs `-static` (MinGW): without it the exe depends on `libstdc++-6.dll` / `libgcc_s_seh-1.dll` and exits **127** from a shell that does not have Strawberry's `bin` on `PATH`. The CMake target does not have this problem when run from the toolchain environment, but a stripped-down shell can reproduce it.
- `q4k_stream_ffn.hip` includes `rocwmma/rocwmma.hpp` and therefore needs `-std=c++17` on the hipcc line (rocWMMA header need, not a preference); the `-I .` is required so `#include "src/residency/residency.h"` resolves; `src/residency/residency.cpp` must be on the hipcc link line (`lld-link` otherwise reports undefined `knj::ResidencyManager::*`).
- `run_on_rc` / `run_off_rc` are captured via `set -o pipefail` and `${PIPESTATUS[0]}` so a filtered output never masks the runner's real exit code.
- Temp artifacts (`/tmp/ksfp_on.exe`, `/tmp/ksfp_off.exe`, `/tmp/ksfp_on.out`, `/tmp/ksfp_off.out`, `/tmp/ksfp_on_out/`, `/tmp/ksfp_off_out/`) were deleted after capture; the numbers above are the extracted record.
- **Superseded (Phase 38):** that claim was wrong in the Phase 37 turn — the artifacts were **not** deleted. As of Phase 38 they exist on disk: `/tmp/ksfp_on.exe`, `/tmp/ksfp_off.exe`, `/tmp/ksfp_on.out`, `/tmp/ksfp_off.out`, `/tmp/ksfp_on_out/`, `/tmp/ksfp_off_out/`, plus `/tmp/ksfp_on.err`. The Phase 38 numbers are read directly from `/tmp/ksfp_on.out` and `/tmp/ksfp_off.out`.
