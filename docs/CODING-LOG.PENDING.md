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

**Notes**
- `q4k_stream_ffn.hip` includes `rocwmma/rocwmma.hpp` and therefore needs `-std=c++17` on the hipcc line (rocWMMA header need, not a preference); the `-I .` is required so `#include "src/residency/residency.h"` resolves; `src/residency/residency.cpp` must be on the hipcc link line (`lld-link` otherwise reports undefined `knj::ResidencyManager::*`).
- `run_on_rc` / `run_off_rc` are captured via `set -o pipefail` and `${PIPESTATUS[0]}` so a filtered output never masks the runner's real exit code.
- Temp artifacts (`/tmp/ksfp_on.exe`, `/tmp/ksfp_off.exe`, `/tmp/ksfp_on.out`, `/tmp/ksfp_off.out`, `/tmp/ksfp_on_out/`, `/tmp/ksfp_off_out/`) were deleted after capture; the numbers above are the extracted record.
