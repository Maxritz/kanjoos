# Worksheet — tools/bench (existing microbenchmark harness)

**What it is:** `tools/bench/run_bench.sh` + `tools/bench/*.hip` — the tier A / tier C microbenchmark
harness. Tier A: 2 suites / 13 checks PASS. Tier C: runs `expert_gemm`, `wmma_layout`, `wmma_run` on the
real GPU, with four outcome states (PASS, exit 6 = arch mismatch skip, REFUSED/UNBUILT = arch legitimately
lacks the instruction, exit 5 = built for no arch at all — a real failure).

**Why it matters to a coder:** this is where the measured numbers in `00` §7 and §8 come from, and where
silent-wrong-answer defects have been caught (every one compiled cleanly, produced a plausible number, and was
wrong). The harness's discipline — full-output correctness sweep, four outcome states, arch guard — is the model
for any new device test.

**Measured facts this harness produced (source of truth, `00` §7–§8):**
* §7.4 — tier C on the real device (1 wave, launch overhead included): SIMT f32 = 1.861 µs (64 `v_mad_f32`);
  SIMT packed f16 = 2.629 µs (32 `v_pk_fma_f16`); WMMA chain = 1.584 µs (8 `v_wmma_f32_16x16x16_f16`).
  Ratios 1.20× vs SIMT f32, 1.60× vs SIMT packed (reproduced 1.15–1.20× and 1.60×). These are **not** an
  expert-GEMM speedup — the three kernels do not compute the same quantity, and at this size the numbers are
  dominated by launch cost. They show WMMA's 8-against-64 issue advantage is real but at K=64 the win is far
  below 2×. **The 2–4× claim remains UNVERIFIED** and now has a number on the side that contradicts it.
* §7.4a — the `_gfx12` A/B operand layout is **not** 8 consecutive f16. Measured on the device with a control
  probe: all eight one-hot A probes give the **same** D, independent of which lane was set. No layout in which A
  is 8 consecutive f16 predicts that — such a layout predicts eight different masks. So the builtin does not consume
  A as one vector of 8 halves, and **a tiled expert GEMM written on that assumption would read the wrong elements.**
  This is the **blocking unknown for the gfx1201 WMMA path.** The control's role: before it was added, the same run
  printed an EMPTY mapping table (a dead kernel). The control is the only reason this reads as a finding rather than
  a mystery.
* §7.5 — six silent defects caught by the CPU-reference tier, all compiled cleanly: (1) packer used a byte offset
  as a weight index, so blocks overlapped and the last 60 of 128 weights were never written; (2) low and high nibbles
  given the **same** activation — weight `2b` and `2b+1` need different ones; (3) `v_lshrrev_b32` modelled on the host
  as a **whole-word** shift, dragging the next byte's low nibble into this byte's high nibble; (4) the fix for (3),
  `(x & 0x0f0f0f0f) << 4`, is an **identity**, so `u_hi` silently equalled `u_lo` for every weight; (5) path A's
  weights packed with the path-F unsigned-with-zero-point layout, but `v_dot8_i32_i4` **sign-extends**, so they were
  read as u-16 instead of u-8; (6) device packed-f16 kernel accumulated into an **uninitialised** register
  (`v_pk_fma_f16` is an accumulating form) — 10× the oracle error.
* §7.6 — W4 pack and group size settled: nibble-packed weights, **unsigned** nibbles `u = q + zp`, int8 activations,
  `v_dot4_i32_i8` with a de-interleaved activation layout. Per 8 MACs: 1 weight load + 2 extract + 2 dot = 0.625
  instr/MAC. Alternate: `v_dot8_i32_i4` with both operands nibble-packed, 0.250 instr/MAC (2.5× fewer, genuinely zero
  unpack) — **rejected** because it needs **4-bit activations**, measured at 81.5 dB worse SNR than int8. Group size 128
  for expert weights, 64 stays the default for KV. group64 costs 4.29% more bytes (0.5469 vs 0.5234 B/weight) = 40
  MB/token = ~10 ms of a ~250 ms token budget, to buy ~0.2–0.3 bits. The int32 accumulator is not a reason to pick 64:
  worst case at group 128 is 243,840 against 2,147,483,647, a headroom of 8807×.
* §8.0 — expert GEMM correctness: all **24,576** output elements of one `[32×2048] * [2048×768]` projection checked
  against a `double` host oracle — not a strided sample. Relative RMSE `1.008e-03`; max abs error 0.0157 = 5.18e-03 of
  the 3.034 output RMS; exactly-zero outputs 0 of 24576. `1e-3` is the expected floor (fp16 inputs, fp32 accumulation,
  K=2048) — it is not "near zero" and should never be asserted as such.
* §8.1 — both bandwidths measured: VRAM stream 512 MB → 589–598 GB/s; PCIe stream 256 MB pinned → 27.9–28.0 GB/s;
  **PCIe one 2.47 MB expert copy → 13.4–14.7 GB/s** (the size that matters, the noisiest cell because 2.47 MB is small
  enough that per-copy latency shows up in the average). Ranges, not point values — the spread observed across four runs.
* §8.2 — derived hardware peak 39.3 TFLOP/s packed-f16 = 32 CU × 128 lanes × 2 × 2 × 2400 MHz (derived, not quoted, so
  a card swap cannot silently invalidate it).
* §8.3 — the verdict: one expert at M=32: 302 MFLOP, 2.47 MB W4 group128. Arithmetic at 100% of derived peak = 7.68 µs;
  arithmetic in this kernel (measured) = 146.5 µs (19× off peak); transfer over PCIe (measured 13.6 GB/s) = 181.3 µs.
  **Even at 100% of peak, one expert's arithmetic is 22× cheaper than moving its weights.** Per decode block of 32 tokens:
  8 distinct experts → 19.8 MB, 1.343 ms transfer, 0.061 ms arithmetic (peak) → TRANSFER-bound; 68 → 168.0 MB, 11.417 ms,
  0.522 ms → TRANSFER-bound; 128 (worst case) → 316.1 MB, 21.491 ms, 0.983 ms → TRANSFER-bound. No tiling, no WMMA, no
  amount of GEMM work changes this. Only a smaller weight format does. This retires the WMMA question as an architecture
  priority: worth hours, not weeks.
* §8.4 — how a wrong verdict was nearly shipped: an earlier revision divided 2.47 MB by the card's 580 GB/s and concluded
  "compute-bound by 217×". **Two independent errors cancelled:** (1) wrong roofline — expert weights arrive over PCIe from
  host RAM (Kanjoos's entire premise); 580 GB/s is the card's own VRAM; (2) the kernel it measured ran at 1% of its own
  roofline, so the arithmetic side was a property of the code, not the silicon. The output looked authoritative and was
  fictional. The fix is structural: measure both sides, and when one side cannot yet be measured, label it derived and say
  which verdict depends on it.
* §8.5 — toolchain facts learned here (both `-nogpulib` properties): **no `gridDim` in any kernel** on gfx1201 (`-nogpulib`
  → `__ockl_get_num_groups` unresolved, link fails with `undefined hidden symbol`; grid-stride loops must take the stride as a
  kernel argument); **no 64-bit indexing needed** (`uint32_t` suffices even for 512 MB); `+wavefrontsize32` is rejected as a
  target feature (gfx1201 is natively wave32, the flag is inert).
* §8.6 — four silent defects in the expert GEMM, all compiled cleanly, all produced a plausible number: (1) `wave = threadIdx.y`
  with `blockDim(64,2)` — waves are cut from the **linear** id, so `wave 1` is `tx 32..63` of `ty 0`, never a wave index;
  (2) a wave claimed 16 rows but addressed only `row` and `row+1`, so every block wrote 4 of its 32 rows — RMSE 0.90 with the
  written rows **exactly right**; (3) an f16 temporary named `acc` shadowed `float acc[]`, so `acc[j] +=` indexed a single
  vector — every output stayed exactly zero, RMSE 1.000; (4) `(tid & 63) * 2` gave each thread two columns across 0..126 while
  the block tile is 64 wide — RMSE 1.13. A strided element sample, a passing exit code and a plausible number are all consistent
  with total nonsense. **Check every output.**
* §8.7 — three more toolchain facts, all found by **refusing to run a mismatch**: (1) **an arch-mismatched binary runs and lies** —
  a build with `--offload-arch=gfx1031` executes on the gfx1201 card with no error; the kernel is never dispatched, every output
  keeps its `hipMemset` value, and the timing table prints **2327% of peak in 0.1 µs**. Only the full-output correctness sweep
  caught it. Every tier C driver now refuses to run unless `KNJ_BUILD_ARCH` matches
  `hipGetDeviceProperties().gcnArchName`. (2) **the guard cannot be written with `-D`, and cannot use `__gfxNNNN__`** — `__gfx1201__`
  is a device-pass macro; `main()` is host code, so a guard written that way compiles to nothing and silently guards nothing. The
  `-DKNJ_BUILD_ARCH="gfx1201"` spelling fails too: hipcc re-spawns clang through a command string, the inner shell eats the quotes,
  and the compiler reports `use of undeclared identifier 'gfx1201'`. The arch is therefore passed through the **environment**, which
  does not pass through that layer. A driver with no `KNJ_BUILD_ARCH` set exits 7 and refuses. (3) `/g/ROCM10RT-gfx1031/bin/hipcc.exe`
  is **not** a gfx1031 toolchain — it execs `G:\ROCM10RT-gfx1201\lib\llvm\bin\clang.exe`, a different, live ROCm tree. So gfx1031 has
  a working `clang` for compiling and **no working `hipcc` for linking**. Tier C on gfx1031 is a real gap on this machine, and
  `kanjoos doctor` should keep reporting it as one rather than as "absent".
* §8.8 — runner status: `tools/bench/run_bench.sh` exits **0**: tier A 2 suites / 13 checks PASS, tier C runs `expert_gemm`,
  `wmma_layout` and `wmma_run` on the real GPU, with the four outcome states.
* §8.9 — expert GEMM retuned: 0.8% → 5.2% of peak (324 → 48.8 µs, 6.6×), by (1) 1 col → **2 cols** per thread (a 32-lane
  group, not 64) → 1.5 → 0.62 loads per FMA, 16 independent accumulator chains; (2) row bound hoisted out of the k loop → removed
  8 compare-and-branch per unrolled iteration that never fire; (3) **split-K swept 1 → 64** → 12 → 768 blocks, 6.0 → 96 waves/CU.
  Correctness held at relative RMSE **1.008e-03** across every change: all 24576 outputs against a `double` oracle, zero unwritten.
  **Why 5.2% is the honest number to stop at:** at M=32 this projection has minimum bytes that must move = 4.72 MB (B once, A once per
  column block); arithmetic intensity = **21.3 FLOP/byte**; machine balance (peak / BW) = **66.7 FLOP/byte**; **hard ceiling at M=32 =
  32.0% of peak, whatever the code does.** 21 FLOP/byte cannot buy 67 FLOP/byte of machine. This kernel reaches 16.4% of that ceiling.
  **Raising M is the only lever** — more tiling, more registers, more split-K cannot beat an intensity ceiling. That is why this number is
  now trustworthy enough to quote.
* §8.10 — what bit-width buys, at measured 13.6 GB/s, full 128-expert sweep (worst case: all 32 tokens route to disjoint experts):
  W4 g128 (0.5234 B/weight, 2.47 MB, **23.25 ms**, 23.6× transfer-bound); W3 g128 (0.3984, 1.88 MB, **17.69 ms**, 18.0×);
  W2 g128 (0.2734, 1.29 MB, **12.14 ms**, 12.4×). **Every row uses one metadata rule: 3 B per group, whatever the bit width.**
  Two separate inconsistencies corrected: (1) the W4 row said 0.5186 B/weight — `gemm_w4.hip` defines 0.5234 (0.5 payload + 3/128),
  and §8.1's own measured "2.47 MB expert copy" already assumed the correct size — the table and the bandwidth table had been quoting
  different packs; (2) the W3 and W2 rows used ~2.4 B/group while the W4 row used 3 — that is not a convention, it is two packs in one
  table, and it made W3 and W2 look cheaper than they are. `tools/kvroof/kv_roofline.py` section K derives all of this from one function
  (`kvroof.pack_bytes_per_weight`) so the rows cannot drift apart again.

**What a coder must preserve (the harness's discipline):**
1. **Full-output correctness sweep against a host oracle**, not a strided sample. The expert GEMM checks all 24,576 outputs; the
   WMMA layout probe checks the operand geometry with a control. Any new device test must check every output (or every element of a
   stated tile), not a sample, and must name the oracle.
2. **Four outcome states for tier C** — PASS, exit 6 (arch mismatch, a skip and explicitly not a result), REFUSED/UNBUILT (arch
   legitimately lacks the instruction — e.g. gfx1031 has no WMMA), exit 5 (built for no arch at all, a real failure). Do not collapse
   these into "pass/fail".
3. **Arch guard is mandatory** — `KNJ_BUILD_ARCH` must match `gcnArchName`, passed through the environment, and the runner must **prove
   the guard works before trusting a run** (run the mismatched binary first, assert exit 6, then run the real one). The mismatched binary
   runs and lies (2327% of peak in 0.1 µs) — the proof-of-guard step is not optional.
4. **A control probe for every geometry claim** — the `_gfx12` layout finding is only a finding because of the one-hot-A control; without it,
   the same run printed an EMPTY table and looked like a dead kernel. Any new test that claims a layout/operand-geometry/result must carry a
   control that would detect a dead or silent-wrong kernel.
5. **Ranges, not point values, when the measurement is noisy** — PCIe cells move 2–4% between runs (`00` §8). Quote the range and say which
   conclusions survive the whole range.
6. **Derived numbers are labelled derived, and the verdict that depends on them is named.** The 39.3 TFLOP/s peak is derived; the §8.3 verdict
   depends on both that and the measured 13.6 GB/s; §8.4 exists because an earlier revision blurred this and got a fictional "compute-bound by 217×".
7. **The "check every output" rule applies even when the number looks plausible.** Four silent defects in the GEMM all produced plausible numbers
   (RMSE 0.90, 1.000, 1.13 — and one of them wrote exactly the right rows and exactly the wrong rows, so a sample would have passed).

**Reproducibility:** this turn, `bash tools/bench/run_bench.sh` was not re-run (it requires the real gfx1201 and the `-nogpulib` toolchain),
but the I7 GPU leg (a tier-C-style device test with the same arch-guard discipline) was run and passed exit 0. The harness's four-outcome-state
and arch-guard discipline is the model for any new device test.

**Files a coder should read before touching this:**
* `tools/bench/run_bench.sh` and every `tools/bench/*.hip` (the source — read the arch-guard proof, the four-outcome-state logic, the full-output
  correctness sweeps, the control probes).
* `docs/00-verified-facts.md` §7.4, §7.4a, §7.5, §7.6, §8.0–§8.10 (the measured facts and the defect catalog).
* `docs/00-verified-facts.md` §0 (standing rule: a capability claim must name the emitted instruction; an exit code is not evidence).
* `docs/02-components.md` — C17 (expert GEMM — the component this harness measures), C16 (attention kernels).
* `docs/00-verified-facts.md` §8.7 (the arch guard, why it is mandatory, why `-D` and `__gfxNNNN__` do not work, why the environment is the
  only spelling).
