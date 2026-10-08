# 03 — Kernel strategy per target

One architecture, two kernel backends behind C2. This document says what each
target actually has, what is built from it, and how each kernel is proved
correct.

The measured capability matrix is in `00-verified-facts.md` §1 and is
reproducible with `tools/isa_probe/run_isa_probe.sh`. Everything below is
consistent with that matrix, not with the assumption that the two targets are
alike.

---

## 1. gfx1201 (RDNA4)

### 1.1 WMMA is available — the gate is a build flag

**gfx1201 has a working matrix path.** It emits a real matrix instruction:

```
v_wmma_f32_16x16x16_f16 v[9:16], v[0:3], v[4:7], v[9:16]
v_wmma_f32_16x16x16_bf16 v[9:16], v[0:3], v[4:7], v[9:16]
```

The invocation is:

```c
__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a /*v8f16*/, b /*v8f16*/, c /*v8f32*/);
```

built with the single target feature

```
-Xclang -target-feature -Xclang +wavefrontsize32
```

Three things this replaces, all of which were previously written as unknowns:

| old belief | measured |
|---|---|
| WMMA needs an unknown `wmma-256b-insts` flag | no. That feature cannot be enabled in this build; the `_gfx12` builtin needs only `+wavefrontsize32` |
| `_gfx12` takes A = 16×f16, B = 8×f16 | **A and B are both 8-wide.** The 16-wide shape belongs to the gfx11 `_w32` builtin, which is refused anyway |
| gfx1201 must also carry a SIMT path | it still must, for numerics fallback, but it is no longer the *expected* path |

Operand shape, measured from the emitted code object:

```
v_wmma_f32_16x16x16_f16 v[9:16], v[0:3], v[4:7], v[9:16]
                         D         A       B        D
```

| operand | registers | contents |
|---|---|---|
| A | `v[0:3]` = **4 VGPRs** | 8 × f16 |
| B | `v[4:7]` = **4 VGPRs** | 8 × f16 |
| C/D | `v[9:16]` = **8 VGPRs** | 8 × f32, f32 accumulate |

A and B are **4 VGPRs each, not 8**. This corrects the earlier note in this
file and in `00-verified-facts.md` §7.3.

“8 × f16 in 4 VGPRs” is a *register count*, not a statement about which
logical elements a lane holds. §7.4a of `00-verified-facts.md` records a
four-probe layout measurement on gfx1201 showing the lane→element mapping is
**not** the naive “8 consecutive f16, lane L holds row L” assumption. Do not
write a tiled WMMA GEMM against that assumption.

Do **not** use the gfx11 `_w32` spelling. It passes the frontend feature check
with 16×f16 operands and then **segfaults the backend in Branch relaxation**.
That is a compiler bug in this snapshot; the runner reports it as `CRASH`, not
`REFUSED`, precisely so nobody builds on it.

This corrects the sibling kanjoos project's `RESEARCH.md`, which records 8 fp16
per lane for both operands — that part was right; the 16-wide reading belongs to
the `_w32` builtin, not `_gfx12`.

### 1.2 Throughput ceiling

From AMD's published R9700 figures (gfx1201, 64 CU), which the 9070 XT shares
the die with at a lower memory clock:

| element type | dense | sparse |
|---|---|---|
| FP16 / BF16 | 191 TFLOPS | 383 |
| FP8 E4M3 | 383 TFLOPS | 766 |
| INT8 | 383 TOPS | 766 |
| INT4 (iu4) | 766 TOPS | 1531 |
| **FP4** | **absent** | **absent** |

Two design consequences:

* **FP8 is the compute story on RDNA4**, not FP4. NVFP4/MXFP4 save bandwidth but
  the matmul still lands on an FP16 WMMA after a dequant prelude, so they buy
  footprint, not throughput. The KV codec default (C15) and the FP8 expert path
  both follow from the FP8 row.
* **INT4 WMMA exists at 4× FP16.** It is not the default because unsigned-4-bit
  matmul calibration is closer to QAT than to the post-training quantisers the
  GGUF ecosystem standardised on. It is a researched option with a quality gate.

### 1.3 What is built

| kernel | math | notes |
|---|---|---|
| attention prefill | WMMA f16/f8, LDS tiles | two-tile ping-pong, selective rescale |
| attention decode | WMMA f16/f8 | KV dequant-to-FP16 in the prologue when KV is FP8 |
| expert W8 | WMMA i8 + scale | or dequant-to-fp16 when int calibration is poor |
| expert W4 (default) | unpack + WMMA f16 | pack format chosen so unpack is LDS-cheap |
| expert FP8 | native WMMA f8 | the highest-value path on this arch |
| norms / RoPE / quant | fused SIMT | memory-bound, no matrix |
| router top-k | SIMT + sampled-boundary certify | |

### 1.4 Tile geometry

Follows the WMMA guide parts already read for this project, restated as the
concrete starting point:

* WMMA shape 16×16×16, `wave32` only. Never assume wave64.
* Because both A and B fragments are indexed by `lane % 16` on both
  generations, both operand tiles live in LDS **row-major** `[idx][k]` — that is
  what makes `load_a` and `load_b` mirror images of each other.
* Low-precision path: 128-bit LDS loads, achieved by padding tiles with
  `LD_PAD = 8` half-elements. The pad removes bank conflicts and keeps the
  per-lane 8-element run naturally aligned.
* The in-register transpose from part 3 of the guide is **deliberately not
  used**: B is produced by our own dequantiser straight into LDS, so its layout
  is a free choice.

### 1.5 Overlap patterns

From the HipKittens work, the two patterns that recur across its kernels:

* **8-wave ping-pong**: two workgroups alternate compute and LDS/global traffic,
  so one group's memory phase covers the other's math phase.
* **4-wave interleave**: finer-grained version, better when LDS pressure is low.

Both map cleanly onto the FA4 two-tile schedule in C16. Default to 8-wave
ping-pong for attention; interleave for the expert GEMM where LDS pressure is
lower and the tiles are smaller.

---

## 2. gfx1031 (RDNA2)

### 2.1 What there is, verified

```
v_pk_add_f16   OK
v_pk_fma_f16   OK          <- the entire compute basis
__builtin_amdgcn_sdot4 (4-scalar-int form)  OK   on gfx1031, REFUSED on gfx1201
WMMA           REFUSED     (no matrix units — hardware, not configuration)
inline-asm dot OK         (v_dot4_i32_i8, v_dot8_i32_i4, v_dot2c_f32_f16 all assemble)
sdot4 builtin OK          (lowers to v_dot4c_i32_i8 — a NATIVE dot instruction)
```

* **No matrix units.** Every multiply-add is VALU. This one refusal is hardware
  and there is no flag for it.
* **Packed FP16 is the throughput lever**: `v_pk_fma_f16` does two lanes per op,
  and `v_pk_mad_f32` is available for the mixed-precision accumulate.
  `v_mad_mix_f32` remains CDNA-only and is rejected on gfx1031; use `v_pk_*`.
* **Dot products: write them as inline asm, once, shared across both arches.**
  This is a reversal. The previous "two `#if`-selected headers" policy existed
  only because inline-asm dots appeared not to assemble — that was an artifact
  of a broken toolchain. Measured, `v_dot4_i32_i8`, `v_dot8_i32_i4` and
  `v_dot2c_f32_f16` assemble on *both* gfx1031 and gfx1201, so one header serves
  both targets.
* **Prefer the asm over the `sdot4` builtin even on RDNA2.** The builtin lowers
  to `v_dot4c_i32_i8`, a *clamping* variant with different semantics, and it
  does not compile for gfx1201 at all.
* **hipBLASLt does not exist in the gfx1031 runtime.** rocBLAS-classic, the
  Tensile kernels in the blas kpack, or hand-written SIMT.

### 2.2 Int8 on RDNA2 is real

`sdot4` on gfx1031 lowers to `v_dot4c_i32_i8` — a single native dot-4
instruction, **not** a shift/add expansion. The W8/W4 integer expert path is
therefore viable on the 6700 XT.

The right shape for RDNA2 is mixed, and this is the one place it differs from
RDNA4:

| weight format | RDNA2 | RDNA4 |
|---|---|---|
| W8 | `v_dot4_i32_i8` int8 path | `v_dot4_i32_i8` int8 path, or dequant to fp16 + WMMA |
| W6 | int8 dot at group size 64 — **unpack is real work on both arches** | WMMA f16 after dequant |
| W4 | `v_dot8_i32_i4` — int4 dot is the natural fit and avoids unpack entirely | WMMA f16 after dequant |

W4 on gfx1031 is not a compromise. `v_dot8_i32_i4` consumes the packed nibbles
directly, so the W4 path has **no unpack step** — which is precisely the case
the original plan worried about. P1-2 in `00-verified-facts.md` settles group
size 64 vs 128 empirically.

### 2.3 What is built

| kernel | approach | notes |
|---|---|---|
| attention prefill | SIMT `v_pk_fma_f16`, LDS tiles, register blocking | explicit software pipelining; the work is in hiding LDS latency |
| attention decode | SIMT packed FP16 | bandwidth-bound on LDS, not on VALU |
| expert GEMM | packed W4/W6/W8 → FP16 in registers → `v_pk_fma_f16` | **pack format must make unpack cheap**, because on this arch unpack is real work |
| norms / RoPE / quant | fused SIMT | as above |
| CPU fallback | same math on the host | shares the reference implementation shape |

The dequant-in-register pattern for RDNA2, spelled out because it is the whole
kernel:

```
per lane:
  load  packed 4-bit codes for 8 weights  (one 32-bit load, LDS or VGPR)
  load  fp16 scale for the group
  unpack codes -> fp16 via nibble extract + lop3 select + fma into the register
  v_pk_fma_f16 against the activation registers
  accumulate in f32
```

Two load-bearing details:

* **The activation tile stays in FP16 in registers for the whole K loop.** The
  packed weight stream is what rotates through LDS. That way `v_pk_fma_f16` runs
  back to back with no unpack in the inner dependency chain.
* **Group size 128 is the quantiser's choice, but 64 may win here.** Smaller
  groups mean fewer unpack ops per scale load and better scale reuse in
  registers; on an arch where unpack is real work, that can beat the accuracy
  advantage of G=128. This is an autotuner question (C3), not a policy question.

### 2.4 What is *not* available, so is not planned

* FP8 KV or FP8 experts — no hardware support. Any FP8 proposal for gfx1031 is
  a proposal to emulate, and emulation here loses more than it saves.
* FP4 — same.
* hipBLASLt tuned GEMM — absent from the runtime.
* WMMA — absent, permanently, and no amount of build-flag archaeology will
  produce it.

The RDNA2 target is therefore honest about what it is: **a capacity and transfer
engine with a competent SIMT compute path**, aimed at 12 GiB VRAM × 48 GiB RAM,
where the RAM warm tier does the heavy lifting.

---

## 3. Cross-target rules

1. **One tiling abstraction, two implementations.** Tiles are expressed once in
   C16/C17 terms (M/N/K, group size, quant format) and lowered per arch. The
   scheduling, residency and transfer code never sees an instruction.
2. **One reference implementation, in portable C++ on the host.** Every device
   kernel is validated against it. The host dequantiser is the oracle, exactly as
   the project's GGML block-layout work already established; when host and device
   change, they change together.
3. **Every kernel gets a correctness test that runs on CPU and GPU and compares
   against the same input**, in CI, per arch, with a tolerance stated per format.
4. **No kernel is enabled by default until it is measured.** A format that loses
   to a simpler format on the target does not ship, regardless of how good it is
   on paper.

---

## 4. Validation matrix

| kernel | inputs | oracle | tolerance |
|---|---|---|---|
| unpack W4/W6/W8 | known packed tensor | host dequantiser | bit-exact |
| expert GEMM | random A, packed B, M ∈ {1,2,4,8,64} | host materialise + GEMM | fp32 rtol 1e-4 |
| grouped expert GEMM | routed token set | per-expert reference, summed | fp32 rtol 1e-4 |
| attention prefill | random Q/K/V, GQA/MQA | host flash reference | fp32 rtol 2e-3 |
| attention decode | same, single query row | host reference | fp32 rtol 2e-3 |
| KV codec round-trip | random K/V through write→store→load→read | host codec | format-specific |
| RadixKV reload | model logits before/after evict+reload | same run, no eviction | **bit-identical** |

The last row is the one that tests invariant I7, and it is the one that catches
the class of bug that makes a tiered cache untrustworthy: any drift introduced
by the storage round trip rather than by the maths.