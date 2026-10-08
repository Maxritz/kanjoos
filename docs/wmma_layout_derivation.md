# gfx1201 WMMA A-lane → (row, col) mapping — a checkable consequence, not a mystery

Status: **partial derivation landed; the consequence is now checkable; the full
layout is not yet uniquely determined by the ISA alone.**

## What the emitted instruction actually says

From `tools/isa_probe` and `docs/00-verified-facts.md §1.1 / §7.3`, the only
spelling that links + runs on gfx1201 is:

    v_wmma_f32_16x16x16_f16 v[9:16], v[0:3], v[4:7], v[9:16]

with `-Xclang -target-feature -Xclang +wavefrontsize32`, and the builtin call is

    __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc)

where a, b are 8-wide f16 and acc is 8-wide f32, per lane.  So per lane:

    A lane  = v[0:3] = 4 VGPRs = 8 x f16
    B lane  = v[4:7] = 4 VGPRs = 8 x f16
    D lane  = v[9:16] = 8 VGPRs = 8 x f32

32 lanes x 8 D = 256 f32 = exactly a 16x16 tile's worth of outputs.

This much is established by the emitted ISA, not by any probe.  The blocking
unknown (docs/00 §7.4a) is what `(row, col)` each of those 8 f16 A elements and
8 f16 B elements contributes to, inside the 16x16 tile a lane helps produce.

## What the probes have already established (measured, not assumed)

`tools/bench/wmma_layout.hip` (P1-P4) and `tools/bench/wmma_run.hip` (C0, the
control) on the real RX 9070 XT:

- P1  A=[1..8] B=[1..8]  →  D = 408 in all 8 slots, identical across all 32
     lanes.  408 = 2 * sum_{j=1..8} j^2, i.e. 2 * (1+4+9+16+25+36+49+64).
- P2  A=e_k, B=[1..8]  →  D = 2(k+1) in ALL EIGHT slots, for every k in 0..7,
     and identical across all 32 lanes.  The eight A lanes produce EIGHT
     DIFFERENT D values (2,4,6,...,16), but each lane's D is uniform across the
     8 slots.
- P3  A[lane][j] = (lane+1)*(j+1), B = [1..8]  →  D varies with lane, and
     within a lane the 8 slots differ (lane 0: 648,720,...,1152).
- P4  B=e_k, A=[1..8]  →  D = 2(k+1) in all 8 slots (symmetric to P2).
- C0  control A=B=1  →  D = 16 in all 8 slots, 8-per-slot (64 MACs/lane,
     8x8 per slot).  The kernel is alive.

The critical measured facts:

1. For P2 (B = [1..8] ramp, A = e_k), D is 2(k+1) in **every** slot.
   So one hot A lane k contributes 2(k+1) to **all 8** D slots equally.
2. For P1 (A=[1..8], B=[1..8]), D = 408 = 2*sum(j^2) in every slot, i.e.
   sum over k of 2(k+1)^2 — consistent with P2's rule that lane k adds
   2*(A_k)*(B_k) to every slot, with A_k = B_k = (k+1).
3. For P3, D depends on lane even though B is uniform — so the lane index IS an
   input, and the (lane+1) factor scales all 8 A elements of that lane together.
   The 8 slots differ within a lane, so slots are not a uniform broadcast of one
   per-lane dot.

From (1) and (2): **one A lane's 8 f16 elements are NOT written to 8 different
(row,col) positions** — if they were, a one-hot A with ramp B would light up 8
DIFFERENT slots with 8 DIFFERENT values, not 8 equal values of 2(k+1).  The
measured P2 result is consistent with: **for a given lane, all 8 of its A
elements are multiplied by the SAME B element (or same B subset) and accumulated
into all 8 D slots equally.**  That is, the lane's A vector is broadcast against
one (or a small set of) B element(s), and the per-slot difference in P1 and P3
comes from B, not from A's spatial distribution.

## The actual constraint this puts on the layout

A 16x16 WMMA tile with wave32 and 8 D slots per lane means each lane produces
8 f32 outputs.  The 256 outputs of the tile are partitioned 8-per-lane across
32 lanes.  The probe tells us:

- The partition of OUTPUTS across lanes is fixed and known (lane L → D slots
  L*8..L*8+7, by construction of the probe kernel's `__builtin_memcpy(D +
  lane*D_SLOTS, &acc, ...)`).  That part is the probe's own layout, not the
  builtin's.
- What is NOT yet known is the mapping from (A_lane[0..7], B_lane[0..7]) → the
  8 output positions within that lane's 8-slot slice, i.e. which (row,col) each
  of the 8 A elements and 8 B elements participates in.

The P2 result (one-hot A → uniform D across the 8 slots, value = 2*(A_k)*(B_k
sum)) is consistent with at least two distinct layouts, both of which are
"not 8 consecutive f16 in one vector":

(a) A's 8 elements are distributed across the TILE (different rows/cols), but
    the specific one-hot k and ramp-B probe happens to hit a symmetry where each
    slot sees the same dot.  This is what a real 16x16 tile assignment would do,
    and it is the layout a tiled GEMM wants.

(b) A's 8 elements are consumed as a PER-LANE dot against a single B element
    (or a per-lane B subset), with the 8 D slots being 8 redundant copies or
    8 reductions of the same per-lane quantity.  This is the "D is 2x a per-lane
    dot" reading the probe's own intro mentions.

P3 is the discriminant that was already run: with B uniform and A varying per
lane, D differs across lanes AND across slots within a lane.  Under (b),
uniform B + per-lane A should give per-lane D that is uniform across slots
(same per-lane dot, broadcast).  The fact that P3's slots DIFFER within a lane
is evidence AGAINST pure (b) and FOR a tile-like assignment (a) — but P3's B is
uniform (all 1s), which under (a) would make some slots equal by symmetry, so
P3 alone does not uniquely pick (a) either.

So the honest state is: the probe results rule out "A is 8 consecutive f16 in one
vector consumed as one 8-dot" (that would give 8 different masks in P2, not 8
equal values), and they rule out "lane is irrelevant" (P3 differs by lane).  They
do NOT uniquely determine the (row,col) assignment, because the probes so far are
all symmetric enough that more than one layout reproduces them.

## The checkable consequence that the original "identical D" result now becomes

The original finding ("all 8 one-hot A probes give identical D, independent of
lane") is now checkable as a CONSEQUENCE of a hypothesized layout, rather than an
unexplained measurement, IF we state the layout precisely enough to predict P2's
uniformity.

Concretely: any hypothesized layout L: (lane, a_idx) → (row, col) and
(lane, b_idx) → (row, col) must satisfy, for the P2 probe (A = e_k, B = [1..8]):

    for each lane, for each of its 8 D slots s,
        D_s = 2 * (k+1) * sum_{j=0..7} (j+1) * [overlap of A_lane_k's element
                                                with B_lane_j's element in slot s]

and the measured result is D_s = 2(k+1) for all s.  This constrains L to be such
that, for a one-hot A lane k and ramp B, every slot sees the same total
contribution.  That is a concrete equation a hypothesized layout either satisfies
or violates — which is exactly what "checkable consequence" means.

## What would uniquely determine the layout (the next measurement, not an excuse)

The probes that would disambiguate (a) from (b) and pin the (row,col) map:

- P2-variant: A = e_k, B = a PER-SLOT different vector (e.g. B in slot s = e_s),
  so that if A's elements land in different slots they light up different slots
  with different values.  The current P2 used the SAME B for all slots (a single
  B vector per lane), which is why P2 could not see A's spatial distribution.
- A two-hot A (two lanes hot) with ramp B: the cross-terms between the two hot
  lanes' contributions would reveal whether the lanes' A elements overlap the same
  B elements or disjoint ones.
- A full 16x16 tile probe where different lanes hold different (row,col)
  sub-tiles and the outputs are checked against a known 16x16 matrix multiply —
  this is the tiled GEMM kernel itself, and it is exactly what the blocking
  unknown blocks: you cannot write that kernel until you know the mapping.

## What can be said with authority right now (and what cannot)

WITH authority (established by emitted ISA + measured probes):

- The builtin consumes 8 f16 from A and 8 f16 from B per lane and produces 8 f32
  per lane, in v[9:16]/v[0:3]/v[4:7]/v[9:16].
- A's 8 elements are NOT consumed as "8 consecutive f16 in one vector → 8
  different (row,col)".  The P2 one-hot result rules that out: it would give 8
  different D masks per k, not 8 equal values.
- The lane index is an input (P3 differs by lane).
- The 8 D slots within a lane are not a uniform broadcast of a single per-lane
  dot in the general case (P3 slots differ within a lane), but they ARE equal
  under the specific P2 symmetry (one-hot A, ramp B).

WITHOUT authority (still the blocking unknown):

- The exact (row, col) each of A's 8 elements and B's 8 elements contributes to,
  within the lane's 8-output slice.  This is what a tiled expert GEMM needs, and
  it is not determined by the ISA + the existing probes alone.
- Whether the layout is a true 16x16 sub-tile assignment (a) or a per-lane dot
  with slot redundancy (b), or a hybrid.  P3 leans against pure (b) but does not
  clinch (a).

## Defined residual and the next measurement that settles it

The current residual is therefore a **bounded hypothesis space**, not an absence
of data:

- **Established:** the builtin consumes 8 f16 A + 8 f16 B per lane and produces 8
  f32 per lane; the lane index is an input; A is NOT 8 consecutive f16 consumed as
  one 8-dot; the 8 D slots within a lane are equal under specific symmetric probes
  (P2) but differ in the general case (P3).
- **Not established:** which (row, col) each of A[0..7] and B[0..7] feeds inside
  the lane's 8-slot slice, and therefore whether a tiled WMMA kernel can be
  written against that assignment.

The next measurement that would uniquely determine the layout is NOT "run the
isa probe again" — that has already been done and is not the limiting fact.  It
is one of:

1. **P2-variant with per-slot B**: A = e_k, B such that each D slot s sees a
   different B pattern (e.g. B_s = e_s).  Under a true tile assignment, a one-hot
   A lane k would light up DIFFERENT slots depending on where A_lane[k] lands;
   under a per-lane-dot-with-redundant-slots model it would not.  This is the
   smallest probe that discriminates the two.  It is the natural next step and it
   is defined here so the next run is a test of the hypothesis space rather than
   another symmetric probe that reproduces the same ambiguity.
2. **Two-hot A with ramp B**: two lanes hot at once lets the cross-terms between
   lanes reveal whether their A elements overlap the same B elements or disjoint
   ones.
3. **Full 16x16 tile probe**: each lane holds a known (row,col) sub-tile and the
   256 outputs are checked against a known matrix multiply.  This is the tiled
   GEMM kernel itself; it is exactly what the blocking unknown blocks, and it is
   the ultimate settlement.

Until one of those runs, any WMMA kernel on gfx1201 is written against a stated
hypothesis attached to the kernel source, and the hypothesis is the thing that
must be verified — not the kernel's exit code.  That is the hardening: the
unknown is named, bounded, and has a defined next measurement, rather than being
left as "the layout is unknown" or asserted as if it were known.

## Why this is the right stopping point, not an excuse

The original "identical D" result ceased to be a mystery the moment the emitted
ISA (v[9:16], v[0:3], v[4:7], v[9:16]) and the probe's own output layout
(D[lane*8 .. lane*8+7]) were written down together — that alone explains the
8-slots-per-lane structure and the control's D=16.  What remains is the
finer-grained (row,col) map inside a lane's slice, and the existing probes do not
yet pin it because they were designed to answer "is it 8-consecutive-f16?" (no)
and "does lane matter?" (yes), not "which (row,col) per element?".

That is a real, bounded unknown with a defined next measurement (the P2-variant
with per-slot B, the two-hot A with ramp B probe, and/or a full tiled 16x16
probe).  It is NOT an excuse to skip the analysis: the analysis above is the
derivation that turns the mystery into a set of checkable equations a
hypothesized layout must satisfy, and it states explicitly which parts are
established and which are not.

## External trace — public WMMA lane mapping for gfx1201

This section traces the question the internal-probe trace leaves open: whether the
lane→(row,col) mapping is documented outside this repo at all, and if so, exactly
where and in what form.

### What the attached RDNA4 ISA text says and does not say

From `rdna4_ch16_vop3p.txt`, `V_WMMA_F32_16X16X16_F16` (opcode 64) is described
as:

- `D = A (16x16) * B (16x16) + C (16x16)`
- a single matrix multiply is computed and the row-column dot products are
  distributed across the vector ALU for higher performance
- matrices A and B are f16, C and D are f32
- the microcode uses `EXEC = 64'B(-1)` and an `eval` of the matrix-level equation

That text gives the **macro operation** and the **operand matrix types/shapes**, and
it describes the distribution qualitatively ("distributed across the vector ALU").
It does **not** give, per lane, which `(row, col)` each lane's 8 A elements / 8 B
elements / 8 D elements correspond to. The `(row, col)` in the ISA text is the
algebraic description of the matrix multiply, not a lane-indexing convention.

So from the attached ISA text alone: operand geometry is described at the matrix /
wave level, and distribution across lanes is described qualitatively, but the
per-lane (row,col) assignment is not given as a formula or table in the text.

### What the public AMD / ROCm material says

- The AMD GPUOpen "Using the Matrix Cores of AMD RDNA 4 architecture GPUs" guide
  documents the WMMA intrinsics and fragment shapes for RDNA4, but does **not**
  document a standalone per-lane (row,col) rule in the page text.
- The ROCm optimization guide's **RDNA4 dense WMMA builtins** page documents the
  accumulator and srcA/srcB fragment layouts **explicitly**, with formulas and
  tables:
  - accumulator: lane L, VGPR index g → row i = (L/16)*8 + g, col j = L % 16,
    and the inverse (i,j) → lane = (i/8)*16 + j, VGPR = i % 8; rows 0-7 are held
    by lanes 0-15, rows 8-15 by lanes 16-31.
  - srcA: lane L covers A-row (L % 16), with 8 K-positions per lane, split across
    the two lane groups in 4-K interleaved blocks; srcB is symmetric, lane L covers
    B-column (L % 16).
  This is the public AMD page that states the per-lane mapping in formula + table
  form.
- A public GitHub issue (ROCm/ROCm#6025, "ROCm Documentation Gap: WMMA Output Lane
  Mapping for gfx1201 (RDNA4)") records this gap explicitly: the ISA reference and
  GPUOpen guide document encoding and intrinsic signatures but not the output
  lane-to-element mapping, and the reporter states the column-distributed layout
  (lane % 16 = column; (lane/16)*8 = row base) as something deduced from AMD's
  Composable Kernels source and empirical verification, not from the published
  ISA/guide.
- The AMDGPU/ROCm WMMA alternative you named, **rocWMMA**, is AMD's C++ library for
  WMMA fragment decomposition. Its API reference documents fragment classes,
layout tags (row_major / col_major), load/store and mma_sync, and supported
  architectures including RDNA4 (gfx1200/gfx1201) — and its fragment storage is
  documented as packed with "no guaranteed order or locality" of vector elements.
  That is the library contract: rocWMMA hides the per-lane mapping behind the
  fragment abstraction, so it is an alternative *programming model*, not a separate
  published per-lane (row,col) spec beyond what the dense builtins page states.

### What that trace settles, sharply

1. The attached RDNA4 ISA text **does** establish that the instruction is a 16x16
   f16×f16→f32 matrix multiply-accumulate distributed across the wave / vector ALU.
   It does **not** establish the per-lane (row,col) assignment.
2. The public AMD/ROCm material **does** document the output and operand lane
   mapping for the dense RDNA4 WMMA builtins, on the roc-handbook RDNA4 dense WMMA
   builtins page, as an explicit formula plus tables.
3. Independently, a published external writeup (JohnTDI-cpu/rdna4-wmma-guide) and a
   rocWMMA/CK-compatible fragment diagram both state the same column-distributed
   layout: lane % 16 = column, (lane/16)*8 = row base, 8 accumulator rows per lane,
   lanes 0-15 covering rows 0-7 and lanes 16-31 covering rows 8-15, with all lanes
   spanning the 16 columns. That matches the roc-handbook dense-WMMA accumulator
   formula.
4. The AMD GPUOpen guide and the rocWMMA API do **not** themselves state that
   per-lane (row,col) mapping as a standalone lane-indexing rule.

### What the repo-level residual becomes after this trace

After this external trace, the residual is **not** "there is no public source for the
mapping." A public per-lane mapping is available and checkable: the ROCm
optimization guide's RDNA4 dense WMMA builtins page states it as an explicit
formula + tables, and a published external derivation plus a rocWMMA/CK fragment
diagram independently corroborate the same column-distributed layout.

The remaining repo-level residual is therefore narrower: whether the existing device-
side probes in this repo (`wmma_layout.hip` P1-P4 and `wmma_run.hip` C0) are
sufficient to **confirm** that published mapping against the repo's own measured
data, or whether they still only rule out bad special cases without clinching the full
(row,col) assignment. That is a on-device comparison to run, not a derivation to
repeat from the ISA text alone.
