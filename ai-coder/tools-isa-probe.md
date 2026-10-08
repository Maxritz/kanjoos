# Worksheet — tools/isa_probe (existing)

**What it is:** compile-time AMDGPU ISA probe. `bash tools/isa_probe/run_isa_probe.sh`
(exit 0 = measured, exit 3 = toolchain broken). It probes `sdot4`, `sdot2`, the inline-asm
dot variants, and the WMMA builtins on both gfx1031 and gfx1201 by compiling to `.s` and
reading the emitted instruction back out.

**What it proves (measured, `00` §1):**
* gfx1031 `sdot4` builtin → `v_dot4c_i32_i8` (a single hardware dot-4, not a shift/add
  expansion) — `00` §1.2.
* gfx1201 WMMA f16 + bf16 via the `_gfx12` builtin + `+wavefrontsize32` → `v_wmma_f32_16x16x16_f16`
  / `_bf16` — `00` §1.1.
* gfx1201 `sdot4` builtin is REFUSED (needs `dot1-insts`); the gfx11 `_w32` WMMA spelling is
  REFUSED on both arches; the gfx11 `_w32` spelling with 16×f16 operands CRASHES the backend
  (Branch relaxation, access violation) — `00` §1.1.
* Inline-asm dot variants (`v_dot4_i32_i8`, `v_dot8_i32_i4`, `v_dot2c_f32_f16`, and more)
  assemble cleanly on **both** arches — `00` §1.3. So dot products can be written once in asm.
* The AMDGPU assembler is **not** gated by `-offload-arch` — dot availability comes from the ISA
  reference, never from the assembler — `00` §7.2. (This is why the probe reads the emitted
  instruction back out: "OK" means the instruction is really in the output, not merely "the
  compiler returned 0".)

**Why it matters to a coder:** this is the source of truth for the cap set C2 reflects, and for
the "write dot products as inline asm once" rule. Any component that touches a dot or WMMA
instruction must cite the probe's verdict, not a datasheet and not "the compiler accepted it".

**Standing rule (verbatim, `00` §0):** a capability claim must name the emitted instruction. An
exit code is not evidence. The probe's four cell values are `OK` (instruction in output),
`REFUSED` (feature gate), `CRASH` (backend fault), `EMPTY` (compiles with exit 0 but emits a
code object with no kernel — `amdhsa.kernels: []`; `00` §7.1). A coder extending the probe must
preserve all four and must not collapse `EMPTY`/`CRASH`/`REFUSED` into "not OK".

**What a coder must not break:**
* The probe includes **no HIP headers** (`00` §0) — a capability probe must not depend on the
  toolchain it is testing; it needs only the AMDGPU frontend + backend. Do not add `<hip/hip_runtime.h>`
  back in.
* The runner **preflights a trivial kernel first** (`00` §0) — if a kernel with no ISA features does
  not compile, the runner exits 3 and prints *no matrix at all*. This is the guard against the "empty
  string because the path was not found" lie that produced the entirely-false matrix in an earlier
  revision.
* The runner distinguishes `REFUSED`/`CRASH`/`OK <insn>` and now `EMPTY` (`00` §7.1). Do not
  collapse these.
* The probe reads the emitted instruction back out of the `.s` and prints it in the cell (`00` §0).
  "OK" must mean the instruction is really there.

**Reproducibility:** this turn, `bash tools/isa_probe/run_isa_probe.sh` exits 0 and reproduces the
matrix: gfx1031 first refusal (sdot4 on RDNA2 needs `dot1-insts`? no — that is gfx1201; the gfx1031
sdot4 builtin produces `v_dot4c_i32_i8`), gfx1201 first refusal (`__builtin_amdgcn_sdot4` needs
`dot1-insts`), and the first emitted instructions (gfx1031 `v_dot4c_i32_i8`, gfx1201 `v_pk_add_f16`).
The WMMA builtins on both arches are FRONTEND FEATURE GATE on the gfx11 `_w32` spelling. This matches
`00` §1.

**Files a coder should read before touching this:**
* `tools/isa_probe/run_isa_probe.sh` and the probe source it invokes.
* `docs/00-verified-facts.md` §0, §1, §7.1, §7.2.
* `docs/02-components.md` — C2, C3 (the components that consume the probe's verdicts).
