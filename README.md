# Kanjoos — MoE inference from disk on consumer AMD GPUs

A C++/HIP inference engine that runs Mixture-of-Experts models whose expert
weights do not fit in VRAM, by streaming them through a three-tier residency
system and predicting far enough ahead to hide the transfer.

**Targets:** **RDNA4 `gfx1201` is the primary target** — and, on this machine,
the only one that can execute. RDNA2 `gfx1031` is a *declared* second target and
**future work**: it is compiled and censused, never the default, and no current
result is gated on it. Both are planned for **Windows 11** and
**Linux**, with ROCm 10.1.
**VRAM profiles:** 6 / 8 / 12 / 16 GiB. **RAM profiles:** 16 / 24 / 32 / 48 /
64 / 96 GiB.

---

## The plan

| document | what it settles |
|---|---|
| [00 — Verified facts](docs/00-verified-facts.md) | What was measured on this machine, with commands. Includes corrections to existing project notes. |
| [01 — System architecture](docs/01-architecture.md) | Tier model, budget arithmetic, the transfer budget that sizes everything, invariants, per-token control flow. |
| [02 — Component specification](docs/02-components.md) | C1–C24, each with interfaces, state machines, invariants, failure modes, acceptance tests. |
| [03 — Kernel strategy per target](docs/03-kernels.md) | gfx1201 WMMA path, gfx1031 SIMT path, tile geometry, the validation matrix. |
| [04 — Memory tiering and residency](docs/04-memory-tiering.md) | ReBAR, VRAM/RAM profile tables, NVMe I/O shape, prefetch, eviction, KV residency. |
| [05 — Speculative decoding](docs/05-speculation.md) | MTP first, one loader for DFlash/DFlash2/DSpark, lossless verification. |
| [06 — Profiling and telemetry](docs/06-profiling.md) | The `--profiling` output contract, how dev/idle/host time is measured, the instrumentation floor. |
| [07 — Build and platforms](docs/07-build-platforms.md) | ROCm 10.1, per-target flags, Windows/Linux specifics, kernel packs, CI, `kanjoos doctor`. |
| [08 — Roadmap](docs/08-roadmap.md) | Phases P0–P7 with gates, whole-engine acceptance criteria, risk register. |
| [09 — Tiered KV engine architecture](docs/09-kv-engine-architecture.md) | Why KV residency is the product and weights are the optimisation, the context-class ladder, the component flowchart, and the gap check against the tree that exists. |

## Tooling and configuration already present

| file | what it is |
|---|---|
| [`tools/isa_probe/isa_probe.hip`](tools/isa_probe/isa_probe.hip) | the compile-only capability probe that decides what any kernel is allowed to use. Run it before writing kernels, and in CI after a ROCm upgrade. |
| [`tools/kvroof/kv_roofline.py`](tools/kvroof/kv_roofline.py) | derives the KV roofline, the crossover context and the Class A/B ceilings **from a model's own `config.json`**. Exits non-zero rather than assuming a missing field. `--check` audits the numbers quoted in `09`. |
| [`tools/route/route_locality.py`](tools/route/route_locality.py) | P0-7. Measures how much of a token's expert selection survives to the next token, from real router weights, and converts the misses into ms/token at the measured PCIe bandwidth. |
| [`tools/i7/i7_bit_identity.py`](tools/i7/i7_bit_identity.py) | the I7 test: resident, movement-codec and tier-split paths must agree **bitwise**; three negative controls must fail. Written before any codec exists. |
| [`models/`](models/README.md) | five real `config.json` files with sha256 provenance, including the target. |
| [`tools/kpack_report.md`](tools/kpack_report.md) | what each `.kpack` on this machine provides, and which of them the engine uses (blas — as the measured reference; the rest deliberately not). |
| [`kanjoos.toml`](kanjoos.toml) | the shipped default configuration described by C22. Every value is a starting point the autotuner overrides. |

```bash
tools/isa_probe/run_isa_probe.sh                     # auto-detects lib/llvm/bin/clang.exe
tools/isa_probe/run_isa_probe.sh /opt/rocm/llvm/bin/clang gfx1201
echo $?                                              # 0 = measured, 3 = toolchain broken

python tools/kvroof/kv_roofline.py models/qwen3-30b-a3b/config.json --check
echo $?                                              # 0 = doc agrees, 5 = doc contradicts the config
python tools/i7/i7_bit_identity.py
echo $?                                              # 0 = I7 holds, 10 = a path diverged
```

Do not point it at `hipcc`: the `hipcc.exe` on this machine hardcodes a clang
path inside a deleted ROCm tree and fails with an empty error. Pass a
`lib/llvm/bin/clang.exe` instead.

The probe's measured output, and what it implies for the kernel plan, is in
[00 — Verified facts §1](docs/00-verified-facts.md).

---

## The findings that shape the design

1. **gfx1201 has a working matrix path; gfx1031 never will.** Measured from
   emitted ISA: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` with
   `-Xclang -target-feature -Xclang +wavefrontsize32` produces a real
   `v_wmma_f32_16x16x16_f16`. The `wmma-256b-insts` flag this plan previously
   hunted for does not exist in this build and is not needed. RDNA2 is refused on
   every spelling because it has no matrix units — that one is hardware.

   A second, larger reversal: **integer dot products are portable across both
   arches as inline asm** (`v_dot4_i32_i8`, `v_dot8_i32_i4`,
   `v_dot2c_f32_f16`). The earlier "two `#if`-selected headers" policy existed
   only because a broken toolchain made the asm look unavailable. RDNA2's
   `sdot4` builtin additionally lowers to `v_dot4c_i32_i8`, a *native* dot —
   so W4 on the 6700 XT runs through `v_dot8_i32_i4` with **no unpack step at
   all**.

   The lesson is in the probe itself: it no longer includes a single HIP header,
   and it preflights the compiler before printing anything. See
   [00 — Verified facts §0](docs/00-verified-facts.md).

2. **MoE-from-disk is a transfer-scheduling problem.** Qwen3-30B-A3B at W4 is
   ~0.93 GB of expert weight per generated token with zero reuse. Against the
   measured ~4 GB/s NVMe that is a ~4 tok/s ceiling; against PCIe it is ~20.
   RAM residency and VRAM expert slots are therefore worth more than any kernel
   micro-optimisation, and NVMe is a capacity tier, never a bandwidth tier.

3. **The weight format is a capacity decision, not a bandwidth one.** P0-7
   measured routing locality and found it weak — 59-81% of a token's experts are
   new, and evicting experts to buy KV context costs 3-9 ms/token for less
   capacity than a single W4→W3 repack frees. The decision is W3 g128: the only
   pack whose FP16 context (37.8 K) fits under the model's 40,960-token ceiling
   while keeping essentially the whole expert bank resident. At W4 the resident
   set is 15.31 GiB and the KV budget on a 16 GiB card is negative. Full
   derivation in `00-verified-facts.md` §9.8 and `tools/kvroof/kv_roofline.py`
   section K (audited by `--check`).

4. **TurboQuant's own community study says don't use it by default.** FP8 keeps
   2× capacity at full throughput; TurboQuant's aggressive variants cost
   40–52% throughput and lose accuracy on reasoning and long context. So: FP8 KV
   on gfx1201 (which has native FP8), FP16 on gfx1031 (which does not), and
   TurboQuant 4bit-nc behind a quality gate.

## Profiling

```
kanjoos serve --profiling
```

prints a fixed-column table of per-component `ops / %dev / dev us / idle us /
host us`, plus the measured instrumentation floor that makes the small numbers
interpretable. Full contract, including how `idle us` is attributed per stream
and why storage waits are never folded into a generic idle number, is in
[06 — Profiling](docs/06-profiling.md).