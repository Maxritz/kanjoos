# Kanjoos — MoE inference from disk on consumer AMD GPUs

Kanjoos is an out-of-core **Mixture-of-Experts (MoE) GGUF inference engine**.
The product is not "load a huge model somehow". The product is:

> A bounded-working-set MoE inference engine in which the model can be much
> larger than VRAM, cold expert weights stay on NVMe, useful experts are
> promoted through RAM into VRAM, routing drives demand and prefetch, and the
> runtime proves through telemetry and full-output correctness that the
> resulting system is correct and faster than naive streaming.

Reference model: **Qwen3-30B-A3B** (`models/qwen3-30b-a3b/config.json`) —
48 layers, 128 experts/layer, top-8 routing, hidden 2048, 32 heads / 4 KV
heads × head_dim 128, vocab 151936, bf16.

**Targets:** **RDNA4 `gfx1201` is the primary target** — and, on this machine,
the only one that can execute. RDNA2 `gfx1031` (RX 6700 XT, second machine) is
a *declared* second target: tier-B census PASS, tier-C runs guarded and passing,
and three throughput pins formally judged inside tolerance
(`records/gfx1031-tierC-enforced-2026-10-11.log`) — never the default, and
rocWMMA/gfx12 drivers correctly refuse it (no WMMA on RDNA2). Both are planned
for **Windows 11** and **Linux**, with ROCm 10.1.
**VRAM profiles:** 6 / 8 / 12 / 16 GiB. **RAM profiles:** 16 / 24 / 32 / 48 /
64 / 96 GiB.

Every number below carries a provenance: **MEASURED** (ran on hardware, command
recorded), **DERIVED** (computed from measured inputs), or **PLAN** (design
intent, not yet built). Nothing here is a wish presented as a result.

---

## Why this exists

A 30B-parameter MoE at 4-bit quantization is ~14 GiB of expert weights alone —
more than a consumer card holds once the KV cache, activations, and the dense
(non-expert) weights take their share. The standard answers are: buy a bigger
card, quantize harder until quality breaks, or offload naively and accept
single-digit tokens/second.

Kanjoos is built on a different observation, measured on the target machine:

- **Expert weights are fixed (~14.13 GiB at W4 g128) and cacheable** —
  routing locality is exploitable: keep useful experts resident, stream the rest.
- **KV cache is unbounded (`tokens × bytes_per_token`) and uncacheable** —
  100% miss every token, forever.

So the governing reframe ([docs/09-kv-engine-architecture.md](docs/09-kv-engine-architecture.md)):

> **KV residency is the product feature. Weight residency is an optimisation.**

Two consequences fall out of the measured transfer budget. One W4 g128 expert
is **2.47 MB**: **7.68 µs** of arithmetic at peak versus **~181 µs** of PCIe
transfer at measured small-transfer bandwidth (MEASURED). Expert movement
dominates everything — so the optimisation chain is always *less unnecessary
expert movement → better prediction → better residency → fewer PCIe misses →
more overlap → lower token latency*, and tiny-GEMM work is secondary until
movement stops dominating. And at W4 the resident set is 15.31 GiB with a
**negative** KV budget on a 16 GiB card, while W3 g128 keeps 126/128 experts
resident with 3.46 GiB of KV headroom (37.8 K FP16 tokens) — so **the weight
format is a capacity decision, not a bandwidth one** (DERIVED, audited by
`tools/kvroof/kv_roofline.py --check`).

---

## Why it is different

Compared against llama.cpp-style offload, vLLM-class server engines, and
vendor stacks, Kanjoos differs in five deliberate ways:

1. **Residency is the interface, not an accident.** A three-tier model —
   NVMe (capacity) → RAM (staging) → VRAM (working set) — with routing-driven
   demand and prefetch, coarse-grained I/O (coalesced multi-expert extents, a
   layer slab — never one tiny random read per expert), and explicit
   accounting: every routed contribution executes, waits at a named dependency,
   falls back to CPU, or is **explicitly refused**. `expert unavailable → skip`
   is prohibited because it silently changes model semantics. No mainstream
   engine makes that guarantee part of its contract.
2. **Correctness is full-output and bit-graded.** The default is
   full-output verification against a host oracle with max abs error, relative
   RMSE, zero/unwritten-output and mismatch-index reporting — because a
   strided sample plus a passing exit code has passed nonsense here before.
   Tier movement is bit-for-bit (invariant I7, canonical reduction order);
   speculation is bit-identical under a fixed seed (I6); a codec that changes
   values is a *declared precision reduction*, never conflated with a move.
3. **Benchmarks refuse to lie.** Every device driver carries an arch guard:
   unset `KNJ_BUILD_ARCH` → exit 7, mismatch → exit 6 `SKIPPED`, because an
   arch-mismatched binary runs and lies. A tier that did not run is *reported*,
   never omitted; exit statuses survive filters (`pipefail`); assertions are
   never weakened to make a check pass. The 24-figure machine audit
   (`python tools/check_docs.py`) keeps the docs honest against the tools.
4. **AMD-first, RDNA4-native compute.** gfx1201's matrix path is rocWMMA 2.2.1
   (measured decision: its gfx12 path calls exactly the needed builtins, so no
   hand-rolled operand layout), while gfx1031's basis is packed-integer dots
   (`v_dot8_i32_i4` consumes W4 nibbles directly, no unpack). The arch split
   lives only in `src/device/`, `kernels/`, `src/platform/`, `cmake/arch.cmake`.
5. **A loader that decodes what the ecosystem actually ships.** 26 GGUF
   quant types decode to f32 bit-exact against their references — the 20
   upstream types (F32/F16/BF16, Q4_0–Q8_1, Q1_0, TQ1_0/TQ2_0, all six K-quants,
   nine IQ types, MXFP4/NVFP4) plus the six fork-experimental ROCmFP4/X types
   from `charlie12345/ROCmFPX` (ids 100–107, validated against the fork's own
   sources since gguf-py has no codecs for them). The Q3_K hmask-first layout
   bug was caught here by triple-checking, not by a user.

What it deliberately is **not**: no training, no multi-GPU, no NVMe-as-VRAM,
no "unlimited context" promises the transfer budget cannot keep, no
general-purpose tiering engine before the MoE case works. (The full non-goals
list with reasoning is [docs/MISSING-ITEMS.md](docs/MISSING-ITEMS.md) §4.)

---

## Status: what runs today (MEASURED)

| area | state | evidence |
|---|---|---|
| Host runtime (`knj_runtime` + `kanjoos-run`) | builds; loader, BPE tokenizer, qwen3moe forward pass, qwen35 front end, profiler, platform layer | `ctest` 8 pass / 1 skipped-by-design (9 tests), exit 0 |
| GGUF decoders, 26 types | bit-exact vs references | `tools/dequant_validate.py` (20/20) + `tools/rocmfp_validate.py` (36/36), both exit 0 |
| qwen35 oracle decoders | 11 types cross-checked identical vs gguf-py on Saluki tensor bytes (F32/Q8_0/Q2_K/Q4_K + IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS) | `python tools/ref_qwen35.py <Saluki> --layer 0` → 11/11 identical, exit 0; ThinkingCap regression (F32/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K) RC=0, 20 vectors |
| qwen35 layer-0 probe vs oracle | 20/20 vectors element-by-element, 0 failed (embeds/xnorm bit-exact, qkv ≤1.53e-05) | `kanjoos-run --qwen35-ref C:/tmp/saluki_ref_new`, probe exit 3 = designed trunk refusal |
| Tokenizer parity vs llama.cpp | 5031/5031 identical, 0 merge-boundary divergences, 36555 tokens per seed | `records/tok-fuzz-5000/summary.txt` (see note on the 3-seed run) |
| Tokenizer build time | ~52 ms `unordered_map` → ~9 ms open-addressing `TokViewMap`, identical ids | `tmp/bench_tokmap.cpp` + fuzz gate |
| Attention prefill (gfx1201) | 9.6 ms → 5.0 ms via QT 8→16 + loop-bottom LDS-race barrier, oracle error 1.38e-04 unchanged | `tools/bench/attn_c16.hip`, pin `PREFILL_US 5500` |
| Bench gate | `run_bench.sh` exit 0 (tier A host, tier B compile census, tier C guarded runs) | `tools/bench/baselines.txt` |
| gfx1031 tier B + tier C (MACX, RX 6700 XT) | census PASS (5 OK, 9 correct rocWMMA refusals); 5 drivers ran and passed; **3/3 pins formally judged and inside tolerance** | `records/gfx1031-tierC-enforced-2026-10-11.log`, `KnjTierC3` exit 0 |
| c21 profiler gate | 30/30 after fixing a Windows early-sleep flake at the stimulus (assertion untouched) | `tests/unit/test_c21_profiler.cpp` GATE 1 |
| Full MoE inference | PLAN — the engine serves, prefills, and profiles; end-to-end generation against the oracle is the open milestone | [docs/08-roadmap.md](docs/08-roadmap.md) P0–P7 |

> **Note on the 3-seed fuzz run (2026-10-11, honest accounting).** The first
> 15,000-prompt campaign (3 seeds × 5000) returned VERDICT: PASS with 0
> merge-boundary divergences, but all three seeds drove the cross-check with
> the *same* `--fuzz-seed 1` — the three logs are byte-identical, so its true
> coverage was one 5032-string series, not three. The harness defect was fixed
> at the cause (`tools/tok_fuzz_campaign.py` now derives a per-seed fuzz seed;
> the summary line records which mode ran), and a true 3-series re-run is in
> flight on MACX. The PASS stands for what was compared; it did not test the
> tokenizer three independent ways. See `docs/CODING-LOG.PENDING.md` Phase 62.

---

## Repository map

```
CMakeLists.txt        top-level build (KNJ_ARCH selects the target arch)
kanjoos.toml          runtime configuration — starting values only, never truth
docs/                 the design evidence: 00…09, COMPONENT-REFERENCE, MISSING-ITEMS
ai-coder/             implementation worksheets c1…c24 — CONTRACTS, not suggestions
kernels/              device code headers + knj_kernels.hip
src/                  the runtime; src/device/ is the only DEVICE C++, the rest
                      (loader, tokenizer, model, profiler, platform, util, cli)
                      builds into knj_runtime + kanjoos-run
tools/bench/          run_bench.sh (tier A/B/C gate) + device bench drivers
tools/ref_qwen3moe.py the numpy oracle for the reference model's forward pass
tools/ref_qwen35.py   the numpy oracle + dequant cross-check for the qwen35 front end
tools/dequant_validate.py   the 20 upstream decoders vs gguf-py, byte for byte
tools/rocmfp_validate.py    the 6 ROCmFP decoders vs the fork's own sources
tools/tok_crosscheck.py     tokenizer parity vs llama.cpp (hand set, --corpus,
                            --fuzz, --separator-sweep over 14 whitespace codepoints)
tools/{ci,doctor,i7,isa_probe,kvroof,route}/   the machine-audit tools
models/               qwen3-30b-a3b config + sha256 provenance
records/              raw measured output, kept verbatim
tests/                host-side tests (profiler, device-contract, residency, tokenizer)
cmake/                arch.cmake and friends
build/                build output (ignored)
```

## The plan (design documents)

| document | what it settles |
|---|---|
| [00 — Verified facts](docs/00-verified-facts.md) | What was measured on this machine, with commands. Includes corrections to older notes. |
| [01 — System architecture](docs/01-architecture.md) | Tier model, budget arithmetic, invariants, per-token control flow. |
| [02 — Component specification](docs/02-components.md) | C1–C24, each with interfaces, state machines, invariants, failure modes, acceptance tests. |
| [03 — Kernel strategy per target](docs/03-kernels.md) | gfx1201 WMMA path, gfx1031 SIMT path, tile geometry, validation tolerances. |
| [04 — Memory tiering and residency](docs/04-memory-tiering.md) | ReBAR, VRAM/RAM profile tables, NVMe I/O shape, prefetch, eviction, KV residency. |
| [05 — Speculative decoding](docs/05-speculation.md) | MTP first, one loader for DFlash/DFlash2/DSpark, lossless verification. |
| [06 — Profiling and telemetry](docs/06-profiling.md) | The `--profiling` output contract, dev/idle/host attribution, the instrumentation floor. |
| [07 — Build and platforms](docs/07-build-platforms.md) | ROCm 10.1, per-target flags, Windows/Linux specifics, kernel packs, CI, `kanjoos doctor`. |
| [08 — Roadmap](docs/08-roadmap.md) | Phases P0–P7 with gates, whole-engine acceptance criteria, risk register. |
| [09 — Tiered KV engine architecture](docs/09-kv-engine-architecture.md) | Why KV residency is the product, the context-class ladder, the gap check. |

Working rules for this repo live in [AGENTS.md](AGENTS.md) — invariants I1–I7,
the layering rule, the toolchain spellings that work, and the mistakes already
paid for. `docs/CODING-LOG.PENDING.md` is the append-only build diary.

## Tooling already present

| tool | what it is |
|---|---|
| [`tools/isa_probe/isa_probe.hip`](tools/isa_probe/isa_probe.hip) | compile-only capability probe that decides what any kernel may use. Run before writing kernels, and in CI after a ROCm upgrade. |
| [`tools/kvroof/kv_roofline.py`](tools/kvroof/kv_roofline.py) | derives the KV roofline and Class A/B ceilings **from a model's own `config.json`**. `--check` audits the numbers quoted in `09`. |
| [`tools/route/route_locality.py`](tools/route/route_locality.py) | measures expert-selection survival across tokens from real router weights, converted to ms/token at measured PCIe bandwidth. |
| [`tools/i7/i7_bit_identity.py`](tools/i7/i7_bit_identity.py) | the I7 test: resident, movement-codec and tier-split paths must agree **bitwise**; three negative controls must fail. |
| [`tools/kpack_report.md`](tools/kpack_report.md) | what each `.kpack` on this machine provides, and which the engine uses. |
| [`kanjoos.toml`](kanjoos.toml) | shipped default configuration (C22). Every value is a starting point the autotuner overrides. |

```bash
python tools/check_docs.py                    # the 24-figure machine audit, expect exit 0
python tools/kvroof/kv_roofline.py models/qwen3-30b-a3b/config.json --check
bash tools/bench/run_bench.sh                 # tier A/B/C gate, expect exit 0
ctest --test-dir build/cmake-host --output-on-failure   # host suite
python tools/dequant_validate.py --probe <abspath>/tmp/dequant_bytes_probe.exe
python tools/rocmfp_validate.py  --probe <abspath>/tmp/dequant_bytes_probe.exe \
                                 --ref <abspath>/tmp/attnrep/rocmfp_ref.exe
```

(Probe paths must be absolute — a relative `tmp/...` path fails in `subprocess`
on this machine. The validators prepend the compiler's `bin` to the probe's
`PATH` themselves, so no staged DLL copies are needed; CMake does the same for
`ctest` via `ENVIRONMENT_MODIFICATION`.)

## The findings that shape the design

1. **gfx1201 has a working matrix path; gfx1031 never will.** Emitted ISA
   shows `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` producing a real
   `v_wmma_f32_16x16x16_f16`. RDNA2 is refused on every spelling — no matrix
   units, which is hardware. Integer dots (`v_dot4_i32_i8`, `v_dot8_i32_i4`,
   `v_dot2c_f32_f16`) are portable inline asm on both arches, so W4 on the 6700
   XT runs through `v_dot8_i32_i4` with **no unpack step at all**. The probe
   itself carries no HIP headers and preflights the compiler first. See
   [00 — Verified facts §0–1](docs/00-verified-facts.md).
2. **MoE-from-disk is a transfer-scheduling problem.** Qwen3-30B-A3B at W4
   streams ~0.93 GB of expert weight per token with zero reuse: ~4 tok/s
   against measured NVMe, ~20 against PCIe. RAM residency and VRAM expert slots
   beat any kernel micro-optimisation; NVMe is a capacity tier, never bandwidth.
3. **Routing locality is weak, so residency must be earned, not assumed.**
   59–81% of a token's experts are new each step (P0-7, MEASURED) — prediction
   and prefetch are load-bearing, and evicting experts to buy KV context costs
   3–9 ms/token for less capacity than a W4→W3 repack frees.
4. **KV codec choice follows hardware, not fashion.** FP8 KV on gfx1201
   (native FP8), FP16 on gfx1031 (no FP8), 4-bit schemes behind a quality gate —
   per the community evidence surveyed in the design docs, aggressive
   throughput-halving codecs don't survive the transfer-budget arithmetic.

## Profiling

```
kanjoos serve --profiling
```

prints a fixed-column table of per-component `ops / %dev / dev us / idle us /
host us`, plus the measured instrumentation floor. Full contract is in
[06 — Profiling](docs/06-profiling.md). `--tokenize-only` opens GGUF metadata
without a full `Model::open`, with per-stage (`gguf-open`, `tok-vocab`,
`tok-merges`, `encode`) attribution.
