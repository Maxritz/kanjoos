# Coding log

Append-only. New phases are appended; earlier entries are not rewritten. A
correction is a new entry naming the entry it corrects.

Format and rules: [`.agents/skills/dox/SKILL.md`](../.agents/skills/dox/SKILL.md).

---

## ⚠ RECOVERY NOTICE — 2026-10-07 17:01 — THIS FILE IS A PARTIAL RECONSTRUCTION

**This file was destroyed by an agent error and has been rebuilt from the only
copy that survived: the transcript of the session that clobbered it.**

* **What happened.** An editing tool was invoked with "create or replace file"
  semantics against `docs/CODING-LOG.md` instead of "replace this substring".
  The tool's own response read `String replace applied successfully`, which is
  what an append looks like. It was a full-file overwrite. The file went from
  **941 lines** to **11 bytes** (`placeholder`).
* **What was searched for a copy, and found nothing** (all negative, 2026-10-07):
  `C:` tree `find -iname 'CODING-LOG*'` (only this path and an unrelated
  `Simple-Ai` project's own log); `docs/*.bak`; editor local history
  (`%APPDATA%\Code\User\History`, Cursor, Windsurf, VSCodium); `%APPDATA%\Freebuff`
  (app data, logs) and the workspace `.freebuff/` (contains only `project-id`);
  `C:\Users\rr\OneDrive\tmp\` (155 entries, none of them this file);
  `Win32_ShadowCopy` (no shadow copies on this volume); `ai-coder.zip` (the
  worksheets only). **This checkout is not a git repository**, so there is no
  `git restore`.
* **Therefore.** Phases 1–29 are **not recoverable from local disk** and are
  **not reproduced here**. They are not summarised, paraphrased or guessed —
  inventing reconstructions of measurements is precisely what the dox rules
  forbid. What follows is only the text that survived verbatim in the session
  transcript: the tail of Phase 29 and the whole of Phase 30.
* **The authoritative copy is OneDrive's version history.** This directory is
  inside `C:\Users\rr\OneDrive\Desktop\`, and `OneDrive.exe` /
  `OneDrive.Sync.Service.exe` are running, so the pre-17:01 content of this file
  exists in the cloud. Restore it from the OneDrive web UI
  (Kanjoos → `docs/CODING-LOG.md` → Version history) or Explorer
  (right-click → *Version history*). **A OneDrive restore supersedes this file
  entirely — do not merge the two.** This reconstruction exists only so that the
  repo is not left with a missing log in the meantime.
* **If the OneDrive restore is not possible**, treat the gap as a real gap:
  re-derive anything from Phases 1–29 that a current decision depends on, and
  record the re-derivation as a new phase naming what it re-derives.

---

## ⛔ GAP — Phases 1 through 29 (before and including most of Phase 29)

**Reconstructed: no. Recoverable from local disk: no.**
941-line file; ~870 lines of it lived here. Every MEASURED number, every
rejected alternative and every falsification recorded in those phases is gone
from this copy. Known from the surviving Phase 29 tail (below) and from
`records/` + `docs/`: the W4/W3 pack work, the rocWMMA matrix backend decision,
the kernel throughput series (19.12 TFLOP/s = 39% of the 48.7 TFLOP/s RDNA4
dense FP16 class peak), the `_gfx12` operand-layout closure, the CU-count
discrepancy, and the routing-trace work all happened in this range.

See the RECOVERY NOTICE above for the OneDrive restore.

---

### Phase 29 — <name lost with the gap>  ·  <status lost with the gap>

**…body lost with the gap…**

**The decode row is the finding, and it is not the one the earlier plan assumed.** M=1 and M=8 cost the *same* (83 µs vs 86 µs). The loss is not "1/16 of the WMMA tile is useful" — it is that at block tile `BM=64` both shapes land in a single block whose cooperative 64-row x K panel load runs in full regardless of how many rows carry data. The lever is grouping across **tokens**, not a different instruction. The driver now prints this instead of asserting a 16x tile-efficiency loss it does not measure.

**Still open — and this is the user's actual gate, still unmet.**
- **Kanjoos has never generated a token.** There is no GGUF loader, no tokenizer, no sampler loop, no KV cache bound to real weights. Every number in the MEASURED table above is either a *kernel* result against a host oracle or an *external* reference implementation. "Inferences correctly", "is coherent" and "good prefill and token gen" are all still OPEN and must not be read out of this table.
- `gemma-4-26B-A4B` is a valid *size* target (13.26 GiB, A4B, spills on consumer VRAM) but is **not a valid coherence oracle on this machine** right now — the harness runs it into repetition. A build that runs Gemma4 correctly must be found before it can be used to judge coherence.
- The measured 19.12 TFLOP/s is 39% of class peak; the 40.8 TFLOP/s MXFP4 reference says 53% is reachable. Unimplemented: 128x128 block tiles, LDS double buffering, removing bounds checks from the inner K loop.
- The 32-vs-64 CU discrepancy is still unexplained and still not load-bearing (it is only printed).

---
### Phase 30 — the CPU reference path inferences coherently; the FFN residual was missing in all 28 layers  ·  DONE

**Believed at the time**
- constraint: I1 (no expert is lost) and the doctrine that *full-output correctness against a host oracle is the default* — so a bisection that names the first tensor where two independent implementations part company is evidence, and "this line looks wrong" is not.
- baseline: Phase 29 closed with "Kanjoos has never generated a token" in its own **Still open**. The user's completion bar is *"it inferences correctly and is coherent and has good speeds of prefill and token gen"*. A forward pass existed and produced text, but the text was not coherent, and the engine's own numbers (threads, speed) were partly assertions.

**Decision**
- chose: bisect against an **independent** implementation (a numpy forward pass written from gguf-py's dequantizers) with the engine dumping 40 named intermediates, rather than reading the C++ and reasoning about which line looked wrong. A disagreement then names the tensor, the index and the value, which is a fact; a reading is an opinion.
- chose: fix the arithmetic defect rather than widen a tolerance, and keep every check at the strictest setting that the f32 association order allows. A tolerance that has to move to accept a change means the change is wrong.
- rejected: comparing only the final logits and a sampled stride. A strided sample plus a passing exit code is consistent with nonsense, and it was nonsense.
- rejected: keeping `dequant_dot_f32`'s documented "one pass" contract as prose while the body staged a 256-float buffer. Either the header or the body had to move; the body did.
- falsified by: a rerun of `tools/hidden_diff.py` that stops reporting `AGREE`, or a `tools/fused_dot_crosscheck.py` run that reports a tensor over tolerance.

**Changed**
- `src/model/model.cpp`, `src/model/model.h` — **the defect.** `moe()` begins by zeroing its destination (`std::fill(out, out + m*H, 0)`), and `forward()` handed it `x_` — the residual stream. So each layer computed `x = moe_out` instead of llama.cpp's `cur = moe_out + ffn_inp`, and the residual was destroyed in **all 28 layers**. Added a dedicated `ffn_out_` buffer: `moe(w, hx_.data(), n, ffn_out_.data())` then `x_[i] += ffn_out_[i]`. The comment on the declaration records why the buffer exists, because the next person to "simplify" it back into `x_` re-introduces the bug silently.
- `src/model/model.cpp` — `dump_f32()` and `KNJ_DUMP_HIDDEN`: 40 named intermediates (`00_embed` … `12_logits`, including `10_after_layer00..27`). Gated on `const bool dbg = (pos_ == 0)`, mirroring the oracle's `_forward_calls == 1` gate — without it a decode call overwrites the prefill's dumps with 1-row tensors and the diff silently compares the wrong thing (that happened; the ref dumps were 24576 B where the engine's were 20480 B).
- `src/util/parallel.h`, `src/model/model.cpp`, `src/cli/main.cpp` — `ThreadPool::total()` and `Model::threads()`. `matmul` guarded on `pool_->size()`, which counts only spawned workers and **excludes the calling thread**, and `ThreadPool(2)` spawns 1 worker — so a 2-thread pool reported 1 and took the serial path, making `--threads 2` exactly as slow as `--threads 1`. The CLI also printed `threads : 0`; it prints the real count now.
- `src/loader/dequant.cpp` — genuinely fused `dot_q4_k` / `dot_q6_k`, dispatched from `dequant_dot_f32` for the 256-weight K-quants. Both apply the sub-block scale to the *sum* (`sum(w·x) = d·sum(q·x) − dmin·sum(x)`), which is what ggml's own vec-dot does; the staged form was the odd one out.
- `src/model/model.cpp`, `src/model/model.h` — per-call heap allocations hoisted into open()-time scratch: `head_out_` (was `std::vector<float> tmp(D)` inside the per-token loop, i.e. 28 allocations per token), `norm_stage_`, `norm_w_`, `col_tmp_`, `probs_`, `chosen_`; and `matmul`'s per-work-partition `std::vector<float> buf(bw)` is now a stack array. **Measured effect: none** — see the measurements table. Kept because it removes a few hundred allocator round-trips per token at no cost in clarity, and *not* recorded as a speedup.
- `tools/ref_qwen3moe.py` — new. An independent numpy forward pass for `qwen3moe` on gguf-py: NEOX RoPE, per-head q/k RMSNorm, top-2-of-4 softmax router with renormalised weights, `cur = moe_out + ffn_inp`.
- `tools/hidden_diff.py` — new, then rewritten. Gate is `|ref-eng| <= atol + rtol*max|ref|`; the stricter per-element count is still printed so a real error cannot hide inside a tensor-scale tolerance. Hard-fails (never silently passes) on a tensor the engine did not write, an all-zero engine buffer where ref is non-zero, NaN/Inf, and any `12_logits` argmax disagreement. Prints top-k overlap.
- `tools/fused_dot_probe.cpp`, `tools/fused_dot_crosscheck.py` — new. The existing `dequant_crosscheck.py` only exercises `dequant_row_f32`, so the fused dot the mat-vec **actually calls** was covered by nothing. The probe calls the engine entry point on real tensors with integer-valued x (cancellation-free), and the cross-check recomputes the same dots from gguf-py's dequantizer with numpy.
- `tools/dequant_probe.cpp` usage documented as `dequant_probe <model.gguf> <outdir> <tensor>...`; `dequant_crosscheck.py` needs both positional arguments (`<model> <dumpdir> <tensor>...`) and raises `IndexError` without them.
- `records/2026-10-07_host_inference_bringup.txt` — new, verbatim command output for everything below.

**Verified**
- `$ bash tools/build_engine.sh` → exit `0`, `build/host/kanjoos-run` 321312 bytes.
- `$ KNJ_DUMP_HIDDEN=build/eng2 ./build/host/kanjoos-run --model <qwen3moe-q4_k_m> --ctx 512 -p "The capital of France is" -n 0 --threads 24` → exit `0`.
- `$ python tools/hidden_diff.py build/ref build/eng2` → exit `0` → `AGREE: every dumped tensor matches the independent oracle within tolerance`, all 40 tensors, including `09b_moe_out` and all 28 `10_after_layer*`, after the allocation hoisting.
- `$ python tools/hidden_diff.py build/ref build/eng2 | tail` → `argmax: ref=12095 (logit +16.3321)  eng=12095 (logit +16.3321)  AGREE`; `top-10 overlap: 10/10`.
- `$ ./build/host/fused_dot_probe <model> build/fdot blk.0.attn_q.weight blk.0.attn_v.weight blk.0.ffn_gate_exps.weight blk.0.ffn_down_exps.weight blk.1.attn_k.weight` → exit `0`.
- `$ python tools/fused_dot_crosscheck.py <model> build/fdot <same five tensors>` → exit `0` → `AGREE` on Q4_K **and** Q6_K, max abs error 1.1e-5 over ~1.1e7 dot products.
- `$ ./build/host/dequant_probe <model> build/deq <3 tensors>` + `python tools/dequant_crosscheck.py …` → exit `0`, `exact == n` on all three (bit-identical to gguf-py).
- `$ python tools/check_docs.py` → exit `0` → `all 24 documented figures match the tools`.
- `$ tasklist | grep -i llama` → no `llama-cli.exe` / `llama-bench.exe` / `llama-completion.exe` running.

**Measurements**
| quantity | value | provenance |
|---|---|---|
| `Qwen3-MoE-4x0.6B` Q4_K_M, greedy 16 tokens, `--temp 0` | ` Paris. The capital of Italy is Rome. The capital of Spain is Madrid.` | MEASURED |
| same prompt, llama.cpp reference (`-ngl 0`) | byte-identical 16-token completion | MEASURED |
| engine decode, `--threads 24`, 16 tokens | 10.50 tok/s (95.2 ms/token) | MEASURED |
| engine decode, `--threads 16`, 16 tokens | 11.20 tok/s (89.3 ms/token) | MEASURED |
| engine decode, `--threads 32`, 16 tokens | 9.58 tok/s (104.4 ms/token) | MEASURED |
| engine prefill, 5 tokens batched, `--threads 24` | 391.50 ms, 12.77 tok/s | MEASURED |
| llama.cpp CPU-only baseline, same host | prompt 0.34 tok/s; eval 156.28 ms/token = 6.40 tok/s | MEASURED |
| engine vs llama.cpp CPU decode | **1.6× faster** | DERIVED |
| engine vs llama.cpp CPU prefill | **~8× faster** (12.04 vs 1.70 tok/s on 5 tokens) | DERIVED |
| `--bench 64 32` spread across 16/24/32 threads | pp 8.55–10.72 tok/s, tg 8.19–10.18 tok/s | MEASURED |
| allocation hoisting, decode ms/token | 98.3–104.4 both before and after — **inside run-to-run noise** | MEASURED (negative) |
| fused vs staged `dequant_dot_f32`, per element | max abs 1.1e-5, max rel 4.0e-4, 0 elements over tolerance | MEASURED |
| `exact/n` on `12_logits` under the strict per-element tolerance | 1649 of 151936 | MEASURED |
| raw evidence | `records/2026-10-07_host_inference_bringup.txt` | MEASURED |

**The host is noisy at ±5%**, which is larger than the allocation-hoisting effect, so that change is reported as a non-regression and not as a speedup. The decode thread sweet spot moved between runs (24 in the earlier series, 16 in this one); treating a single run as an optimum would be reading noise.

**Still open — the user's bar is met on coherence and beaten on speed only against a CPU baseline**
- **The GPU path has never run the model.** Every number here is the host reference path. `gfx1201` decode is still the kernel driver at M=1, 0.0832 ms per GEMM launch against a DERIVED 154.9 tok/s expert-GEMM ceiling — no engine runs on it. This is the remaining distance to "good speeds" *on the GPU*.
- `run_bench.sh` tier B still refuses every driver on this install (`'hip/hip_runtime.h' file not found`, `unknown type name '__host__'`); `expert_gemm.hip` additionally needs `-I kernels/` for `knj_pack.h`. The per-driver hipcc build + arch-guarded run + diff is the load-bearing check in the meantime, and tier B remains a REFUSED report, not a pass.
- `tools/bench/wmma_tile_probe.hip`'s `H_FLAT` baseline uses non-integer operands, so its 256-slot match is not yet evidence; the unexplained 2× MAC factor and the P3 (row,col) progression are still open.
- The repo-wide audit for checks that can pass without measuring anything (the C4 uninitialised-oracle class) has not been done.
- `gemma-4-26B-A4B` is still not a coherence oracle on this machine (degenerate repetition), so the 13.26 GiB streaming target has no correctness evidence yet.
- Per-call allocation is not fully gone: `moe()`'s per-expert `std::memcpy` gather and `sample()`'s `std::vector<std::pair<…>>(n_vocab)` in the CLI remain.
