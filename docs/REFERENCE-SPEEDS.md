# Reference speeds — what "good" looks like, from other people's measurements

Every number here is **REFERENCE**: produced elsewhere, by someone else, on other
hardware. Nothing here is a Kanjoos result. Our own MEASURED numbers live in
[records/2026-10-07_throughput_and_reference_speeds.txt](../records/2026-10-07_throughput_and_reference_speeds.txt)
and in [CODING-LOG.md](CODING-LOG.md) phase 29.

The purpose of this file is narrow: when someone says the engine is "fast" or
"slow", there has to be a published number to compare against. Anything not
sourceable is not in here.

---

## 1. The ones that decide whether the project is worth doing

These are the only references that bear on the core thesis — that keeping hot
experts resident beats re-streaming them.

| what | number | source | why it matters to us |
|---|---|---|---|
| GPT-OSS-120B, 8 GB VRAM, two-tier GPU-slot expert cache (SLRU + admission filter + pinned RAM), **steady state** | **12–14 tok/s**, hit rate ~98–100% | llama.cpp issue [#20757](https://github.com/ggml-org/llama.cpp/issues/20757) (e1n00r, Mar 2026) | The direct precedent. On the **same 8 GB card** with **no** cache (`--cpu-moe` only): **0.5–1 tok/s**. That is **~14–25×** from residency alone. |
| same, cold start / warming | 1.9–2.5 / 2.6–5.1 tok/s, hit 48–56% / 53–78% | same | Says the win is **warm-up shaped**: the first ~160 tokens are slow and then it settles. Our scheduler must be judged at steady state and be honest about the ramp. |
| MoE routing skew | **~15–20% of experts serve ~80% of tokens** | same | The asymmetry the whole design rests on. If this did not hold, no cache policy would help. |
| Qwen3-30B-A3B expert-aware SSD streaming, CPU-only, M1 16 GB | **4.7 tok/s** prototype (100 tok, ~2 GB RAM); compiled follow-up 4.46 tok/s at 100% hit, compute ceiling 4.4 tok/s | llama.cpp discussion [#27149](https://github.com/ggml-org/llama.cpp/discussions/27149) (jerryjokesalot, Aug 2026) | The no-GPU counterpart. Confirms the *I/O* side is feasible before any GPU is involved. |
| Qwen3-30B-A3B per-layer expert traffic | 345 MB/layer (all 128 experts) → **31 MB/layer** (8 active) = **11.4×** | same | Independent confirmation of our own 0.95–1.07 GB/token zero-reuse figure for this model class. |
| expert-contiguous relayout | 1 expert = **192 pages** vs 6912 interleaved = **36×** less mmap amplification | same (compiled follow-up) | A **format** finding, not a hardware one. Our pack must be expert-contiguous for the same reason; check `kernels/knj_pack.h` against this. |
| Metal LRU expert paging, M1 Pro 16 GB | 13 tok/s | llama.cpp discussion #23324, as cited in #27149 | A second implementation reaching a similar place by a different route. |

**Read this as: the prize is real and ~15–25×, and it is won at *steady state*,
not at cold start.** Any Kanjoos speed claim that does not separate those two
regimes is not comparable to #20757 and should not be made.

---

## 2. gfx1201 / RDNA4 matrix throughput

| what | number | source | why it matters |
|---|---|---|---|
| fused MXFP4 GEMM, Radeon AI PRO R9700 (RDNA4) | **40.8 TFLOP/s** = **53%** of FP16 WMMA theoretical; 3.8× faster than separate dequant + hipBLAS for batch ≤ 32 | [JohnTDI-cpu/rdna4-wmma-guide](https://github.com/JohnTDI-cpu/rdna4-wmma-guide) | The best published RDNA4 WMMA result, and it is **fused-dequant** — exactly our W4 path. 53% is the credible target, not 100%. |
| RDNA4 FP16 WMMA FLOPS/CU | 512 → **1024** (2× RDNA3) | zolotukhin.ai | Sanity check on our per-CU math. |
| RDNA4 dense FP16 theoretical, 9070 XT class | **~48.7 TFLOP/s** | derived from the above | Our MEASURED 19.12 TFLOP/s is **39%** of this. Stated as where we are, not as a result. |
| RDNA4 sparsity / low-precision | 4:2 sparsity; INT8/INT4 throughput quadrupled vs RDNA3 | zolotukhin.ai | Explains why the INT4 (`iu4`) WMMA path is worth having. |

---

## 3. Same class of GPU, spilling a big MoE (the problem we exist to solve)

| what | number | source |
|---|---|---|
| RX 9070 XT, ROCm llama.cpp, Llama-2-7B Q4_0 | pp512 **5055** / tg128 **101** t/s | llama.cpp discussion #15021 (commit 583cb83) |
| ROCm 7.2.4 advertised, 9070 XT | prefill **4253** t/s | ROCm release notes |
| **Qwen3-Coder-30B-A3B Q4_K_M, spilling, 9070 XT** | ROCm pp **274** / gen **30.6**; Vulkan pp **529** / gen **22.6** | community reports |
| Ollama on 9070 XT | gpt-oss:20b MoE **91.9** t/s; qwen3:14b 52.2 t/s; qwen3.5:27b Q4 (~17 GB, spills) **6.3** t/s | community reports |
| Vulkan vs HIP on 9070 XT at token gen | Vulkan **5–7× slower** | llama.cpp issue #26663 |
| RTX 4090, Qwen3-30B-A3B, 57K ctx | ~**74.6** tok/s | community reports |

**The Colab-note here:** the moment a MoE *spills*, the same-class numbers fall to
**30 t/s (ROCm) / 22 t/s (Vulkan)** on a 9070 XT. That is the regime Kanjoos
targets, and the spread between 30 and 22 says the backend choice alone is worth
more than most tiling work.

---

## 4. Our own bracket on this machine (for convenience — MEASURED, see records/)

Not reference, repeated here only so the comparison is in one place.

| model | config | pp512 | tg128 |
|---|---|---|---|
| Qwen3-MoE-4x0.6B-2.4B Q4_K_M (914 MiB) | `-ngl 99` | 13817.56 | 214.82 |
| gemma-4-26B-A4B Q4_0 (13.26 GiB) | `-ngl 99` | 765.06 | **92.42** |
| gemma-4-26B-A4B Q4_0 (13.26 GiB) | `-ngl 99 --n-cpu-moe 99` | 372.13 | **19.70** |

So on the real streaming target the achievable band is **19.70 → 92.42 t/s** for
token generation. The engine has to be measured inside that band, against the
same model, on the same machine, or the comparison means nothing.

---

## 5. Caveat that must travel with section 4

`gemma-4-26B-A4B` is a valid **size** target but the HIP llama.cpp harness on
this machine (`build b0-unknown`) runs it into **degenerate repetition** while
printing Gemma4 loader warnings. It is therefore **not a coherence oracle** until
a build that runs it correctly is found. Its *speed* numbers are usable as a
bracket; its *output* is not usable as truth.

The small `Qwen3-MoE-4x0.6B` **is** coherent on this harness and is the right
first end-to-end target for that reason alone.
