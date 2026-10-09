# C21 baseline — the one working MoE model, threads × context (2026-10-08)

**MEASURED.** Every number below came from a run recorded in this directory.
This is the first real *performance baseline* for the engine on this machine, and
the reference profile the C21 regression gate now pins.

| | |
|---|---|
| model | `C:/Users/rr/OneDrive/Desktop/kraken/models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf` (964,653,184 B, `qwen3moe`) |
| geometry | 28 layers, hidden 1024, 16/8 heads × 128, 4 experts top-2, expert_ff 3072, vocab 151936 |
| path | `kanjoos-run` — the **CPU reference path** (host clock, no device event backend) |
| machine | RX 9070 XT host, Windows 11, 32 logical CPUs (`nproc` 32) |
| bench | `--bench 64 32` (64-token prefill, 32-token decode), `--profile-warmup 4` → 92 steps kept |
| profiler | C21, `--profiling=json --profile-dir <cell>`, floor 256 empty ops |

Commands, verbatim (the matrix driver is reproducible from these):

```sh
export PATH="/c/Strawberry/c/bin:$PATH"     # MinGW runtime DLLs, else exit 127
M="C:/Users/rr/OneDrive/Desktop/kraken/models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf"
./build/cmake-host/kanjoos-run.exe --model "$M" --bench 64 32 --threads 8 --ctx 2048 \
    --profiling=json --profile-dir records/c21-baseline-2026-10-08/t08-ctx2048 --profile-warmup 4
```

Each cell holds `run.log` (stdout+stderr, the exe's own rc), and
`profile.txt` / `profile.json` / `profile.csv` from the run itself. No cell was
re-run or edited after the fact.

---

## 1. Axis 1 — threads (ctx 2048)

| threads | prefill ms | prefill tok/s | decode ms | decode ms/token | profile total µs | speedup vs 1 |
|---|---|---|---|---|---|---|
| 1 | 35 345.06 | 1.81 | 17 694.87 | 552.965 | 50 702 635.52 | 1.00× |
| 2 | 18 498.33 | 3.46 | 9 341.79 | 291.931 | 26 609 405.82 | 1.91× |
| 4 | 10 581.97 | 6.05 | 5 338.09 | 166.815 | 15 226 272.93 | 3.33× |
| **8** | **6 407.28** | **9.99** | **3 288.98** | **102.781** | **9 264 798.52** | **5.47×** |
| 16 | 5 228.81 | 12.24 | 2 614.00 | 81.687 | 7 433 994.22 | **6.82×** |
| 32 | 6 395.95 | 10.01 | 3 641.09 | 113.784 | 9 527 605.11 | 5.32× |

* **The ceiling is 6.82× on 16 threads**, and 16 is the best measured cell. That
  is 43% of linear on a 32-logical-CPU machine — the serial fraction is large.
* **32 threads is worse than 16** (pp −22%, decode −39% relative to the 16-thread
  cell; 5.32× vs 6.82×): 32 logical CPUs are 16 physical cores, so this is SMT
  oversubscription, measured rather than assumed.
* Run-to-run noise on the pinned cell, three runs: prefill 6407.28 / 6315.91 /
  6344.78 ms, decode 3288.98 / 3267.44 / 3267.16 ms → **≤1.5%**.

## 2. Axis 2 — context (8 threads)

| ctx | prefill ms | decode ms | profile total µs |
|---|---|---|---|
| 512 | 6 559.10 | 3 372.04 | 9 482 810.11 |
| 1024 | 6 412.31 | 3 331.61 | 9 300 111.01 |
| 2048 | 6 407.28 | 3 288.98 | 9 264 798.52 |
| 4096 | 6 342.74 | 3 278.50 | 9 163 783.50 |
| 8192 | 6 367.84 | 3 266.33 | 9 212 181.13 |

**Flat: 3.4% spread across a 16× change in `--ctx`.** In this path `--ctx` is an
*allocation* knob, not an arithmetic one: prefill runs 64 tokens and decode
attends over `n_past` (96), so the KV budget never binds and the attention walk
never sees more than 96 keys. The measurement is what says so; the code is not
being inferred here from the numbers, the numbers are being reported as flat.

## 3. Where the time actually goes (the pinned cell: 8 threads, ctx 2048)

| component | ops | share | self µs |
|---|---|---|---|
| moe-down | 5 152 | 18.5% | 1 718 015.74 |
| head (lm_head) | 92 | 17.8% | 1 653 002.54 |
| moe-gate | 5 152 | 16.3% | 1 511 200.34 |
| moe-up | 5 152 | 16.1% | 1 489 985.34 |
| qkv | 2 576 | 12.7% | 1 174 536.82 |
| moe-act | 5 152 | 7.5% | 690 825.24 |
| attn-o | 2 576 | 5.8% | 537 236.22 |
| attention | 2 576 | 4.8% | 443 635.62 |
| moe / moe-router / norm / norm-ffn / prefill / decode / scatter / gather / residual / embed | — | 0.6% together | ≤14 494.82 each |

Grouped:

* **expert GEMMs (down + gate + up) = 50.9%** of the run.
* **the whole `moe` path = 58.9%** (the three GEMMs, the activation, the router,
  the gather and the scatter) — and the **router is 0.1%**.
* attention block (qkv + attn-o + attention) = 23.3%.
* **`head` alone = 17.8%** — the untied `output.weight` (151936 × 1024) is read in
  full at *every* generated token, which is the largest single non-expert row.
* Everything else — both norms, the router, the gather/scatter, residual, embed —
  is **0.6%**. This is the measured basis for the AGENTS.md §11 chain: the
  micro-costs are micro.

## 4. The gate, run on the matrix

Reference pin: [`tools/c21/baseline/ref-8t-ctx2048.json`](../../tools/c21/baseline/ref-8t-ctx2048.json)
(the 8-thread cell; the reason is recorded inside the file's `pin` block).
Metric `dev_ns_self`, tolerance ±10%, `--min-us 50`.

| cell | rc | verdict |
|---|---|---|
| `t08-ctx2048-r2`, `-r3` | 0 | PROFILE OK — repeatability, 18 judged, 0 failed |
| `t08-ctx4096` | 0 | PROFILE OK |
| `t08-ctx1024` | 1 | 1 failed: `moe-gather` +13.6% (a **2.2 ms** row) |
| `t08-ctx8192` | 1 | 1 failed: `norm-ffn` +15.2% (a **3.9 ms** row) |
| `t16-ctx2048` | 1 | 7 failed — see below |
| `t32-ctx2048` | 1 | 11 failed |
| `t01-ctx2048` | 1 | 10 failed |

**The 16-thread cell is the interesting one: it is 22% faster overall and still
fails the gate 7 times.**

```
parallel rows, 16 vs 8 threads      serial rows, 16 vs 8 threads
head        -32.0%  improved        moe-gather   +44.4%  REGRESSED
moe-down    -24.2%  improved        moe-scatter  +33.2%  REGRESSED
moe-gate    -23.8%  improved        prefill      +30.4%  REGRESSED
moe-up      -23.7%  improved        decode       +26.6%  REGRESSED
attn-o      -18.0%  improved        moe          +26.3%  REGRESSED
qkv         -11.6%  improved        norm-ffn     +12.9%  REGRESSED
                                     norm         +12.5%  REGRESSED
```

The split is not noise, it is the parallel/serial boundary: at 16 threads every
matmul **gains 12–32%**, and every single-threaded scope **loses 12–44%** because
the per-op pool wake-up and join cost lands inside it. The wall clock hides this
(one number, 22% better); the component table does not. **Raising the thread
count buys GEMM time and taxes the serial glue**, and on this model the serial
glue is 0.6% of the total, which is why 16 threads still wins despite it.

At 32 threads the small *parallel* regions start to lose too (`qkv` +61.4%,
`attn-o` +35.0%) — splitting 2 576 ops over 32 contending workers costs more than
it saves — while `head`, the largest region, still gains 43.1%.

**A finding about the gate, not about the engine:** the two single-failure cells
(`moe-gather` on a 2.2 ms row, `norm-ffn` on a 3.9 ms row) show that the default
`--min-us 50` is far too low for this model — a 0.02%-of-total row can move 13%
on scheduling alone. For this baseline the meaningful threshold is ~5 000 µs
(0.05% of the run); the default is left as-is because it is a global default and
changing it to flatter one model would be exactly the kind of tolerance-tuning
AGENTS.md §4 rule 8 forbids.

## 5. What this record does NOT claim

* **Not a quality measurement.** `--bench` repeats the prompt token; the model's
  output is not evaluated.
* **Not the GPU path.** Clock domain is `host`; `dev` and `host` are the same
  interval by construction here. Residency and transfer footers correctly read
  `NOT MEASURED on this path`. The device-clock numbers are in Phase 44.
* **Not a hardware ceiling.** This is the CPU reference implementation, which
  exists so the engine has a correct anchor, not to be fast.
* No cell was selected after the fact: the matrix was defined before the runs
  (threads ∈ {1,2,4,8,16,32} at ctx 2048; ctx ∈ {512,1024,2048,4096,8192} at 8
  threads), and the pin is the 8-thread cell rather than the fastest one.
