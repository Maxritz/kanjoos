# 05 — Speculative decoding

On this engine speculation is not a throughput feature bolted on at the end. It
is a **bandwidth** feature: the decode step's bottleneck is often expert transfer,
and drafting several tokens per verification pass amortises that transfer across
more than one token. A drafter that is wrong costs nothing but the transfer we
were going to pay anyway.

That framing drives every decision below.

---

## 1. Order of implementation, and why

### 1.1 MTP first — because it is already resident

The current loader reads `nextn_predict_layers` and subtracts it from `n_layer`.
The MTP head is therefore **in VRAM right now and never used**. Turning an
already-resident head into a drafter costs zero extra I/O, which on a workload
whose bottleneck is compulsory expert reads is the only free acceleration
available. Everything else is a second file and a second 0.6–2.5 GiB of reads.

```cpp
struct MtpDrafter {
    // head lives in the target's own weight space; no separate file
    DraftTokens draft(const HiddenState& h, int n_steps);
};
```

* The head is a small number of extra blocks at the end of the model file. It
  is resident because the loader keeps it resident.
* Drafting n tokens costs n forward passes through **only the head's blocks**,
  not the full model. That is the entire reason it is free and the reason it
  comes first.
* Verification runs the full model once over the n candidates. If the model
  genuinely benefits from speculation, n·(head cost) + full pass < n·full passes.

### 1.2 One loader for DFlash / DFlash2 / DSpark

All three arrive with `arch = dflash`, a `dflash.*` metadata block, and `blk.N.*`
tensors. **One loader path serves all three.** Implementing them one at a time
is three times the work for one code path.

```cpp
struct DflashSpec {
    std::string decoder_arch;      // must match the target's decoder family
    int block_count;               // dense blocks in the drafter
    int block_size;                // candidates per pass
    std::vector<int> target_layers;// layers re-run for verification
    int embedding_dim, n_head, n_head_kv, head_dim;
};

class DflashDrafter {
public:
    bool compatible_with(const TargetArch&) const;   // decoder_arch must match
    DraftTree draft(const HiddenRing&, int width);
};
```

* **Compatibility check is mandatory.** A drafter whose `decoder_arch` does not
  match the target's decoder is refused at load with a clear message. The Laguna
  head's tensors *look* compatible (same embedding dim, same head count, same
  head_dim, an attention output gate, no experts) but that is an inference from
  the tensors, not a licence to run: match the declared decoder arch.
* **The cross-attention ring** is the interesting part: the drafter reads the
  target's intermediate hidden states through a small ring, so its forward pass
  is cheap relative to a 30B target, and it needs no separate copy of the
  target's weights.

### 1.3 DSpark's confidence head, wired into the prefetcher

DSpark is a semi-autoregressive parallel backbone plus a serial Markov correction
plus a **confidence head that estimates each draft token's survival
probability**. For this engine that last part is the valuable one, because it is
the signal the scheduler needs:

```
if (drafted tokens > 0 and next verification pass would need cold experts
        and confidence of token i < threshold)
    stop drafting at i
```

In other words: **do not draft tokens you will verify slowly.** On a warm cache
with idle bandwidth, draft deep. On a cold cache where the next verification
pass is going to stall on NVMe anyway, draft shallow and let the CPU fallback
carry the tail. This is the one place where speculation and residency control
should talk to each other, and it is worth building even if the drafter itself
is stock.

---

## 2. Verification

Losslessness is a requirement, not a hope. Every speculation path must produce
**bit-identical output to non-speculative decoding under the same seed**.

```
draft  ->  verify (target forward over all candidates)
       ->  accept the longest prefix whose target distribution agrees
       ->  resample the first divergent position from the target's own logits
       ->  emit
```

* Greedy verification: accept while `argmax(target) == candidate`.
* Sampling verification: accept while the candidate falls inside the target's
  probability mass, resampling from the residual — the standard scheme, and the
  only one that keeps I6 true under stochastic sampling.
* The verifier is the **target's own forward pass** with a batch of candidates,
  not a second model. That is what makes verification cheap enough to be worth
  it.

**Budget interaction:** a verification pass over n candidates computes n rows of
the batch. Its expert demand is the **union** of the drafted rows' routing, which
is larger than a single row's. On a bandwidth-bound engine that union can be the
whole reason speculation backfires — more experts touched per accepted token
than the naive estimate. The profiler (C21) therefore reports
`experts touched per accepted token` alongside the acceptance rate, and the
scheduler reduces draft width when that ratio rises.

---

## 3. Where drafters live

Drafters are just another streamed object: NVMe → RAM → VRAM, with the same
residency manager (C9), the same transfer engine (C11), the same eviction policy
(C12). A drafter block that is not resident is not a special case.

But there is one asymmetry worth encoding: **the drafter is more valuable warm
than the target's experts are warm**, because it is on the critical path of every
token, and its own size is small (0.6–2.5 GiB). So the warm tier's admission
policy gives drafter blocks a bonus when they would otherwise be evicted. This is
a one-line change in C12's score and it is worth the one line.

---

## 4. Configuration

| method | drafter file cost | extra I/O per token | when to enable |
|---|---|---|---|
| MTP | 0 (already resident) | ~0 | always on, when the model has `nextn` layers |
| DFlash | 0.6–1.8 GiB | amortised over the block size | after MTP, when acceptance is measured and the drafter fits the profile |
| DFlash2 | 0.7–1.1 GiB | same | same; more candidates per pass at slightly more latency |
| DSpark | 1.4–2.5 GiB | same | same; adds the confidence head, which is the reason to prefer it when transfer-bound |

Default: MTP on. The others are opt-in and must earn their place on a measured
acceptance rate and a measured tok/s delta, not on a published headline.

---

## 5. Acceptance criteria

| criterion | threshold |
|---|---|
| losslessness | bit-identical to non-speculative under fixed seed, greedy and sampled |
| MTP overhead | zero additional H2D bytes at load |
| drafter I/O amortised | drafter H2D bytes per accepted token below the configured bound |
| expert-union awareness | reported `experts touched / accepted token`, and draft width adapts when it rises |
| cold-cache behaviour | on a deliberately cold cache, shallow drafting is chosen and the CPU fallback counter rises rather than stall time |
| confidence coupling | with the confidence gate on, acceptance-adjusted throughput improves or is neutral versus the same configuration with it off |

Every one of these is a profiler assertion, not a comment in a benchmark script:
the engine asserts them itself when `--profiling=full` is on, so a regression
fails the run rather than waiting to be noticed.