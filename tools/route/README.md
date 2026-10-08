# tools/route — P0-7, routing locality

The last blocking unknown: **how much of a token's expert selection can be kept
resident?** It decides the expert-slot budget, and whether a spilled tier is
viable for experts as well as for KV.

```
# source 1: a real MoE run end to end, gates see their own hidden states
python tools/route/route_locality.py --source local \
    --model /g/VLLM-Models/Qwen1.5-MoE-A2.7B-Chat \
    --corpus docs/00-verified-facts.md --max-tokens 768

# source 2: the target model's REAL gate matrices, fed real hidden states
python tools/route/route_locality.py --source target-gates \
    --donor /path/to/qwen3-1.7b --corpus docs/00-verified-facts.md --max-tokens 768
```

| file | role |
|---|---|
| `hf_range.py` | reads individual tensors out of a remote safetensors file using HTTP Range — the 48 router matrices are 25.2 MB inside a 61 GB checkpoint, and measuring routing does not require 61 GB |
| `route_locality.py` | trace collection, locality statistics, slot-budget curves, and the byte/time consequences |
| `gate_cache_*.npz` | cached gate matrices; delete, or pass `--refetch`, to re-pull |
| `route_local.json`, `route_target-gates.json` | the measured results, per layer and aggregated |

## Two sources, on purpose

Neither is sufficient alone, and the tool labels each one in its own output.

- **Source 1 (`local`)** is **self-consistent**: real gates, real hidden states,
  real routing — but it is a *different model* (60 experts, top-4), so its
  numbers do not transfer, only its shape.
- **Source 2 (`target-gates`)** uses the target model's **real 48 gate
  matrices** but feeds them hidden states borrowed from Qwen3-1.7B (same
  family, same hidden width 2048). It is a **proxy**: not the target model's
  own forward pass, and it says so in every printed report and in the JSON.

## What is measured

1. **Popularity** — experts ever selected, routing entropy, and the share of
   selections falling in the most-used quarter of experts.
2. **Temporal locality** — the fraction of a token's top-k that survives to the
   next token, and the new-expert rate per token.
3. **Slot budget** — hit rate vs resident slots per layer under two policies:
   `static` (an oracle on measured popularity, with the least popular resident
   evicted on a miss) and `lru` (realistic).
4. **The consequence** — misses per token, converted to bytes at the W4 pack and
   then to ms/token at the **measured** 13.4–14.7 GB/s (§8.1).

## Result, in one line

Routing locality is **weak**: 59–81% of a token's experts are new, the hit-rate
curve is close to linear in slot count, and 70% of the bank is needed for a 95%
hit rate. Full working in `00-verified-facts.md` §9.6.

## Exit codes

| code | meaning |
|---|---|
| 0 | trace collected and reported |
| 2 | a required config field is absent — no guess is made |
| 3 | model files missing or unreadable |
| 5 | **no routing recorded** — the gate hooks never fired. Treated as a failure, because a silently empty trace is the classic way this goes wrong |

Exit 5 is the one worth keeping: it is the "it ran, exit 0, and the array was
empty" failure mode that this repo has been bitten by twice already.