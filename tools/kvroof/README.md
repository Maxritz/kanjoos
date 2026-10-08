# tools/kvroof — the KV roofline, derived from the model's own config

Every KV number in this repo used to rest on an *assumed* head configuration.
This tool removes the assumption by reading `config.json`, and it **refuses to
produce a number** when a field it needs is missing rather than substituting a
default.

```
python tools/kvroof/kv_roofline.py models/qwen3-30b-a3b/config.json
python tools/kvroof/kv_roofline.py models/*/config.json --json
python tools/kvroof/kv_roofline.py models/qwen3-30b-a3b/config.json --check
```

| file | role |
|---|---|
| `kvroof.py` | geometry resolution + traffic derivations. Pure functions, no I/O, no printing, no exit codes. |
| `kv_roofline.py` | the CLI: report rendering, the doc-claim audit, exit codes. |

## What it prints

**A** the geometry, with the provenance of each field — including whether
`head_dim` was read or derived as `hidden_size / num_attention_heads`.
**B** KV bytes per token for six codecs, metadata included.
**C** weight bytes per generated token, broken down, including the LM head.
**D** the crossover context per codec, expert-only and with all traffic.
**E** which resource dominates at 2 K / 8 K / 16 K / 32 K / 128 K.
**F** the attention roofline: FLOP/byte against machine balance.
**G, G.1** Class A ceilings, and the KV-budget-vs-expert-slot trade.
**H** Class B, priced at the measured PCIe range.
**I** with `--check`: an audit of the numeric claims in `docs/09` §2 and §5.

## Exit codes

| code | meaning |
|---|---|
| 0 | report produced |
| 2 | a required field is missing and cannot be derived from present fields |
| 3 | config unreadable or not valid JSON |
| 4 | architecture not exactly computable (MLA, hybrid attention). Output only with `--allow-foreign-arch`, and it is **not** a roofline |
| 5 | `--check` found a doc claim this config contradicts |
| 6 | a numeric argument was missing or non-positive |

Exit 4 is the interesting one. The Edge0-8B-A1B config on this machine carries
`kv_lora_rank`, hybrid KDA/full attention and `layer_group_size`; a tool that
produced a confident number for it would be wrong, so this one stops.

## Measured vs derived

Nothing here touches the GPU. The machine constants in `kvroof.py` (VRAM
589–598 GB/s, PCIe 13.4–14.7 GB/s, 39.3 TFLOP/s) are the **measured** values
from `00-verified-facts.md` §8.1–8.2 and are cited to that section in the source.
Everything else is arithmetic on a real config file. Both kinds are labelled
separately wherever they meet.