# Sherlock-style trace — full-scale MoE engine, rq3-30b-a3b baseline shape

**What this is:** the expected per-component trace shape for a resident decode step
on the target model, written against the profiler contract in
`docs/06-profiling.md` and the component taxonomy in `docs/02-components.md`
C16/C17/C21. The numbers are **illustrative shape, not a measured run**, because
no `kanjoos` binary exists yet. The point is the table structure, the column
meanings, the floor line, and the two device-time consumers the engine is built
around.

**Model:** Qwen3-30B-A3B, from `models/qwen3-30b-a3b/config.json`
(48 layers, 32 attention heads, 4 KV heads, head_dim 128, 128 experts, top-8,
moe_intermediate_size 768, hidden_size 2048, vocab 151936, max_position_embeddings
40960, bf16).

**Geometry used to size the categories:** from `tools/kvroof/kv_roofline.py
--json` on that config:
- KV bytes/token FP16: **98,304 B/token** (96.0 KiB)
- expert-only weight bytes/token at M=1: **948 MB/token**
- total weight bytes/token incl. LM head: **1,586 MB/token**
- resident weight bytes at W4 G128: **15.31 GiB**
- KV budget at 16 GiB VRAM: **negative** (the model leaves no room for FP16 KV
  at W4 on this card)

That is the scene the trace is diagnosing: a tiering engine whose whole job is to
keep hot experts resident while the rest stays on disk, on a model that at W4
leaves zero room for long KV on 16 GiB.

---

## Why this table exists

`docs/06-profiling.md` §10: the table separates GPU busy from GPU idle from CPU
time, prints its own floor, and names components rather than kernels. The shape
that matters is:

- **attention dominant in `dev us`** is expected and good — it is where decode
  spends its device time (C16).
- **transfer with a large `idle us` and a larger `host us`** is the signature of
  a cold expert or cold KV page that the scheduler failed to land in time
  (C9/C11/C19). That is the failure the profiler exists to make visible.
- **everything else rounding noise** is the correctly balanced resident case.

---

## Reference table shape (illustrative)

```
stdout:

  kanjoos 0.1.0   device AMD Radeon RX 9070 XT (gfx1201, 32 CU)   rebar: full aperture
                 vram 14.00/16.00 GiB   ram 62.4/96.0 GiB warm   nvme 4.1 GB/s   pinned 6.0 GiB
                 kv codec fp8   expert format w4   slots 211   profile 12g/96g
                 steps 24   tokens 512   draft 1.31x   accept 0.71   wall 41.882 s   tok/s 12.22

     component           ops   %dev      dev us     idle us     host us
     attention          2,400  95.4%  535880.80      377.62       49.20
     expert-gemm        3,840   3.2%   17723.32     2026.19      460.80
     expert-gemm-unpack 3,840   1.1%    6127.59     1170.02      143.80
     router              48     0.2%    1335.28     1102.27      111.60
     transfer             12     0.0%      79.80     1087.10     1084.30
     kv-write            48     0.0%      72.98      784.24      139.40
     kv-read             48     0.0%      23.40      198.44       73.30
     norms              144     0.0%      21.52      372.84       67.70
     residual            96     0.0%       0.48       25.44        2.40
     head+sample          1     0.0%      98.20       91.80       56.90
     bias                 18     0.0%      90.03      265.52       66.60
   instrumentation floor, 256 empty ops timed through the same begin/end path: 0.70 us host, 19.54 us device each.

  residency   vram 91.4%   ram 8.1%   nvme 0.5%   slots 211/211   evictions 47   prefetch hit 0.82   stall 0.00 ms
  transfer    h2d 4.19 GB   d2h 0.02 GB   nvme read 0.31 GB   nvme write 0.00 GB   peak pinned 5.8 GiB
  cold store  quota 1.4 TiB   used 812 GiB   write bytes/token 0
```

**Column meanings are normative from `06-profiling.md` §3:**
- `ops`: timed ops attributed to the component.
- `%dev`: that component's `dev us` ÷ total `dev us`.
- `dev us`: GPU busy time on the same stream, start event → end event, summed.
- `idle us`: GPU idle attributable to the component on the critical stream,
  bounded by enqueue time, per §4.
- `host us`: CPU time inside the component's submit path, RAII begin → end,
  excluding blocking waits.

`dev us` never includes idle. `idle us` is never folded into `dev us`. That
separation is the whole point: a component can be 95% of device time and a
different component can be 95% of wall time, and the table is supposed to tell
those apart.

---

## How to read this specific table

### Attention is the device-time center of gravity

`attention` at ~95% of `dev us` is the expected resident-case shape. C16 is the
attention kernels; C14/C15 are the paged KV and codecs underneath them. In a
resident step the attention kernel is reading already-resident KV, so the dominant
device time is compute/latency on the tile path, not a transfer. If that
dominance drops and `transfer` rises, the story is no longer "attention is slow";
it is "the KV page did not arrive".

### Expert GEMM and its unpack are the other device-time half

C17 is the MoE expert GEMM. The table intentionally splits it into
`expert-gemm` and `expert-gemm-unpack` for exactly the reason `06` §5.1 states:
on some targets unpack is real work, and a fused number would hide a pack-format
problem. If `expert-gemm-unpack` is more than a few percent of `expert-gemm`, the
pack format is the thing to fix, not the GEMM scheduling.

### Router is small but it is the trigger for everything else

C18 router top-8 selection is cheap in device time and must be exact and unbiased.
Its job in the trace is mostly as the upstream event that creates the demand set
for C9/C10/C11. A router row that is large in `host us` or `idle us` usually means
the submit path or the top-k work is on the critical CPU thread, not that routing
is expensive.

### Transfer is the row you read when the step is slow

`transfer` should be small in a resident step. When it is not, the diagnosis is
not "the kernel is bad"; it is one of:
- a cold expert miss that C10/C11 did not prefetch in time,
- a cold KV page in a Class B / storage-assisted session,
- a pinned-pool or NVMe path that is the actual bottleneck.

The residency/transfer/cold-store footer rows are there so the transfer row can be
read against bytes moved, not just microseconds.

---

## What the floor line is for

`docs/06-profiling.md` §6: the floor is measured every run by running N empty ops
through the identical begin/end path. It is reported and, by default, subtracted
from every component, so `dev us` reads as device work rather than as measurement
overhead. A component at or below the floor is reported but marked, so a tiny
`residual` row is not silently rounded into a false claim.

The floor line is part of the contract. A profiler that reports numbers without
reporting its own overhead is making a claim it cannot support.

---

## The fail case this trace is meant to expose

The worst case for this engine is not "attention slow". It is:
- `transfer` with real `dev us`,
- a large `idle us` on the consumer,
- a non-zero `STALL_TIME_MS` in the residency block,
- and the residency label saying `storage-bound` or `RAM-resident` instead of
  `VRAM-resident`.

That is the shape `06` §9.3 says must be visible, because it is the shape that
distinguishes an I/O problem from a kernel problem. An engine that cannot produce
that row when it is cold is an engine that will misdiagnose itself.

---

## What "both backends" means here

The component table is backend-agnostic by design: `attention`, `expert-gemm`,
`expert-gemm-unpack`, `router`, `transfer`, and the footer residency rows are the
same categories on gfx1201 and gfx1031. What changes between backends is the
underlying math and therefore the expected `dev us` balance:

- gfx1201: C16 can use WMMA f16/f8 and C17 can use WMMA i8 or dequant+WMMA;
  the attention row is the device-time heavyweight.
- gfx1031: C16 is `v_pk_fma_f16` SIMT and C17 W4 is `v_dot8_i32_i4` with no
  unpack; the relative weight of `expert-gemm-unpack` falls to zero by design on
  the W4 path, and attention remains dominant.

So the same table shape is the right output contract on both backends; the
backend shows up in the numbers and in the capability header, not in the
component names.

---

## Open gap this report is explicitly not filling

`06-profiling.md` §9 acceptance criterion 3 is the one that cannot be satisfied
by a document: a deliberately cold run must produce the transfer/idle/stall
signature. That is a measurement, and it is the proof that the profiler can see
the failure it exists to find. Until there is a binary that can be made cold on
demand, this report is a shape specification, not a validated trace.
