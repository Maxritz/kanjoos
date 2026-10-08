# tools/i7 — the I7 bit-identity test

Written **before there is a codec to violate it**, which is the only time it is
worth writing. Two programs, because they prove different things:

```
bash tools/i7/run_i7.sh          # both legs, exit 0
python tools/i7/i7_bit_identity.py   # host reference only
```

| file | what it proves |
|---|---|
| `i7_bit_identity.py` | the **contract**: what the canonical reduction order is, and that every wrong order fails. Runs anywhere. |
| `i7_hip.hip` | the **device**: that a real GPU, a real PCIe hop and a real pinned-host bank reproduce the resident path in every bit. Needs gfx1201. |

| exit | meaning |
|---|---|
| 0 | all positive paths agree bitwise and every negative control was detected |
| 10 | a path that must match diverged (reported with the first differing element and both bit patterns) |
| 11 | a negative control did **not** diverge — the test is toothless |
| 12 | the oracle or the device failed; nothing was certified |
| 6 / 7 | arch mismatch / arch guard not configured |

## Two claims, deliberately not the same one

1. **Identity is GPU-vs-GPU.** Resident, movement-codec and tier-split all call
   the same `reduce_pages()` routine on the same device, so they execute
   identical instructions on identical bits. That is the I7 claim, and it is
   bitwise.
2. **Correctness is GPU-vs-CPU, within tolerance** (measured worst relative
   error 1.1e-05 against a double oracle). `__expf` is the native
   `v_exp_f32` and is not required to be correctly rounded, so cross-platform
   bit-identity against libm `exp` is neither asserted nor achievable.

Conflating those is how a bit-identity test ends up "passing" on a tolerance
while proving nothing.

## The paths

| | path | must match bitwise |
|---|---|---|
| A | resident — every page from T1 | yes |
| B | movement — every page round-tripped through a **lossless** storage codec and host RAM | yes |
| C | tier split — hot pages from T1, cold pages streamed from T2 | yes |

| | negative control | must diverge |
|---|---|---|
| N1 | flat single accumulator, no page partials | yes |
| N2 | cold-first merge — streamed pages merged before hot ones | yes |
| N3 | lossy codec round-trip | yes |

## The canonical order is the contract

Floating-point addition is not associative. For identity to be structural
rather than lucky, the reduction order is part of the specification:

```
two-pass:  pass 1 computes fp32 scores and the global max (max IS associative,
           so its order does not matter)
           pass 2 visits pages in ASCENDING index order,
                  keys within a page in ASCENDING order,
                  accumulating into per-page partials,
           then merges the page partials LEFT TO RIGHT, once.
```

The canonical order is also the cheap one. It is exactly what a paged attention
kernel already does, so complying costs nothing — which is the argument for
writing it down before the kernel exists rather than retrofitting it.

## What writing this changed

1. **I7 had to be re-scoped.** As originally worded — "any codec must reproduce
   the same bits" — it is unsatisfiable: a quantising codec changes values by
   construction. Correct scope: **tier movement is bit-identical** and its
   storage codec must be lossless. A codec that changes values is a **declared
   precision reduction** — priced, recorded, and never conflated with a move.
2. **The obvious implementation is not the canonical one.** N1 diverges. So the
   flat form must never be written, and the page decomposition must be enforced.

## The bug this caught in our own code

The first version of path C passed `K` and `V` whole on the hot branch while
the cold branch passed page slices — so the hot path indexed `V` with a
page-local offset. It produced output that looked entirely reasonable and was
wrong at element 0. A tolerance check would have passed it. This is the
argument for the test existing at all.