#!/usr/bin/env python3
"""Gate: kernels/knj_q4k.h must decode REAL GGUF Q4_K/Q6_K bytes bit-for-bit.

WHY THIS IS A GATE AND NOT A STATISTIC
--------------------------------------
knj_q4k.h is a hand transcription of ggml's Q4_K / Q6_K layout, and every part of
that layout is a place to be wrong in a way that still yields plausible floats:
the 6-bit scales steal their top two bits from a neighbouring byte, the Q4_K
4-bit codes interleave two sub-blocks per byte, and Q6_K's high bits are packed
two-per-byte across a 64-byte plane that is not 2-byte aligned (a Q6_K block is
210 bytes long). None of that shows up as a shape error.

So the comparison is exact, and a *tolerance* is only reported, never accepted:
ggml itself computes `d * (float)sc * q` in fp32, and this header does the same
multiplications in the same order, so the correct result is bit-identical. Any
non-zero ULP count is a real disagreement about the decode, not rounding.

The oracle is gguf-py's `dequantize` -- a separate implementation, by the ggml
authors, reading the same file. The probe (tools/q4k_probe.cpp) links nothing but
the header under test, so a disagreement can only mean the header is wrong.

Ordering, checked rather than assumed: gguf-py returns dequantised data in
*reversed* ggml `ne` order, i.e. shape (e, n, k) with k fastest. Row-major
flattening that is exactly the ggml linear element order, which is the order the
probe walks its blocks in. Verified on this model before writing this file:
  blk.0.ffn_gate_exps.weight ne=(1024,3072,4) -> dequantize shape (4,3072,1024)

Usage:
  python tools/q4k_unpack_crosscheck.py <model.gguf> <dumpdir> [--layer N]
"""
import os
import sys

import numpy as np
import gguf
from gguf.quants import dequantize

TENSORS = {
    "gate": "blk.{L}.ffn_gate_exps.weight",
    "up": "blk.{L}.ffn_up_exps.weight",
    "down": "blk.{L}.ffn_down_exps.weight",
}

# fp32 d*sc*q performed in the same order in both implementations => the only
# correct number of differing bits is zero.
ULP_LIMIT = 0


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    gpath, dumpdir = args[0], args[1]
    layer = 0
    if "--layer" in args:
        layer = int(args[args.index("--layer") + 1])

    reader = gguf.GGUFReader(gpath)
    by_name = {t.name: t for t in reader.tensors}

    bad = 0
    total_elems = 0
    total_bad = 0
    print(f"{'role':5s} {'tensor':30s} {'qtype':6s} {'n':>10s} {'bit-identical':>14s} "
          f"{'maxabs':>11s} {'max_ulp':>8s} {'zeros(probe)':>13s}")
    for role, pat in TENSORS.items():
        name = pat.format(L=layer)
        if name not in by_name:
            print(f"{role:5s} MISSING TENSOR {name}")
            bad += 1
            continue
        t = by_name[name]
        ref = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(-1)
        path = os.path.join(dumpdir, role + ".f32")
        if not os.path.exists(path):
            print(f"{role:5s} MISSING DUMP {path}")
            bad += 1
            continue
        got = np.fromfile(path, dtype=np.float32)
        if got.size != ref.size:
            print(f"{role:5s} SIZE MISMATCH probe={got.size} gguf-py={ref.size}")
            bad += 1
            continue

        # Bit comparison, not value comparison: compare the raw f32 bit patterns,
        # so +0.0 vs -0.0 and any ULP difference are both visible.
        gb = got.view(np.uint32)
        rb = ref.view(np.uint32)
        neq = gb != rb
        n_bad = int(np.count_nonzero(neq))
        diff = np.abs(got - ref)
        maxabs = float(diff.max()) if diff.size else 0.0
        # Ordered-integer ULP distance, meaningful only when signs agree.
        same_sign = (gb ^ rb) >> 31 == 0
        ulp = np.where(same_sign, np.abs(gb.astype(np.int64) - rb.astype(np.int64)), 1 << 30)
        max_ulp = int(ulp.max()) if ulp.size else 0
        zeros = int(np.count_nonzero(got == 0.0))

        total_elems += ref.size
        total_bad += n_bad
        status = "yes" if n_bad == 0 else f"NO ({n_bad} differ)"
        print(f"{role:5s} {name:30s} {t.tensor_type.name:6s} {ref.size:10d} "
              f"{status:>14s} {maxabs:11.4e} {max_ulp:8d} {zeros:13d}")
        if n_bad:
            bad += 1
            worst = int(np.argmax(neq))
            print(f"      first mismatch idx={worst} "
                  f"probe=0x{gb[worst]:08x} ({got[worst]!r}) "
                  f"gguf-py=0x{rb[worst]:08x} ({ref[worst]!r})")
            print(f"      block={worst // 256} r={worst % 256} "
                  f"sub-block={ (worst % 256) // 32 }")
        else:
            print(f"      bit-identical on all {ref.size} weights "
                  f"({'block-aligned, non-zero' if zeros == 0 else 'ZERO-VALUED WEIGHTS PRESENT'})")

    print()
    print(f"compared {total_elems} weights across {len(TENSORS)} real expert tensors; "
          f"{total_bad} bit differences (limit {ULP_LIMIT})")
    if bad == 0:
        print("VERDICT AGREE -- kernels/knj_q4k.h decodes real GGUF Q4_K/Q6_K "
              "bit-for-bit with gguf-py")
    else:
        print(f"VERDICT DISAGREE on {bad} tensor(s)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
