#!/usr/bin/env python3
"""Check the engine's *fused* dequant dot against gguf-py + numpy.

The engine's mat-vec calls knj::dequant_dot_f32 per output column. That path
(Q4_K/Q6_K) decodes and dots in one pass and never writes an f32 weight buffer,
so dequant_crosscheck.py -- which reads dequant_row_f32 dumps -- does not cover
it. This compares the fused results against a reference dot computed from
gguf-py's independent dequantiser.

Usage:
  python tools/fused_dot_crosscheck.py <model.gguf> <probedir> <tensor> [<tensor> ...]
"""
import os
import sys

import numpy as np
import gguf
from gguf.quants import dequantize


def main():
    gpath, probedir = sys.argv[1], sys.argv[2]
    names = sys.argv[3:]
    reader = gguf.GGUFReader(gpath)
    by_name = {t.name: t for t in reader.tensors}

    total_bad = 0
    print(f"{'tensor':28s} {'n_in':>6s} {'rows':>7s} {'nvec':>4s} "
          f"{'max_abs':>11s} {'max_rel':>11s}  verdict")
    for name in names:
        t = by_name[name]
        n_in = int(t.shape[0])
        rows = int(np.prod(t.shape[1:])) if len(t.shape) > 1 else 1

        x = np.fromfile(os.path.join(probedir, name + ".fdot_x.f32"), dtype=np.float32)
        y = np.fromfile(os.path.join(probedir, name + ".fdot_y.f32"), dtype=np.float32)
        nvec = x.size // n_in
        if x.size != nvec * n_in or y.size != nvec * rows:
            print(f"{name:28s} SIZE MISMATCH x={x.size} y={y.size} "
                  f"want x={nvec * n_in} y={nvec * rows}")
            total_bad += 1
            continue
        x = x.reshape(nvec, n_in)

        # Independent reference: gguf-py dequantises the whole tensor (ne0
        # fastest), reshape to (rows, n_in) so row r is output column r.
        ref_w = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float64)
        ref_w = ref_w.reshape(rows, n_in)
        ref = ref_w @ x.astype(np.float64).T  # (rows, nvec)
        got = y.reshape(nvec, rows).T.astype(np.float64)

        diff = np.abs(got - ref)
        denom = np.maximum(np.abs(ref), 1e-12)
        rel = diff / denom
        scale = float(np.abs(ref).max())
        atol = 1e-5 * max(scale, 1e-30)
        rtol = 1e-4
        over = diff > (atol + rtol * np.abs(ref))
        n_over = int(np.count_nonzero(over))
        worst = int(np.argmax(diff))
        wr, wv = divmod(worst, nvec)
        total_bad += 1 if n_over else 0
        print(f"{name:28s} {n_in:6d} {rows:7d} {nvec:4d} "
              f"{diff.max():11.4e} {rel.max():11.3e}  "
              f"{'OK' if n_over == 0 else f'DIFF {n_over} over tol'}")
        if n_over:
            print(f"    worst row={wr} vec={wv} engine={got[wr, wv]:.6f} "
                  f"ref={ref[wr, wv]:.6f} diff={diff[wr, wv]:.3e}")

    print(f"\n{'AGREE' if total_bad == 0 else f'DISAGREE on {total_bad} tensor(s)'}")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
