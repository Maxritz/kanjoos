#!/usr/bin/env python3
"""Compare the engine's dequantisers against gguf-py's, on real model bytes.

gguf-py's dequantisers come from the ggml authors; the engine's are a separate
transcription. Agreement on real tensors is evidence that the engine's block
decoders are right, which is exactly the thing a strided sample cannot show.

Usage:
  python tools/dequant_crosscheck.py <model.gguf> <dumpdir> <tensor> [<tensor> ...]
"""
import os
import sys

import numpy as np
import gguf
from gguf.quants import dequantize


def main():
    gpath, dumpdir = sys.argv[1], sys.argv[2]
    names = sys.argv[3:]
    reader = gguf.GGUFReader(gpath)
    by_name = {t.name: t for t in reader.tensors}

    bad = 0
    for name in names:
        t = by_name[name]
        ref = dequantize(t.data, t.tensor_type)
        ref = np.asarray(ref, dtype=np.float32).reshape(-1)
        path = os.path.join(dumpdir, name + ".f32")
        got = np.fromfile(path, dtype=np.float32)
        if got.size != ref.size:
            print(f"{name:28s} SIZE MISMATCH engine={got.size} gguf-py={ref.size}")
            bad += 1
            continue
        diff = np.abs(got - ref)
        scale = np.maximum(np.abs(ref), 1e-6)
        worst = int(np.argmax(diff))
        exact = int(np.count_nonzero(diff == 0))
        print(
            f"{name:28s} n={ref.size:9d} exact={exact:9d}/{ref.size:<9d} "
            f"maxabs={diff.max():.6e} maxrel={float((diff / scale).max()):.3e} "
            f"worst_idx={worst} engine={got[worst]:.6f} ref={ref[worst]:.6f}"
        )
        if diff.max() > 1e-5 * max(1.0, float(np.abs(ref).max())):
            bad += 1
    print(f"\n{'AGREE' if bad == 0 else f'DISAGREE on {bad} tensor(s)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
