#!/usr/bin/env python3
"""Bisect two implementations of the same forward pass.

The engine writes named f32 intermediates when KNJ_DUMP_HIDDEN is set; the
numpy oracle (tools/ref_qwen3moe.py) writes the same names. This compares them
and reports the *first* name in forward order where they part company, which is
the only useful thing to know: after the first divergence every later tensor is
downstream and its disagreement is not independent evidence.

Tolerance model
---------------
Two independent fp32 implementations of the same reduction do not agree
elementwise in absolute terms when the *result* is near zero but the *terms*
are not -- that is cancellation, and fp32 carries no information about it.
A logit whose true value is 1e-3 computed as a 2048-term dot product of order-13
magnitudes inherits an absolute error of order sqrt(2048)*eps*13 ~ 7e-5. So the
gate is:

  * scale-relative:   |ref - eng| <= atol + rtol * max|ref|     (the pass gate)
  * strict per-element: |ref - eng| <= atol + rtol * |ref|      (printed, not the gate)

Both counts are always printed, so a real error is never hidden behind the
looser one. On top of that the tool hard-fails -- never silently passes -- on:

  * a tensor the engine did not write, or wrote as all zeros while ref is not;
  * any NaN or Inf in the engine output;
  * the argmax of 12_logits disagreeing with the oracle.

Those three are the defects that were previously invisible: a zeroed buffer and
a wrong-but-finite buffer both used to be reported as "OK".

Usage:
  python tools/hidden_diff.py <refdir> <engdir> [--rtol 1e-4] [--atol 1e-6]
                                     [--show 8] [--align-rows]
"""

import argparse
import os
import sys

import numpy as np

LOGITS = "12_logits"


def compare(name, ref, eng, rtol, atol):
    """Return (status, bad_strict, bad_scale, dmax, worst, rv, ev, note)."""
    d = np.abs(ref.astype(np.float64) - eng.astype(np.float64))
    scale = float(np.max(np.abs(ref.astype(np.float64)))) if ref.size else 0.0
    tol_strict = atol + rtol * np.abs(ref.astype(np.float64))
    tol_scale = atol + rtol * scale
    bad_strict = int(np.count_nonzero(d > tol_strict))
    bad_scale = int(np.count_nonzero(d > tol_scale))
    worst = int(np.argmax(d)) if d.size else 0
    rv = float(ref[worst]) if d.size else 0.0
    ev = float(eng[worst]) if d.size else 0.0
    dmax = float(d.max()) if d.size else 0.0

    if not np.all(np.isfinite(eng)):
        return "NONFIN", bad_strict, bad_scale, dmax, worst, rv, ev, "engine has NaN/Inf"
    if float(np.max(np.abs(eng))) == 0.0 and scale != 0.0:
        return "ZEROS", bad_strict, bad_scale, dmax, worst, rv, ev, "engine buffer is all zero"
    if bad_strict == 0:
        return "OK", bad_strict, bad_scale, dmax, worst, rv, ev, ""
    if bad_scale == 0:
        note = f"{bad_strict} element(s) beyond per-element tol, all within tensor scale"
        return "OK~", bad_strict, bad_scale, dmax, worst, rv, ev, note
    return "DIFF", bad_strict, bad_scale, dmax, worst, rv, ev, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refdir")
    ap.add_argument("engdir")
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--align-rows", action="store_true",
                    help="if the engine tensor is an integer multiple of the "
                         "reference, compare against its trailing block")
    args = ap.parse_args()

    ref_names = sorted(
        f[: -len(".f32")] for f in os.listdir(args.refdir) if f.endswith(".f32")
    )
    hard_fail = []
    first_bad = None
    missing = 0
    rows = []
    eng_logits = None

    for name in ref_names:
        rp = os.path.join(args.refdir, name + ".f32")
        ep = os.path.join(args.engdir, name + ".f32")
        if not os.path.exists(ep):
            print(f"{name:26s} MISSING from engine dump")
            missing += 1
            hard_fail.append(f"{name}: not written by the engine")
            if first_bad is None:
                first_bad = name
            continue
        ref = np.fromfile(rp, dtype=np.float32)
        eng = np.fromfile(ep, dtype=np.float32)
        if ref.size != eng.size:
            ratio = eng.size / ref.size if ref.size else 0
            if args.align_rows and ratio >= 1 and float(ratio).is_integer():
                eng = eng[-ref.size:]
                print(f"{name:26s} aligned: engine is {int(ratio)}x ref, using trailing block")
            else:
                print(f"{name:26s} SIZE MISMATCH ref={ref.size} eng={eng.size} (ratio {ratio:.4g})")
                hard_fail.append(f"{name}: shape mismatch")
                if first_bad is None:
                    first_bad = name
                continue
        status, bs, bsc, dmax, worst, rv, ev, note = compare(
            name, ref, eng, args.rtol, args.atol
        )
        if name == LOGITS:
            eng_logits = (ref, eng)
        rows.append((name, status, bs, dmax, worst, rv, ev, note))
        if status in ("DIFF", "ZEROS", "NONFIN") and first_bad is None:
            first_bad = name

    print(f"\n{'tensor':26s} {'':5s} {'strict':>8s} {'maxabs':>12s} {'worst':>8s}")
    for name, status, bs, dmax, worst, rv, ev, note in rows:
        print(f"{name:26s} {status:5s} {bs:8d} {dmax:12.4e} {worst:8d}  ref={rv:+.6f} eng={ev:+.6f}")
        if note and status != "OK":
            print(f"{'':26s}   {note}")

    # --- the semantic gate: does the answer agree? -------------------------
    if eng_logits is not None:
        ref, eng = eng_logits
        ra, ea = int(np.argmax(ref)), int(np.argmax(eng))
        rc, ec = ref[ra], eng[ea]
        agree = ra == ea
        print(f"\nargmax: ref={ra} (logit {rc:+.4f})  eng={ea} (logit {ec:+.4f})  "
              f"{'AGREE' if agree else 'DISAGREE'}")
        if not agree:
            hard_fail.append(f"{LOGITS}: argmax ref={ra} eng={ea}")
        k = min(10, ref.size)
        rt = set(np.argsort(-ref)[:k].tolist())
        et = set(np.argsort(-eng)[:k].tolist())
        print(f"top-{k} overlap: {len(rt & et)}/{k}")

    print()
    if missing:
        print(f"{missing} tensor(s) missing from the engine dump")
    if first_bad is None and not hard_fail:
        print("AGREE: every dumped tensor matches the independent oracle within tolerance")
        return 0
    if first_bad is not None:
        print(f"FIRST DIVERGENCE: {first_bad}")
        print("  every tensor after it is downstream of this one and is not independent evidence")
    for f in hard_fail:
        print(f"FAIL {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
