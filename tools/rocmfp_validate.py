#!/usr/bin/env python3
"""Validate the engine's ROCmFP decoders against the fork's own references.

For each of the six fork-experimental types (GGML ids 100/101/102/103/104/107):
craft blocks (seeded-random + all-zero + all-0xFF + scale-sweep edge), run the
engine's dequant_bytes_probe on the raw bytes, and compare BIT-EXACTLY against
tmp/attnrep/rocmfp_ref.exe, which links the fork's unmodified rocmfp4.c /
rocmfpx.c. gguf-py has no codecs for these types, so the fork itself is the
oracle (same pattern as the Q3_K ggml-C triple-check in Phase 50).
Fork provenance: https://github.com/charlie12345/ROCmFPX @ fb08d7c
(2026-09-23). See the COLLISION PROTOCOL in src/loader/gguf.h: ids 100+ are
fork-only and must move if upstream ggml ever reassigns them.

Usage:
  python tools/rocmfp_validate.py [--probe PATH] [--ref PATH] [--seed N]
Exit 1 on any mismatch.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

TYPES = {
    # engine TYPE arg, fork ref name, block bytes, weights per block
    "Q4_0_ROCMFP4":      ("q4_0_rocmfp4", 18, 32),
    "Q4_0_ROCMFP4_FAST": ("q4_0_rocmfp4_fast", 17, 32),
    "Q2_0_ROCMFPX":      ("q2_0_rocmfpx", 10, 32),
    "Q3_0_ROCMFPX":      ("q3_0_rocmfpx", 14, 32),
    "Q6_0_ROCMFPX":      ("q6_0_rocmfpx", 26, 32),
    "Q8_0_ROCMFPX":      ("q8_0_rocmfpx", 33, 32),
}

NBLOCKS = 4  # blocks per case: covers cross-block state (there is none, prove it)


def run_decoder(exe, args, blob, n, tmp, tag):
    bp = os.path.join(tmp, f"{tag}.bin")
    op = os.path.join(tmp, f"{tag}.f32")
    with open(bp, "wb") as f:
        f.write(blob)
    r = subprocess.run([exe] + args + [bp, str(n), op],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None, f"rc={r.returncode} {r.stderr.strip()[:100]}"
    return np.fromfile(op, dtype=np.float32), ""


def check(probe, ref, tname, rname, bb, bw, blob, n, tmp):
    got, err = run_decoder(probe, [tname], blob, n, tmp, "e")
    if got is None:
        return False, f"engine {err}"
    want, err = run_decoder(ref, [rname], blob, n, tmp, "r")
    if want is None:
        return False, f"ref {err}"
    if got.shape != want.shape or got.shape != (n,):
        return False, f"shape {got.shape} vs {want.shape}"
    bad = np.nonzero(~(got == want) & ~(np.isnan(got) & np.isnan(want)))[0]
    if len(bad):
        i = int(bad[0])
        return False, f"{len(bad)}/{n} differ, first @{i}: e={got[i]!r} r={want[i]!r}"
    exact = int(np.count_nonzero(got == want))
    return True, f"exact={exact}/{n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=os.path.join("tmp", "dequant_bytes_probe.exe"))
    ap.add_argument("--ref", default=os.path.join("tmp", "attnrep", "rocmfp_ref.exe"))
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    if not os.path.isfile(a.probe):
        print(f"probe missing: {a.probe} -- build it first")
        return 2
    if not os.path.isfile(a.ref):
        print(f"ref missing: {a.ref} -- build tmp/attnrep/rocmfp_ref.exe first")
        return 2

    rng = np.random.default_rng(a.seed)
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        for tname, (rname, bb, bw) in TYPES.items():
            n = NBLOCKS * bw
            nb = NBLOCKS * bb
            cases = {
                "zeros": bytes(nb),
                "ones": bytes([0xFF] * nb),
            }
            # scalesweep: last byte(s) of each block cycle 0x7c..0x82, hitting
            # the 0x7E scale-validity edge in every scale position.
            blob = bytearray()
            for b in range(NBLOCKS):
                blk = bytearray(rng.integers(0, 256, bb, dtype=np.uint8).tolist())
                blk[-1] = 0x7C + (b % 7)
                if bb >= 14:
                    blk[-2] = 0x7C + ((b + 3) % 7)
                blob += blk
            cases["scalesweep"] = bytes(blob)
            for i in range(3):
                cases[f"rand{i}"] = rng.integers(
                    0, 256, nb, dtype=np.uint8).tobytes()
            for cname, blob in cases.items():
                ok, msg = check(a.probe, a.ref, tname, rname, bb, bw,
                                blob, n, tmp)
                print(f"{tname:18s} {cname:10s} {'PASS' if ok else 'FAIL'}  {msg}")
                fails += not ok
    print("AGREE" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
