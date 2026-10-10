#!/usr/bin/env python3
"""Validate the engine's new block decoders against gguf-py, byte for byte.

For each type: craft blocks (seeded-random + all-zero + all-0xFF edge),
run the engine's dequant_bytes_probe on the raw bytes, and compare against
gguf.quants.dequantize on the SAME bytes. Types gguf-py cannot decode
(Q8_1, Q8_K, Q1_0) get exact-construction checks instead (hand-built blocks
with exactly-representable values must decode bit-exactly).

Usage:
  python tools/dequant_validate.py [--probe PATH] [--seed N]
Exit 1 on any mismatch past tolerance.
"""
import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

import gguf
from gguf.constants import GGMLQuantizationType as T

PROBE_DEFAULT = os.path.join("build", "dequant_probe", "bytes_probe.exe")

# Block geometry (weights, bytes) mirrors gguf-py's GGML_QUANT_SIZES, except
# Q8_1 where the C struct (36 B) overrules gguf-py's size table (40 B, and
# gguf-py ships no Q8_1 codec at all).
GEOM = {
    "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_1": (32, 36), "Q1_0": (128, 18),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q8_K": (256, 292),
    "MXFP4": (32, 17), "NVFP4": (64, 36),
    "TQ1_0": (256, 54), "TQ2_0": (256, 66),
    "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ3_XXS": (256, 98),
    "IQ1_S": (256, 50), "IQ4_NL": (32, 18), "IQ3_S": (256, 110),
    "IQ2_S": (256, 82), "IQ4_XS": (256, 136), "IQ1_M": (256, 56),
}
GGUF_CODEC = set(GEOM) - {"Q8_1", "Q8_K", "Q1_0"}
TOL = 1e-5


def probe_env():
    # MinGW probe binaries need libstdc++-6.dll et al. at load; a missing DLL
    # surfaces as FileNotFoundError, not a loader error. Inherit everything
    # and prepend the compiler that (presumably) built the probe, so standalone
    # runs work without staged DLL copies. ctest does the same via CMake's
    # ENVIRONMENT_MODIFICATION; this is the no-CMake equivalent.
    e = dict(os.environ)
    g = shutil.which("g++")
    if g and sys.platform == "win32":
        b = os.path.dirname(g)
        e["PATH"] = b + os.pathsep + e.get("PATH", "")
    return e


def run_probe(probe, tname, blob, n, tmp):
    bp = os.path.join(tmp, f"{tname}.bin")
    op = os.path.join(tmp, f"{tname}.f32")
    with open(bp, "wb") as f:
        f.write(blob)
    r = subprocess.run([probe, tname, bp, str(n), op],
                       capture_output=True, text=True, timeout=120,
                       env=probe_env())
    if r.returncode != 0:
        return None, f"probe rc={r.returncode}: {(r.stderr or '').strip()[:200]}"
    return np.fromfile(op, dtype=np.float32), ""


def check_codec(probe, tname, tmp, rng):
    bw, bb = GEOM[tname]
    nblocks = 4
    rand = rng.integers(0, 256, size=(nblocks - 2) * bb, dtype=np.uint8)
    edge = np.zeros(2 * bb, dtype=np.uint8)
    edge[bb:] = 0xFF
    blob = np.concatenate([rand, edge]).tobytes()
    n = nblocks * bw
    got, err = run_probe(probe, tname, blob, n, tmp)
    if got is None:
        return False, err
    ref = np.asarray(
        gguf.quants.dequantize(np.frombuffer(blob, dtype=np.uint8),
                               T.__members__[tname]),
        dtype=np.float32).reshape(-1)
    if got.shape != ref.shape:
        return False, f"shape {got.shape} vs {ref.shape}"
    # The all-0xFF edge block makes fp16 scales NaN, so both sides emit NaN.
    # NaN positions must agree exactly; values are compared elsewhere.
    nan_got = np.isnan(got)
    nan_ref = np.isnan(ref)
    if not np.array_equal(nan_got, nan_ref):
        return False, (f"NaN positions differ: engine "
                       f"{int(nan_got.sum())} vs ref {int(nan_ref.sum())}")
    live = ~nan_ref
    # Verdict by equality (inf == inf is True in IEEE): maxabs over
    # inf-agreeing elements would report inf-inf as nan and fail a run whose
    # values actually agree. maxabs is measured over finite pairs only.
    agree = bool(np.all(got[live] == ref[live]))
    fin = live & np.isfinite(got) & np.isfinite(ref)
    fdiff = np.abs(got[fin] - ref[fin])
    scale = max(1.0, float(np.abs(ref[fin]).max(initial=0.0)))
    exact = int(np.count_nonzero(got == ref)) + int(nan_ref.sum())
    m = float(fdiff.max(initial=0.0))
    msg = (f"n={n} exact={exact}/{n} maxabs={m:.3e} "
           f"(tol {TOL * scale:.3e})")
    return agree and m <= TOL * scale, msg


def check_exact_q8k(probe, tmp):
    # d=2.0, qs = int8 ramp: every output exactly 2*qs in fp32.
    blob = bytearray()
    want = []
    for _ in range(2):
        blob += struct.pack("<f", 2.0)
        qs = [(i % 256) - 128 for i in range(256)]
        blob += struct.pack("256b", *qs)
        blob += bytes(32)
        want += [2.0 * q for q in qs]
    got, err = run_probe(probe, "Q8_K", bytes(blob), 512, tmp)
    if got is None:
        return False, err
    want = np.array(want, np.float32)
    return bool(np.array_equal(got, want)), \
        f"maxabs={np.abs(got - want).max():.3e}"


def check_exact_q81(probe, tmp):
    # d=0.5, s=garbage (must be ignored), qs = ramp: exact 0.5*qs.
    blob = bytearray()
    want = []
    for _ in range(2):
        blob += np.float16(0.5).tobytes()
        blob += b"\x7c\x7c"  # s = garbage nan-ish, must not leak into values
        qs = [(i % 256) - 128 for i in range(32)]
        blob += struct.pack("32b", *qs)
        want += [0.5 * q for q in qs]
    got, err = run_probe(probe, "Q8_1", bytes(blob), 64, tmp)
    if got is None:
        return False, err
    want = np.array(want, np.float32)
    return bool(np.array_equal(got, want)), \
        f"maxabs={np.abs(got - want).max():.3e}"


def check_exact_q10(probe, tmp):
    # d=1.5, qs=0xAA -> alternating -1.5/+1.5 starting with bit0=0.
    blob = bytearray()
    want = []
    for _ in range(2):
        blob += np.float16(1.5).tobytes() + bytes([0xAA] * 16)
        for l in range(128):
            want.append(1.5 if (l % 2) else -1.5)
    got, err = run_probe(probe, "Q1_0", bytes(blob), 256, tmp)
    if got is None:
        return False, err
    want = np.array(want, np.float32)
    return bool(np.array_equal(got, want)), \
        f"maxabs={np.abs(got - want).max():.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=PROBE_DEFAULT)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for tname in sorted(GEOH for GEOH in GEOM):
            if tname in GGUF_CODEC:
                ok, msg = check_codec(a.probe, tname, tmp, rng)
            elif tname == "Q8_K":
                ok, msg = check_exact_q8k(a.probe, tmp)
            elif tname == "Q8_1":
                ok, msg = check_exact_q81(a.probe, tmp)
            else:
                ok, msg = check_exact_q10(a.probe, tmp)
            print(f"{tname:10s} {'PASS' if ok else 'FAIL'}  {msg}")
            bad += not ok
    print("AGREE" if bad == 0 else f"DISAGREE on {bad} type(s)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
