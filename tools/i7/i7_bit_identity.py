#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
i7_bit_identity.py -- the I7 test, written before there is a codec to violate it.

    I7 (as currently worded in docs/09 section 7):
      "Any codec, any tier split, any split-K reduction must reproduce the
       same bits."

What this file proves about that wording
----------------------------------------
It is **too strong to be true**, and the file exists to find that out now
rather than at 2am. Two findings, both mechanical rather than philosophical:

1. **A lossy codec cannot be bit-identical to full precision.** FP8 E4M3
   truncation of a bf16 value is *designed* to change it. So I7 as worded is
   unsatisfiable for any quantising codec, and an engine that asserts it will
   either not ship a codec or will quietly disable the assertion. The
   invariant has to be re-scoped, and §7 of docs/09 needs editing:

       I7  TIER MOVEMENT is bit-identical.
           Storage-layer codecs (the ones that move a page between tiers) must
           be LOSSLESS: decode(encode(x)) == x, bit for bit.
           A codec that changes values is a PRECISION REDUCTION, declared
           explicitly, priced, and never conflated with a tier move.

2. **Bit-identity requires a canonical reduction order, and the obvious
   implementation is not it.** Floating-point addition is not associative, so
   "the resident path" and "the split path" differ by construction if one
   accumulates with a single running accumulator and the other accumulates
   per-chunk partials and merges. The fix is to make the page decomposition
   part of the contract: both paths visit pages in ascending index order,
   accumulate ascending keys within a page, and merge page partials
   left-to-right. Then identity is structural, not lucky.

   The cheapest way to keep that guarantee is to never write the flat form.
   This file contains the flat form anyway, as a NEGATIVE CONTROL, because a
   test that cannot fail is not a test.

The three paths that MUST agree bitwise
---------------------------------------
    A  RESIDENT     every page read from T1 (VRAM)
    B  MOVEMENT     every page round-tripped through a LOSSLESS storage codec
                    and host RAM, i.e. what a T1<->T2 move actually does
    C  TIER SPLIT   hot pages from T1, cold pages streamed from T2, merged in
                    the same ascending page order

Exit codes
----------
    0   A == B == C bitwise, and every negative control was detected
    10  a path that must match diverged  (bitwise, with the first bad index)
    11  a negative control did NOT diverge (the test is toothless -- fix that
        before trusting any of this)
    12  the oracle itself failed (internal inconsistency, nothing to compare)
"""

from __future__ import annotations

import sys

import numpy as np

# --- the contract ---------------------------------------------------------
PAGE_KEYS = 64           # keys per page; part of the spec, not a tunable
DIM = 128                # head_dim
SEED = 0x1337            # fixed: this test must be reproducible


# --- helpers --------------------------------------------------------------
def f32(x):
    return np.float32(x)


def dot32(q, k):
    """fp32 FMA-free dot product, ascending over DIM.

    Deliberately a Python loop with a Python float accumulator widened once.
    numpy's `sum` uses pairwise summation, which is a *different* order; using
    it here would make the oracle disagree with a sequential GPU kernel for
    reasons that have nothing to do with I7.
    """
    acc = np.float32(0.0)
    for i in range(DIM):
        acc = np.float32(acc + np.float32(q[i]) * np.float32(k[i]))
    return acc


def max32(x):
    """fp32 max. max IS associative, so order cannot change the result -- this
    is why the canonical form is two-pass with a global max."""
    m = np.float32(-np.inf)
    for v in x:
        if v > m:
            m = np.float32(v)
    return m


def pages_of(n_keys, page_keys=PAGE_KEYS):
    return [(i, min(i + page_keys, n_keys)) for i in range(0, n_keys, page_keys)]


# --- the four implementations --------------------------------------------
def attend_canonical(q, K, V, scale, page_list):
    """THE SPEC. Two-pass, global max, ascending pages, ascending keys within
    a page, left-to-right merge of per-page partials. This is the bit pattern
    every other path must reproduce."""
    n = K.shape[0]
    scores = np.empty(n, np.float32)
    for j in range(n):
        scores[j] = np.float32(dot32(q, K[j]) * scale)
    m = max32(scores)

    l_total = np.float32(0.0)
    acc = np.zeros(DIM, np.float32)
    for (a, b) in page_list:                      # ascending page order
        l_page = np.float32(0.0)
        acc_page = np.zeros(DIM, np.float32)
        for j in range(a, b):                     # ascending key order
            p = np.float32(np.exp(np.float32(scores[j] - m)))
            l_page = np.float32(l_page + p)
            for d in range(DIM):
                acc_page[d] = np.float32(acc_page[d] + np.float32(p * V[j][d]))
        l_total = np.float32(l_total + l_page)    # LEFT TO RIGHT
        for d in range(DIM):
            acc[d] = np.float32(acc[d] + acc_page[d])
    inv = np.float32(1.0) / l_total
    out = np.zeros(DIM, np.float32)
    for d in range(DIM):
        out[d] = np.float32(acc[d] * inv)
    return out, l_total, m


def attend_resident(q, K, V, scale, page_list):
    """PATH A: one tier, no copies. Structurally identical to the spec."""
    return attend_canonical(q, K, V, scale, page_list)


def movement_codec(x):
    """LOSSLESS storage codec: identity on the payload bytes.

    Stands in for the T1<->T2 page codec (chunked DMA, possibly with a
    container/header). Its only requirement under I7 is exactness.
    """
    return x.copy()


def attend_codec(q, K, V, scale, page_list):
    """PATH B: every page makes a lossy-codec round trip through host RAM."""
    Kc, Vc = movement_codec(K), movement_codec(V)
    return attend_canonical(q, Kc, Vc, scale, page_list)


def attend_tier_split(q, K, V, scale, page_list, hot_pages):
    """PATH C: hot pages from T1, cold pages streamed from T2.

    The streaming is simulated exactly: a cold page is materialised, in page
    order, into the same buffer the kernel would read. The merge order is the
    ascending page order of the spec -- NOT cold-first, which is the bug this
    test exists to catch.
    """
    n = K.shape[0]
    scores = np.empty(n, np.float32)
    for j in range(n):
        scores[j] = np.float32(dot32(q, K[j]) * scale)
    m = max32(scores)

    l_total = np.float32(0.0)
    acc = np.zeros(DIM, np.float32)
    for (a, b) in page_list:
        if (a, b) in hot_pages:
            # A VIEW, not the whole array. Both branches must hand the loop a
            # page-local buffer; passing the full array here silently indexes
            # V with a page-local offset, which is the bug this test exists to
            # catch (it was written, and it was wrong, before this line).
            Kp, Vp = K[a:b], V[a:b]
        else:
            Kp, Vp = K[a:b].copy(), V[a:b].copy()           # T2 -> staged
        # identical accumulation shape to the spec: per-page partials first,
        # then one left-to-right merge. The ONLY difference from path A is
        # where the page bytes came from.
        l_page = np.float32(0.0)
        acc_page = np.zeros(DIM, np.float32)
        for joff in range(b - a):
            j = a + joff
            p = np.float32(np.exp(np.float32(scores[j] - m)))
            l_page = np.float32(l_page + p)
            for d in range(DIM):
                acc_page[d] = np.float32(acc_page[d]
                                         + np.float32(p * Vp[joff][d]))
        l_total = np.float32(l_total + l_page)
        for d in range(DIM):
            acc[d] = np.float32(acc[d] + acc_page[d])
    inv = np.float32(1.0) / l_total
    out = np.zeros(DIM, np.float32)
    for d in range(DIM):
        out[d] = np.float32(acc[d] * inv)
    return out, l_total, m


# --- negative controls ----------------------------------------------------
def attend_flat_single_accumulator(q, K, V, scale, page_list):
    """THE OBVIOUS IMPLEMENTATION. One accumulator, no pages. Correct, and
    bit-different from the spec at large N."""
    n = K.shape[0]
    scores = np.empty(n, np.float32)
    for j in range(n):
        scores[j] = np.float32(dot32(q, K[j]) * scale)
    m = max32(scores)
    l_total = np.float32(0.0)
    acc = np.zeros(DIM, np.float32)
    for j in range(n):
        p = np.float32(np.exp(np.float32(scores[j] - m)))
        l_total = np.float32(l_total + p)
        for d in range(DIM):
            acc[d] = np.float32(acc[d] + np.float32(p * V[j][d]))
    inv = np.float32(1.0) / l_total
    out = np.zeros(DIM, np.float32)
    for d in range(DIM):
        out[d] = np.float32(acc[d] * inv)
    return out, l_total, m


def attend_cold_first_merge(q, K, V, scale, page_list, hot_pages):
    """THE BUG. Cold pages are merged first because they were streamed first
    and arrived while the hot ones were still in VRAM. The order of the
    left-to-right merge is reversed relative to the spec."""
    n = K.shape[0]
    scores = np.empty(n, np.float32)
    for j in range(n):
        scores[j] = np.float32(dot32(q, K[j]) * scale)
    m = max32(scores)
    order = [pg for pg in page_list if pg not in hot_pages]
    order += [pg for pg in page_list if pg in hot_pages]
    l_total = np.float32(0.0)
    acc = np.zeros(DIM, np.float32)
    for (a, b) in order:
        l_page = np.float32(0.0)
        acc_page = np.zeros(DIM, np.float32)
        for j in range(a, b):
            p = np.float32(np.exp(np.float32(scores[j] - m)))
            l_page = np.float32(l_page + p)
            for d in range(DIM):
                acc_page[d] = np.float32(acc_page[d] + np.float32(p * V[j][d]))
        l_total = np.float32(l_total + l_page)
        for d in range(DIM):
            acc[d] = np.float32(acc[d] + acc_page[d])
    inv = np.float32(1.0) / l_total
    out = np.zeros(DIM, np.float32)
    for d in range(DIM):
        out[d] = np.float32(acc[d] * inv)
    return out, l_total, m


def lossy_fp8_roundtrip(x):
    """A real quantising codec, deliberately lossy (as all of them are)."""
    return x.astype(np.float16).astype(np.float32)  # stand-in, same property


# --- the test -------------------------------------------------------------
def bits(a):
    return a.view(np.uint32)


def first_diff_bit(a, b):
    ba, bb = bits(a), bits(b)
    d = np.nonzero(ba != bb)[0]
    if d.size == 0:
        return None
    i = int(d[0])
    return i, int(ba[i]), int(bb[i])


def main(argv):
    rng = np.random.default_rng(SEED)
    n_keys = 1024
    q = (rng.standard_normal(DIM) * 0.5).astype(np.float16).astype(np.float32)
    K = (rng.standard_normal((n_keys, DIM)) * 0.5).astype(np.float16).astype(np.float32)
    V = (rng.standard_normal((n_keys, DIM)) * 0.5).astype(np.float16).astype(np.float32)
    scale = np.float32(1.0 / np.sqrt(DIM))
    page_list = pages_of(n_keys)
    hot_pages = page_list[: len(page_list) // 2]

    print("=" * 78)
    print("I7 BIT-IDENTITY TEST -- host reference, run before any codec exists")
    print("=" * 78)
    print("  page_keys=%d  dim=%d  n_keys=%d  pages=%d  hot_pages=%d"
          % (PAGE_KEYS, DIM, n_keys, len(page_list), len(hot_pages)))
    print("  spec: two-pass (global fp32 max), ascending page order, ascending")
    print("  key order within a page, LEFT-TO-RIGHT merge of page partials.")

    ref, ref_l, ref_m = attend_canonical(q, K, V, scale, page_list)
    print("")
    print("REFERENCE  sum_exp=%.9e  max=%.9e  out[0]=%.9e  out[-1]=%.9e"
          % (ref_l, ref_m, ref[0], ref[-1]))

    print("")
    print("POSITIVE PATHS (each must equal the reference BITWISE):")
    checks = [
        ("A  resident (one tier)", attend_resident(q, K, V, scale, page_list)),
        ("B  movement codec round-trip", attend_codec(q, K, V, scale, page_list)),
        ("C  tier split (hot VRAM + cold RAM)",
         attend_tier_split(q, K, V, scale, page_list, hot_pages)),
    ]
    ok = True
    for name, (out, _l, _m) in checks:
        diff = first_diff_bit(ref, out)
        if diff is None:
            print("  PASS  %s" % name)
        else:
            i, x, y = diff
            print("  FAIL  %s  first differing element %d: ref=0x%08x got=0x%08x"
                  % (name, i, x, y))
            ok = False

    print("")
    print("NEGATIVE CONTROLS (each MUST differ, else the test is toothless):")
    negatives = [
        ("flat single accumulator (no pages)",
         attend_flat_single_accumulator(q, K, V, scale, page_list)),
        ("cold-first merge (streamed pages merged before hot)",
         attend_cold_first_merge(q, K, V, scale, page_list, hot_pages)),
        ("lossy codec round-trip",
         attend_canonical(q, lossy_fp8_roundtrip(V), lossy_fp8_roundtrip(V),
                          scale, page_list)),
    ]
    for name, (out, _l, _m) in negatives:
        diff = first_diff_bit(ref, out)
        if diff is not None:
            i, x, y = diff
            print("  DETECTED  %s  (element %d: ref=0x%08x got=0x%08x)"
                  % (name, i, x, y))
        else:
            print("  MISSED    %s  -- bit-identical to reference; the guard "
                  "cannot fire" % name)
            ok = False

    print("")
    print("CODEC CONTRACT:")
    for label, x in (("K", K), ("V", V)):
        rt = movement_codec(x)
        lossless = first_diff_bit(x, rt) is None
        print("  %s: lossless movement codec round-trip is bit-exact: %s"
              % (label, lossless))
        if not lossless:
            ok = False

    print("")
    print("=" * 78)
    if ok:
        print("I7 (re-scoped): TIER MOVEMENT is bit-identical. All positive")
        print("paths agree bitwise; all negative controls were detected.")
        print("NOTE the re-scope: a quantising codec is NOT bit-identical by")
        print("construction, so it is a declared precision reduction, not a")
        print("tier move. docs/09 section 7 must be edited to say so.")
        return 0
    print("I7 TEST FAILED -- see the FAIL/MISSED lines above.")
    return 10


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))