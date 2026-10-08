#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pack_common.py -- the single source of the W4/W3/W2 weight-pack constants for the
Python tools (kvroof + route). Every Python file that needs "bytes per stored
weight" imports from here, so the 0.5+3/128 number can only drift in one place.

The C side's canonical definition lives in kernels/knj_pack.h
(KNJ_BITS_PER_WEIGHT_*, KNJ_META_BYTES_PER_GROUP, KNJ_W*_BYTES_PER_WEIGHT,
knj_bytes_per_weight()). This module is the PYTHON mirror of that header, kept
deliberately small and imported by everything that needs it, so that the
3 B/group rule and the bit-width ladder are stated once in Python and once in C
-- never seven times across bench + kvroof + route.

Provenance of every constant below: docs/00-verified-facts.md sections 7.6 and
8.10, and kernels/knj_pack.h. Change them there first; mirror here.
"""

from __future__ import annotations

# ---- machine-independent pack constants (mirrors kernels/knj_pack.h) ----

# Metadata is 3 bytes per weight-group for EVERY bit width: fp16 scale (2 B) +
# uint8 zero-point (1 B). This is the single metadata rule across W8/W6/W4/W3/W2.
META_BYTES_PER_GROUP = 3

# Group sizes. Expert weights use 128; KV keeps 64 by default (the metadata cost
# is amortised over KV's repeated reads).
GROUP_EXPERT = 128
GROUP_KV = 64

# Bit widths, in bits per stored weight (payload only).
BITS_W8 = 8
BITS_W6 = 6
BITS_W4 = 4
BITS_W3 = 3
BITS_W2 = 2

# Pre-composed bytes/weight for the packs the docs and the benches actually cite.
# W4 g128 = 0.5 + 3/128 = 0.5234  (the headline number; the one that drifted to
# 0.5186 once and is now the reason this module exists).
W4_BYTES_PER_WEIGHT = BITS_W4 / 8.0 + META_BYTES_PER_GROUP / GROUP_EXPERT      # 0.5234
W3_BYTES_PER_WEIGHT = BITS_W3 / 8.0 + META_BYTES_PER_GROUP / GROUP_EXPERT      # 0.3984
W2_BYTES_PER_WEIGHT = BITS_W2 / 8.0 + META_BYTES_PER_GROUP / GROUP_EXPERT      # 0.2734
W4G64_BYTES_PER_WEIGHT = BITS_W4 / 8.0 + META_BYTES_PER_GROUP / GROUP_KV       # 0.5469
W8_BYTES_PER_WEIGHT = BITS_W8 / 8.0 + META_BYTES_PER_GROUP / GROUP_EXPERT      # 1.0234

# Generic helper, mirrors knj_bytes_per_weight(bits, group) from knj_pack.h.
def bytes_per_weight(bits: int, group: int) -> float:
    """Bytes per stored weight for `bits`-bit payload at `group`-size metadata."""
    return bits / 8.0 + META_BYTES_PER_GROUP / float(group)
