#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_docs.py -- do the figures printed in the docs still match the tools?

`kv_roofline.py --check` audits the claims in `09`. This audits the claims in
`00` section 9, which are the ones that were *promoted* out of 09 -- so if they
drift, the drift lands in the file everything else cites as the source of truth.

It re-runs both tools, reads their machine-readable output, and asserts that
the numbers written in the prose are the numbers the tools actually produce.
Transcription error is a real failure mode for a document made of tables.

    python tools/check_docs.py

Exit codes:
    0   every documented figure matches the tool that produced it
    1   at least one documented figure is stale (listed)
    3   a tool failed to run, so nothing could be checked (NOT a pass)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC00 = os.path.join(ROOT, "docs", "00-verified-facts.md")
DOC09 = os.path.join(ROOT, "docs", "09-kv-engine-architecture.md")
DOC01 = os.path.join(ROOT, "docs", "01-architecture.md")
TARGET = os.path.join(ROOT, "models", "qwen3-30b-a3b", "config.json")

RESULTS = []


def check(label, condition):
    RESULTS.append((label, bool(condition)))


def _lru(doc, slots):
    """Slot key is an int in the tool output and formatted into prose here."""
    d = doc["aggregate"]["lru"]
    return d[slots] if slots in d else d[str(slots)]


def main():
    if not os.path.exists(TARGET):
        print("check_docs: %s missing -- vendored config required"
              % TARGET, file=sys.stderr)
        return 3

    # --- the roofline tool, machine-readable ------------------------------
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "kvroof",
                                      "kv_roofline.py"), TARGET, "--json"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        print("check_docs: kv_roofline.py failed with %d\n%s"
              % (proc.returncode, proc.stderr[-500:]), file=sys.stderr)
        return 3
    j = json.loads(proc.stdout)[0]

    # --- the I7 test ------------------------------------------------------
    i7 = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "i7", "i7_bit_identity.py")],
        capture_output=True, text=True)
    if i7.returncode not in (0, 10, 11):
        print("check_docs: i7_bit_identity.py failed with %d"
              % i7.returncode, file=sys.stderr)
        return 3

    # --- the routing artifacts -------------------------------------------
    r2_path = os.path.join(ROOT, "tools", "route", "route_target-gates.json")
    r1_path = os.path.join(ROOT, "tools", "route", "route_local.json")
    for p in (r2_path, r1_path):
        if not os.path.exists(p):
            print("check_docs: %s missing -- run route_locality.py first"
                  % os.path.basename(p), file=sys.stderr)
            return 3
    r2 = json.load(open(r2_path, encoding="utf-8"))
    r1 = json.load(open(r1_path, encoding="utf-8"))

    d00 = open(DOC00, encoding="utf-8").read()
    d09 = open(DOC09, encoding="utf-8").read()
    d01 = open(DOC01, encoding="utf-8").read()

    # --- section 9.2 / 9.3 / 9.5 -----------------------------------------
    check("9.2 FP16 KV B/token is 98,304",
          j["kv_bytes_per_token_fp16"] == 98304 and "98,304 (96.0 KiB)" in d00)
    check("9.3 expert-only weight traffic is 948 MB/token",
          round(j["weight_bytes_per_token_expert_only"] / 1e6) == 948
          and "**948**" in d00)
    check("9.3 total weight traffic is 1,586 MB/token",
          round(j["weight_bytes_per_token_total"] / 1e6) == 1586
          and "**1,586**" in d00)
    check("9.5 resident weights are 15.31 GiB",
          abs(j["resident_weight_bytes"] / 1024.0 ** 3 - 15.31) < 0.01
          and "15.31 GiB" in d00)
    check("9.5 KV budget really is negative at 16 GiB",
          j["kv_budget_bytes"] < 0)
    check("9.1 section is a sibling of section 8, not a replacement",
          "## §9 Measured" in d00 and "## §8 Measured" in d00)

    # --- section 9.6 ------------------------------------------------------
    for slots, docval in ((26, 67.7), (51, 86.2), (64, 90.4),
                          (90, 95.7), (115, 98.1)):
        got = 100.0 * _lru(r2, slots)
        check("9.6 target-gates LRU@%d = %.1f%%" % (slots, docval),
              abs(got - docval) < 0.15 and ("%.1f" % docval) in d00)
    for slots, docval in ((12, 35.2), (24, 55.3), (30, 63.5),
                          (42, 78.3), (54, 92.1)):
        got = 100.0 * _lru(r1, slots)
        check("9.6 local-MoE LRU@%d = %.1f%%" % (slots, docval),
              abs(got - docval) < 0.15 and ("%.1f" % docval) in d00)
    check("9.6 new-expert rate, source 1, is 80.5%",
          abs(100.0 * (1 - r1["aggregate"]["reuse_med"]) - 80.5) < 0.15
          and "80.5%" in d00)
    check("9.6 new-expert rate, source 2, is 59.1%",
          abs(100.0 * (1 - r2["aggregate"]["reuse_med"]) - 59.1) < 0.15
          and "59.1%" in d00)
    check("9.6 carries the proxy caveat on source 2",
          "PROXY" in r2["caveat"].upper() and "proxy" in d00.lower())

    # --- section 9.7 ------------------------------------------------------
    check("9.7 I7 test passes", i7.returncode == 0)
    check("9.7 every negative control was detected",
          i7.stdout.count("DETECTED") == 3 and "MISSED" not in i7.stdout)

    # --- corrections propagated to the sibling docs ------------------------
    check("09 no longer quotes the stale 13.8 GiB bank", "13.8 GiB" not in d09)
    check("09 no longer quotes INT4 g128 at 24 KB",
          "| INT4 g128 | 24 KB" not in d09)
    check("01 carries the corrected 14.1 GiB bank", "14.1 GiB" in d01)

    for label, ok in RESULTS:
        print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    failed = [lab for lab, ok in RESULTS if not ok]
    print("")
    if failed:
        print("check_docs: %d of %d documented figures are STALE."
              % (len(failed), len(RESULTS)))
        return 1
    print("check_docs: all %d documented figures match the tools."
          % len(RESULTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())