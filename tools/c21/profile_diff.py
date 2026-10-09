#!/usr/bin/env python3
"""Diff two C21 JSON profiles and gate on a regression.

WHY THIS EXISTS
---------------
`kanjoos-run --profiling=json` prints a component table. A table you look at is
not a gate: it tells you what is slow *today*, not whether today is worse than
yesterday, and it cannot fail a build. This tool makes the comparison the
artifact.

It reads a PINNED reference profile and the CURRENT profile, compares one metric
per component, and exits non-zero when a component's self time regresses beyond a
tolerance.

THREE DECISIONS THAT ARE NOT OBVIOUS
------------------------------------
1. A component that is in the reference and MISSING from the current profile is a
   **regression**, not a skip. This is the same rule `tools/bench/run_bench.sh`
   already applies to a pinned metric it cannot read: a measurement that vanished
   is indistinguishable from a component that stopped being measured, and the
   second case is worse than a slowdown. Renaming a scope therefore fails the
   gate, which is the point -- a rename silently breaks every comparison built on
   the old name.

2. A component whose reference self time is below `--min-us` is **not judged** and
   the report says so. The profiler's own instrumentation floor is published
   (`floor` in the JSON, `--profile-floor`) and a row near that floor is mostly
   marker cost; a 40% "regression" on 0.4 us is noise wearing a percentage.

3. The clock domains must match. Host interval time and device event time are
   different clocks measuring different things, so comparing them is meaningless
   rather than merely imprecise, and the tool refuses instead of producing a
   number.

RE-PINNING
----------
    python tools/c21/profile_diff.py --ref ref.json --cur new.json --repin \
        --reason "why the new number is the right number"
A re-pin without `--reason` is refused (exit 2). The written reference records
the reason, the previous reason it replaced, the metric, the tolerance, the clock
domain, and the time -- so the next person can see WHY the pin moved rather than
only that it did. Everything in that block is written by this tool; the reference
is otherwise the profile exactly as the profiler produced it.

EXIT CODES
    0  no judged component regressed beyond the tolerance
    1  at least one judged component regressed (or a reference component vanished)
    2  usage error (missing --reason on --repin, missing file, bad tolerance)
    3  the inputs are not comparable (invalid JSON, different clock domains)

RUN
    python tools/c21/profile_diff.py --self-test      # no profiles needed
"""

import argparse
import json
import os
import sys
import tempfile
import time

DEFAULT_TOL_PCT = 10.0
DEFAULT_METRIC = "dev_ns_self"


# ----------------------------------------------------------------------- io ---
def load_profile(path):
    """Read one C21 JSON profile. Raises ValueError with a reason, never a guess."""
    if not os.path.isfile(path):
        raise ValueError("no such file: %s" % path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError("%s is not valid JSON (%s)" % (path, e))
    if not isinstance(obj, dict):
        raise ValueError("%s is not a C21 profile object" % path)
    comps = obj.get("components")
    if not isinstance(comps, list):
        raise ValueError(
            "%s has no `components` array; is it a C21 profile? "
            "(`kanjoos-run --profiling=json`)" % path)
    out = {}
    for c in comps:
        if not isinstance(c, dict) or "component" not in c:
            continue
        out[str(c["component"])] = c
    obj["_by_name"] = out
    return obj


def metric_of(row, metric):
    """The compared number, and whether it was the requested metric or a fallback.

    An older reference written before `host_ns_self` existed falls back to `dev_ns`
    rather than failing, but the fallback is reported: silently comparing a
    different metric than the one asked for is how a gate becomes decorative.
    """
    if metric in row:
        return float(row[metric]), metric
    if metric != "dev_ns" and "dev_ns" in row:
        return float(row["dev_ns"]), "dev_ns"
    return None, metric


def fmt_us(ns):
    return "%.2f" % (ns / 1000.0)


# -------------------------------------------------------------------- gates ---
def compare(ref, cur, metric, tol_pct, min_us, allow_domain_change=False):
    """The whole comparison, as data. `--self-test` grades this same function."""
    r_dom = ref.get("clock_domain", "?")
    c_dom = cur.get("clock_domain", "?")
    if r_dom != c_dom and not allow_domain_change:
        raise ValueError(
            "clock domains differ: reference %s, current %s. Host intervals and "
            "device events are different clocks, so the comparison would be "
            "meaningless. Re-pin against a profile from the same clock, or pass "
            "--allow-domain-change if that is really what you want." % (r_dom, c_dom))

    min_ns = min_us * 1000.0
    verdicts = []
    judged = 0
    failed = 0

    for name, r_row in ref["_by_name"].items():
        r_ns, r_key = metric_of(r_row, metric)
        c_row = cur["_by_name"].get(name)
        v = {"component": name, "ref_ns": r_ns, "cur_ns": None, "delta_pct": None,
             "ref_key": r_key}
        if r_ns is None:
            v["verdict"] = "REF UNUSABLE"
            v["why"] = "the reference row has no %s and no dev_ns" % metric
            failed += 1
            verdicts.append(v)
            continue
        if c_row is None:
            v["verdict"] = "REMOVED"
            v["why"] = "in the reference, absent from this run: a vanished" \
                       " measurement is a regression, not a skip"
            failed += 1
            verdicts.append(v)
            continue
        c_ns, c_key = metric_of(c_row, metric)
        if c_ns is None:
            v["verdict"] = "UNUSABLE"
            v["why"] = "the current row has no %s" % metric
            failed += 1
            verdicts.append(v)
            continue
        v["cur_ns"] = c_ns
        v["cur_key"] = c_key
        v["delta_pct"] = (c_ns - r_ns) / r_ns * 100.0 if r_ns > 0 else (
            float("inf") if c_ns > 0 else 0.0)
        if r_ns < min_ns:
            v["verdict"] = "NOT JUDGED"
            v["why"] = "reference is below the %.1f us noise floor" % min_us
            verdicts.append(v)
            continue
        judged += 1
        if v["delta_pct"] > tol_pct:
            v["verdict"] = "REGRESSED"
            failed += 1
        elif v["delta_pct"] < -tol_pct:
            v["verdict"] = "improved"
        else:
            v["verdict"] = "ok"
        verdicts.append(v)

    for name, c_row in cur["_by_name"].items():
        if name not in ref["_by_name"]:
            c_ns, _ = metric_of(c_row, metric)
            verdicts.append({"component": name, "ref_ns": None, "cur_ns": c_ns,
                             "delta_pct": None, "verdict": "NEW",
                             "why": "not in the reference; informational, not a"
                                    " failure (a rename shows up as REMOVED + NEW)"})
    return verdicts, judged, failed


def print_report(ref, cur, ref_path, cur_path, verdicts, judged, failed, metric,
                 tol_pct, min_us):
    pin = ref.get("pin") or {}
    print("C21 profile diff")
    print("  reference  %s" % ref_path)
    print("             clock %s, detail %s, warmup %s, steps kept %s" % (
        ref.get("clock_domain"), ref.get("detail"), ref.get("warmup_steps"),
        ref.get("steps_kept")))
    if pin:
        print("             pinned   %s" % pin.get("pinned_at", "?"))
        print("             reason   %s" % pin.get("reason", "(none recorded)"))
        if pin.get("replaced_reason"):
            print("             replaced %s" % pin["replaced_reason"])
    else:
        print("             NO PIN BLOCK: this reference does not record why it is"
              " the reference. Re-pin with --reason to fix that.")
    print("  current    %s" % cur_path)
    print("             clock %s, detail %s, warmup %s, steps kept %s" % (
        cur.get("clock_domain"), cur.get("detail"), cur.get("warmup_steps"),
        cur.get("steps_kept")))
    print("  metric %s, tolerance +/-%.1f%%, rows under %.1f us not judged"
          % (metric, tol_pct, min_us))
    print()
    print("  %-20s %12s %12s %9s  %s" % ("component", "ref", "current", "delta",
                                          "verdict"))
    for v in sorted(verdicts, key=lambda x: (-(x["ref_ns"] or 0), x["component"])):
        r = fmt_us(v["ref_ns"]) if v["ref_ns"] is not None else "-"
        c = fmt_us(v["cur_ns"]) if v["cur_ns"] is not None else "-"
        d = ("%+.1f%%" % v["delta_pct"]) if v["delta_pct"] is not None else "-"
        note = ""
        if v.get("why"):
            note = "  (%s)" % v["why"]
        elif v.get("ref_key") and v["ref_key"] != metric:
            note = "  (read from %s: the requested metric is absent)" % v["ref_key"]
        print("  %-20s %10sus %10sus %9s  %s%s"
              % (v["component"], r, c, d, v["verdict"], note))
    print()
    print("  %d component(s) judged, %d failed" % (judged, failed))


def repin(ref_path, cur, cur_path, reason, metric, tol_pct, allow_domain_change):
    """Write `cur` as the new reference, with a recorded reason."""
    out = dict(cur)
    out.pop("_by_name", None)
    previous = None
    if os.path.isfile(ref_path):
        try:
            previous = (load_profile(ref_path).get("pin") or {}).get("reason")
        except ValueError:
            previous = None
    pin = {
        "reason": reason,
        "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "source": cur_path,
        "metric": metric,
        "tol_pct": tol_pct,
        "clock_domain": cur.get("clock_domain"),
        "detail": cur.get("detail"),
    }
    if previous:
        pin["replaced_reason"] = previous
    out["pin"] = pin
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print("re-pinned %s" % ref_path)
    print("  reason   %s" % reason)
    if previous:
        print("  replaced %s" % previous)
    print("  metric %s, tolerance +/-%.1f%%, clock %s"
          % (metric, tol_pct, cur.get("clock_domain")))
    return 0


# ---------------------------------------------------------------- self-test ---
def _mk(components, domain="host", detail="class", **kw):
    """A profile shaped exactly as load_profile() returns one, including the
    `_by_name` index, so the self-test grades the same function the CLI uses."""
    obj = {"clock_domain": domain, "detail": detail, "warmup_steps": 2,
           "steps_kept": 22, "components": components}
    obj.update(kw)
    obj["_by_name"] = {str(c["component"]): c for c in components}
    return obj


def self_test():
    """Grade the gate itself. No model, no GPU, no profiles on disk."""
    ok = True
    checks = 0

    def check(cond, what):
        nonlocal ok, checks
        checks += 1
        if not cond:
            ok = False
            print("      FAIL  %s" % what)

    # 1. inside tolerance -> quiet pass
    ref = _mk([{"component": "moe", "ops": 10, "dev_ns_self": 1_000_000.0},
               {"component": "qkv", "ops": 10, "dev_ns_self": 500_000.0}])
    cur = _mk([{"component": "moe", "ops": 10, "dev_ns_self": 1_050_000.0},
               {"component": "qkv", "ops": 10, "dev_ns_self": 500_000.0}])
    v, judged, failed = compare(ref, cur, DEFAULT_METRIC, 10.0, 50.0)
    check(failed == 0 and judged == 2, "5% rise on both rows must pass")

    # 2. beyond tolerance -> fail
    cur = _mk([{"component": "moe", "ops": 10, "dev_ns_self": 1_200_000.0},
               {"component": "qkv", "ops": 10, "dev_ns_self": 500_000.0}])
    v, judged, failed = compare(ref, cur, DEFAULT_METRIC, 10.0, 50.0)
    check(failed == 1, "a 20% rise on one row must fail exactly one row")
    check([x for x in v if x["component"] == "moe"][0]["verdict"] == "REGRESSED",
          "the failing row must be named REGRESSED")

    # 3. a vanished component is a failure, not a skip
    cur = _mk([{"component": "moe", "ops": 10, "dev_ns_self": 1_000_000.0}])
    v, judged, failed = compare(ref, cur, DEFAULT_METRIC, 10.0, 50.0)
    check(failed == 1, "a removed component must fail the gate")
    check([x for x in v if x["component"] == "qkv"][0]["verdict"] == "REMOVED",
          "and it must be reported as REMOVED")

    # 4. below the noise floor is not judged
    ref = _mk([{"component": "norm", "ops": 616, "dev_ns_self": 4000.0}])
    cur = _mk([{"component": "norm", "ops": 616, "dev_ns_self": 9000.0}])
    v, judged, failed = compare(ref, cur, DEFAULT_METRIC, 10.0, 50.0)
    check(failed == 0 and judged == 0, "a sub-floor row must not be judged")
    check(v[0]["verdict"] == "NOT JUDGED", "and it must say so")

    # 5. a clock-domain change is refused, not compared
    ref = _mk([{"component": "moe", "ops": 10, "dev_ns_self": 1_000_000.0}],
              domain="host")
    cur = _mk([{"component": "moe", "ops": 10, "dev_ns_self": 1_000_000.0}],
              domain="device")
    try:
        compare(ref, cur, DEFAULT_METRIC, 10.0, 50.0)
        check(False, "differing clock domains must raise")
    except ValueError as e:
        check("clock domains differ" in str(e), "and must name the reason")

    # 6. an older reference falls back to dev_ns, and says it did
    ref = _mk([{"component": "moe", "ops": 10, "dev_ns": 1_000_000.0}])
    cur = _mk([{"component": "moe", "ops": 10, "dev_ns_self": 1_000_000.0}])
    v, judged, failed = compare(ref, cur, DEFAULT_METRIC, 10.0, 50.0)
    check(failed == 0 and v[0]["ref_key"] == "dev_ns",
          "a reference without the metric must fall back and report the fallback")

    # 7. a re-pin with no reason is refused
    with tempfile.TemporaryDirectory() as d:
        cur_path = os.path.join(d, "cur.json")
        with open(cur_path, "w", encoding="utf-8") as f:
            json.dump(_mk([{"component": "moe", "ops": 10, "dev_ns_self": 1.0}]), f)
        argv = ["--cur", cur_path, "--ref", os.path.join(d, "ref.json"), "--repin"]
        try:
            args = parse_args(argv)
            rc = main_with(args)
            check(rc == 2, "a re-pin without --reason must exit 2")
        except SystemExit as e:
            check(e.code == 2, "a re-pin without --reason must exit 2 (got %s)" % e.code)
        # 8. with a reason it writes a pin block that records the why
        argv = argv + ["--reason", "self-test: the numbers moved for a stated reason"]
        rc = main_with(parse_args(argv))
        check(rc == 0, "a re-pin with a reason must succeed")
        with open(os.path.join(d, "ref.json"), "r", encoding="utf-8") as f:
            pinned = json.load(f)
        check((pinned.get("pin") or {}).get("reason") ==
              "self-test: the numbers moved for a stated reason",
              "the pin block must record the reason")

    print("profile_diff self-test: %d check(s), %s"
          % (checks, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ---------------------------------------------------------------------- cli ---
def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Diff two C21 JSON profiles and gate on a regression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit 0 pass, 1 regression, 2 usage error, 3 not comparable.")
    ap.add_argument("--ref", metavar="REF.json",
                    help="the pinned reference profile")
    ap.add_argument("--cur", metavar="CUR.json",
                    help="the profile to compare against the reference")
    ap.add_argument("--metric", default=DEFAULT_METRIC,
                    choices=["dev_ns_self", "host_ns_self", "dev_ns", "host_ns"],
                    help="the compared number (default %s)" % DEFAULT_METRIC)
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL_PCT,
                    help="regression tolerance in percent (default %g)"
                         % DEFAULT_TOL_PCT)
    ap.add_argument("--min-us", type=float, default=50.0,
                    help="rows whose reference self time is below this many "
                         "microseconds are not judged (default 50)")
    ap.add_argument("--allow-domain-change", action="store_true",
                    help="compare despite differing clock domains (you probably "
                         "do not want this)")
    ap.add_argument("--repin", action="store_true",
                    help="write --cur as the new --ref; requires --reason")
    ap.add_argument("--reason", metavar="TEXT",
                    help="why the new reference is the right reference")
    ap.add_argument("--self-test", action="store_true",
                    help="grade the gate itself and exit")
    return ap.parse_args(argv)


def main_with(args):
    if args.self_test:
        return self_test()
    if not args.cur:
        print("error: --cur is required", file=sys.stderr)
        return 2
    if args.tol < 0:
        print("error: --tol must not be negative", file=sys.stderr)
        return 2
    if args.repin and not args.reason:
        print("error: --repin requires --reason. A pin that does not say WHY it "
              "moved\n       is indistinguishable from a pin that was moved to "
              "silence a gate.", file=sys.stderr)
        return 2
    if not args.repin and not args.ref:
        print("error: --ref is required (or use --repin)", file=sys.stderr)
        return 2
    try:
        cur = load_profile(args.cur)
        if args.repin:
            return repin(args.ref, cur, args.cur, args.reason, args.metric,
                         args.tol, args.allow_domain_change)
        ref = load_profile(args.ref)
        verdicts, judged, failed = compare(ref, cur, args.metric, args.tol,
                                          args.min_us, args.allow_domain_change)
    except ValueError as e:
        print("error: %s" % e, file=sys.stderr)
        return 3
    print_report(ref, cur, args.ref, args.cur, verdicts, judged, failed,
                 args.metric, args.tol, args.min_us)
    if failed:
        print()
        print("PROFILE REGRESSION -- %d component(s) regressed or vanished beyond "
              "the %.1f%% tolerance." % (failed, args.tol))
        print("Re-pin ONLY with a reason:")
        print("  python tools/c21/profile_diff.py --ref %s --cur %s --repin "
              "--reason \"...\"" % (args.ref, args.cur))
        return 1
    print()
    print("PROFILE OK -- no judged component regressed beyond %.1f%%." % args.tol)
    return 0


def main(argv=None):
    return main_with(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
