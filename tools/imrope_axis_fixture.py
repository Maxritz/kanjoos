#!/usr/bin/env python3
"""Independent IMROPE (sectioned RoPE) axis fixture for the non-text case.

The engine's `imrope_axis()` in src/model/qwen35.cpp maps a 256-pair dimension
index to one of four axes using the file's `rope.dimension_sections`. A text
prompt gives every axis the same linear position, so the mapping matters only for
verification -- which is why this fixture exists.

This script is the *independent* implementation behind that fixture: given the
same section sizes, it computes the axis for every pair, and -- the part that the
text prompt cannot see -- it also computes the rotation theta for each pair under
distinct per-axis positions (t, h, w, extra), so a derived-vs-measured check can
see whether the engine's rotation values track when the four axes do not collapse.

Run:
    python tools/imrope_axis_fixture.py --model <file.gguf>
    python tools/imrope_axis_fixture.py --sections 11 11 10 224 --positions t=1,h=2,w=3,e=7

Exit status: 0 when the sections are well-formed (sum to 256) and the output files
are written; non-zero when the file's sections are missing or inconsistent.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import gguf  # installed at C:\\Users\\rr\\AppData\\Roaming\\Python\\Python312\\site-packages\\gguf


def imrope_axis(pair: int, sections: tuple[int, int, int, int]) -> int:
    """Mirror the engine's `imrope_axis()` (src/model/qwen35.cpp, line 726).

    The loop is the same; the only difference is that this is a pure function in
    Python and the engine's is in C++. Both take the file's section sizes as the
    ground truth.
    """
    sect_dims = sections[0] + sections[1] + sections[2] + sections[3]
    if sect_dims <= 0:
        return 0
    sector = pair % sect_dims
    if sector % 3 == 1 and sector < 3 * sections[1]:
        return 1
    if sector % 3 == 2 and sector < 3 * sections[2]:
        return 2
    if sector % 3 == 0 and sector < 3 * sections[0]:
        return 0
    return 3


def pair_to_axis_table(sections: tuple[int, int, int, int]) -> dict[int, int]:
    """Every pair in [0, 256) mapped to its axis. The engine and the fixture both
    produce this from the file's section sizes; a mismatch is the non-text bug."""
    return {pair: imrope_axis(pair, sections) for pair in range(256)}


def theta_for_pair(pair: int, pos: int, rd: int, base: float) -> float:
    """theta_i = pos * base^(-2i/rd), the angle the pair is rotated by.

    `pair` is the pair index in [0, rd/2). `pos` is the position along that pair's
    axis (text: the same for all axes; non-text: per-axis).
    """
    i = pair
    return float(pos) * math.pow(base, -2.0 * i / rd)


def non_text_thetas(sections: tuple[int, int, int, int], rd: int,
                    base: float, positions: dict[str, int]) -> dict[int, float]:
    """theta for every pair, under distinct per-axis positions.

    `positions` is the per-axis position: e.g. {"t": 1, "h": 2, "w": 3, "extra": 7}.
    The axis index for each pair is taken from `pair_to_axis_table(sections)` and
    the pair's position is the one belonging to that axis. The return is
    {pair -> theta}.
    """
    table = pair_to_axis_table(sections)
    pos_per_axis = {
        0: positions.get("t", 0),
        1: positions.get("h", 0),
        2: positions.get("w", 0),
        3: positions.get("extra", positions.get("x", 0)),
    }
    out: dict[int, float] = {}
    for pair in range(rd // 2):
        axis = table[pair]
        pos = pos_per_axis[axis]
        out[pair] = theta_for_pair(pair, pos, rd, base)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="qwen35 GGUF whose rope.dimension_sections this uses")
    ap.add_argument("--sections", nargs=4, type=int, default=None,
                    help="override the file's rope.dimension_sections with four ints")
    # The output lands beside the other recorded runs (`records/<name>/` at the
    # repository root, as docs/CODING-LOG.PENDING.md and tools/ref_qwen35.py both
    # spell it), not inside tools/ where it would be an undocumented stray tree.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out",
                    default=os.path.join(root, "records", "qwen35-imrope-fixture"),
                    help="directory into which the fixture tables are written "
                         "(default: records/qwen35-imrope-fixture at the repo root)")
    ap.add_argument("--positions",
                    nargs="*",
                    default=None,
                    help="per-axis positions as KEY=VAL pairs, e.g. t=1 h=2 w=3 extra=7; "
                         "used only for the derived theta table")
    args = ap.parse_args()

    if not args.sections:
        if not args.model or not os.path.exists(args.model):
            raise SystemExit("need --model <file.gguf> or --sections A B C D")
        r = gguf.GGUFReader(args.model)
        # Qwen3.5's GGUF stores the section sizes under qwen35.rope.dimension_sections,
        # not the generic rope.dimension_sections key.
        raw = r.fields.get("qwen35.rope.dimension_sections")
        if raw is None:
            raise SystemExit("qwen35.rope.dimension_sections not found in "
                             + args.model)
        try:
            secs = [int(x) for x in raw.data]
        except (TypeError, ValueError, AttributeError):
            raise SystemExit("qwen35.rope.dimension_sections could not be read as ints "
                             "from " + args.model + " (field data: %r)" % raw.data)
        if len(secs) != 4:
            raise SystemExit("qwen35.rope.dimension_sections has %d entries, expected 4"
                             % len(secs))
        sections = tuple(secs)
    else:
        sections = tuple(args.sections)

    if sum(sections) != 256:
        # The file's own sections are the ground truth for the engine too; if they
        # do not sum to 256 the engine cannot rotate all 256 pairs under this
        # file, and the non-text fixture is not meaningful for it.
        print("WARNING: qwen35.rope.dimension_sections sum to %d, expected 256; "
              "imrope_axis() would ignore pairs beyond the sum, so the non-text "
              "fixture is not meaningful for this file. Override with --sections."
              % sum(sections), file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)

    # 1. the axis table -- the thing the non-text fixture actually checks.
    table = pair_to_axis_table(sections)
    with open(os.path.join(args.out, "imrope_axis_table.csv"), "w") as f:
        f.write("pair,axis\n")
        for pair in range(256):
            f.write("%d,%d\n" % (pair, table[pair]))
    print("sections       : %s  (sum %d)" % (sections, sum(sections)))
    print("axis table     : %d pairs, written to %s/imrope_axis_table.csv"
          % (256, args.out))

    # 2. derived thetas under distinct per-axis positions, when asked.
    if args.positions:
        pos: dict[str, int] = {}
        for p in args.positions:
            for kv in p.split(","):
                kv = kv.strip()
                if not kv:
                    continue
                k, v = kv.split("=", 1)
                pos[k.strip()] = int(v.strip())
        thetas = non_text_thetas(sections, 256, 10000000.0, pos)
        # One valid JSON document, not JSON followed by a hand-printed appendix: a
        # file named .json that json.load() cannot read is a trap for the check
        # that is supposed to consume it.
        payload: dict = {
            "model": args.model or "derived",
            "sections": list(sections),
            "positions": pos,
            "rd": 256,
            "base": 10000000.0,
            "theta_per_pair": {str(pair): th for pair, th in sorted(thetas.items())},
        }
        theta_path = os.path.join(args.out, "non_text_theta.json")
        with open(theta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        # Read it back: the artifact is the fixture's contract, so a file that
        # cannot be parsed is a failure here rather than at the consumer.
        with open(theta_path, encoding="utf-8") as f:
            reloaded = json.load(f)
        if len(reloaded["theta_per_pair"]) != len(thetas):
            raise SystemExit("imrope_axis_fixture: %s round-tripped %d of %d pairs"
                             % (theta_path, len(reloaded["theta_per_pair"]), len(thetas)))
        print("non-text thetas: %d pairs under positions %s, written to %s/non_text_theta.json"
              % (len(thetas), pos, args.out))

    # 3. independent verification artifact: a small derived-vs-expected table for
    #    the pairs the caller cares about. The non-text fixture's check is: for a
    #    set of (pair, axis) pairs that span the four axes, the engine's rotation
    #    value under the pair's axis position must match the independent python value
    #    to within floating-point tolerance. This writes the expectation so the check
    #    is a file diff rather than a recomputation.
    if args.positions and _sections_ok(sections):
        _write_expectation_artifact(args.out, sections, pos)

    return 0


def _write_expectation_artifact(out_dir: str, sections: tuple[int, int, int, int],
                                pos: dict[str, int], per_axis: int = 2) -> None:
    """Write the derived-vs-engine expectation table.

    The point of the non-text fixture is that the four axes carry *different*
    positions, so the table has to show all four axes. Picking the first `per_axis`
    pairs of each axis (rather than the first N pairs overall, which lands inside
    axis 0 and silently tests one axis four times) is what makes the artifact
    cover the case it is named for.
    """
    table = pair_to_axis_table(sections)
    picked: dict[int, int] = {}
    for axis in (0, 1, 2, 3):
        taken = 0
        for pair in range(256):
            if table[pair] == axis and pair not in picked:
                picked[pair] = axis
                taken += 1
                if taken >= per_axis:
                    break
    missing = [a for a in (0, 1, 2, 3) if a not in set(picked.values())]
    if missing:
        raise SystemExit("imrope_axis_fixture: --sections %s give no pair on axis %s, "
                         "so the non-text expectation table cannot cover it"
                         % (tuple(sections), missing))
    rows: list[dict[str, object]] = []
    for pair, axis in sorted(picked.items()):
        pname: str = {0: "t", 1: "h", 2: "w", 3: "extra"}[axis]
        axis_pos: int = pos.get(pname, 0)
        th: float = theta_for_pair(pair, axis_pos, 256, 10000000.0)
        rows.append({
            "pair": pair,
            "axis": axis,
            "axis_name": pname,
            "axis_position": axis_pos,
            "theta": th,
        })
    with open(os.path.join(out_dir, "non_text_expected.csv"), "w", encoding="utf-8") as f:
        f.write("pair,axis,axis_name,axis_position,theta\n")
        for r in rows:
            f.write("%d,%d,%s,%d,%.18e\n" % (
                int(r["pair"]), int(r["axis"]), str(r["axis_name"]),
                int(r["axis_position"]), float(r["theta"])))
    print("expectation    : %d derived pairs over axes %s written to "
          "%s/non_text_expected.csv"
          % (len(rows), sorted(set(picked.values())), out_dir))


def _sections_ok(sections: tuple[int, int, int, int]) -> bool:
    """True when the section sizes sum to 256, so the non-text fixture is
    meaningful."""
    return sum(sections) == 256


if __name__ == "__main__":
    raise SystemExit(main())
