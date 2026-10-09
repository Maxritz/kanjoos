#!/usr/bin/env python3
"""Re-derive the engine's pre-tokeniser dispatch from the reference's ACTUAL regexes.

Why this exists
---------------
The engine implements two pre-tokeniser rules (`tokenizer.cpp`, `PreRule::Qwen2`
and `PreRule::Qwen35`) and a table of `tokenizer.ggml.pre` names that select one
of them. That table was written from the *prose* of llama.cpp's name map -- a
comment saying which names "are grouped with QWEN2" -- and prose drifts: the
comment cited `llama_vocab_pre_type_from_name`, a function that does not exist in
the revision this machine measures against, and the table omitted `megrez`,
which that revision maps to `LLAMA_VOCAB_PRE_TYPE_QWEN2` -- the same regex the
engine already implements.

A name that selects a rule the engine does not implement is a refusal, and a
refusal is safe. A name that selects a rule it *does* implement and is missing
from the table is a model the engine cannot load for no reason -- and a name
whose reference rule is *not* what the table says is silent wrong ids.

This tool removes the prose from the loop. It reads the reference's own source
and derives, mechanically:

  * the `tokenizer.ggml.pre` name -> pre-type map (`llama_vocab::impl::load`);
  * the pre-type -> `regex_exprs` table (the `llm_tokenizer_bpe` constructor);
  * so, for every name the reference knows: which regex list its pre-type runs.

Then it compares that derivation with the engine's table, and asserts:

  1. every name the engine claims is known to the reference;
  2. every claimed name's reference regex list is byte-identical to one of the
     two lists the engine emulates, and the engine's rule for that name is the
     one that corresponds (a `stablelm2` that ran the qwen35 rule would pass a
     "name exists" check and fail this one);
  3. the table is COMPLETE: every reference name whose regex list is one of the
     two, and whose branch sets no tokenisation-affecting flag, is in the table;
  4. no claimed name's branch sets a flag the engine does not implement
     (`clean_spaces` is detokenisation-only and is reported, not required);
  5. both regex strings are present verbatim inside the oracle binary -- the
     shared library that `llama-tokenize.exe` loads, i.e. the library that
     produced the measured token ids. That is the link from "the source says X"
     to "the binary that measured it contains X".

A leg that cannot run (no reference tree, no oracle library) is reported as NOT
RUN and exits 3 -- never as a pass.

    python tools/tok_pre_rules.py                     # report + check
    python tools/tok_pre_rules.py --emit <path>       # also write the expectation

`--emit` writes the file the C++ dispatch test reads
(tests/fixtures/tok/pre_rules.expected): one `name rule` line per name, with the
provenance in a comment header. The test asserts SET EQUALITY against
`knj::known_pre_names()`, so a name added to either side without the other fails
the suite instead of drifting.

Exit codes:
    0   every configured leg ran and agreed
    1   a disagreement (missing name, wrong rule, drift, an unimplemented flag)
    3   a leg did NOT run (reference source or oracle library missing) -- not a pass
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REFERENCE = "C:/Users/rr/OneDrive/Desktop/dxl/llama.cpp"

# One backslash, spelled without an escape: the regex patterns in this file are
# full of them and a mis-escaped comparison would silently match nothing.
BS = chr(92)

# Flags in a name's branch that change what the tokenizer PRODUCES. The engine
# implements the regexes and nothing else, so a name whose branch sets one of
# these is not a name the engine can honour, however identical its regex is.
ENCODING_FLAGS = (
    "escape_whitespaces",
    "ignore_merges",
    "byte_encode",
    "add_space_prefix",
    "add_sep",
    "treat_whitespace_as_suffix",
)
# Reported per name, not required: clean_up_tokenization_spaces is applied in
# llama.cpp's DETOKENIZE (the `if (clean_spaces)` block in the detokenize loop),
# not in tokenize, so it cannot change an id list -- which is what this tool and
# the cross-check compare. A name whose branch leaves it at the BPE default
# `true` is noted so a decode-side difference is not mistaken for a tokenizer one.
DECODE_FLAGS = ("clean_spaces",)

# The two pre-types whose regex lists the engine emulates. The mapping from
# pre-type to list is parsed; these names are the *anchor* (the engine's
# `pre_rule_name()` prints exactly these strings).
ANCHORS = {"qwen2": "LLAMA_VOCAB_PRE_TYPE_QWEN2",
           "qwen35": "LLAMA_VOCAB_PRE_TYPE_QWEN35"}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def c_unquote(lit: str) -> str:
    """Decode the body of a C/C++ string literal.

    A regex string in the reference is written with doubled backslashes: two
    characters in the file are one character at runtime. The literals may also
    contain an escaped quote (the punctuation class starts with one). Any other
    escape is refused rather than guessed, because a mangled regex compares equal
    to nothing and the failure would be reported as a missing name.
    """
    simple = {BS: BS, '"': '"', "n": "\n", "t": "\t", "r": "\r"}
    out: list[str] = []
    i = 0
    while i < len(lit):
        c = lit[i]
        if c == BS and i + 1 < len(lit):
            n = lit[i + 1]
            if n not in simple:
                raise SystemExit(
                    "tok_pre_rules: unsupported escape %s%s in a regex literal "
                    "(%r...); refusing rather than comparing a string this tool "
                    "cannot decode" % (BS, n, lit[:48]))
            out.append(simple[n])
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _brace_span(text: str, open_idx: int) -> tuple[int, int]:
    """(inner_start, inner_end) of the {...} group at `open_idx`, ignoring braces
    inside string literals -- every regex here contains a brace inside a `p{L}`
    character class, so a }-bounded search stops inside the first literal."""
    depth = 0
    i = open_idx
    instr = False
    while i < len(text):
        c = text[i]
        if instr:
            if c == BS:
                i += 2
                continue
            if c == '"':
                instr = False
        elif c == '"':
            instr = True
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return open_idx + 1, i
        i += 1
    raise SystemExit("tok_pre_rules: unbalanced braces in a regex_exprs initializer")


def parse_regex_table(ref_text: str) -> dict[str, dict]:
    """pre-type -> {regexes: [...], byte_encode: bool|None}."""
    m = re.search(r"switch \(vocab\.get_pre_type\(\)\) \{", ref_text)
    if not m:
        raise SystemExit(
            "tok_pre_rules: the reference has no `switch (vocab.get_pre_type())` "
            "-- this parser is written for the revision it was measured against; "
            "refusing to guess")
    start = m.end()
    end = ref_text.find("std::vector<std::string> regex_exprs;", start)
    if end < 0:
        raise SystemExit("tok_pre_rules: could not find the end of the pre-type "
                         "switch (no `std::vector<std::string> regex_exprs;`)")
    region = ref_text[start:end]
    # Line comments are dropped BEFORE the literals are read: every case group in
    # this revision carries its original tokenizer.json regex in a comment, and a
    # parser that read the comments would compare the engine against a regex
    # llama.cpp does not run.
    stripped = "\n".join(re.sub(r"//.*$", "", ln) for ln in region.splitlines())

    table: dict[str, dict] = {}
    for chunk in stripped.split("break;"):
        labels = re.findall(r"\bcase\s+(LLAMA_VOCAB_PRE_TYPE_\w+):", chunk)
        if re.search(r"(?<!\w)default\s*:", chunk):
            # the switch's `default:` catches every pre-type without an explicit
            # case, including PRE_TYPE_DEFAULT; name it as the enum the name map uses
            labels.append("LLAMA_VOCAB_PRE_TYPE_DEFAULT")
        if not labels:
            continue
        mm = re.search(r"regex_exprs\s*=\s*" + BS + "{", chunk)
        lits: list[str] = []
        if mm:
            a, z = _brace_span(chunk, mm.end() - 1)
            lits = [c_unquote(x) for x in
                    re.findall('"((?:[^"' + BS + BS + ']|' + BS + BS + '.)*)"',
                               chunk[a:z])]
        enc = re.search(r"byte_encode\s*=\s*(true|false);", chunk)
        for lab in labels:
            if lab in table:
                raise SystemExit("tok_pre_rules: label %s appears twice" % lab)
            table[lab] = {"regexes": lits,
                          "byte_encode": None if enc is None else enc.group(1) == "true"}
    return table


def parse_name_map(ref_text: str) -> dict[str, dict]:
    """`tokenizer.ggml.pre` name -> {pre_type, flags, line}."""
    a = ref_text.find("if (type == LLAMA_VOCAB_TYPE_BPE) {")
    if a < 0:
        raise SystemExit("tok_pre_rules: no BPE branch in the reference")
    b = ref_text.find("} else if (type == LLAMA_VOCAB_TYPE_SPM) {", a)
    if b < 0:
        b = len(ref_text)
    region = ref_text[a:b]

    marks: list[tuple[int, str]] = [(m.start(), m.group(1)) for m in
                                    re.finditer(r'tokenizer_pre == "([^"]*)"', region)]
    empty = region.find("if (tokenizer_pre.empty())")
    if empty >= 0:
        marks.append((empty, ""))  # the absent key: llama.cpp warns and uses DEFAULT
    marks.sort()

    out: dict[str, dict] = {}
    flag_re = re.compile(r"\b(%s)\s*=\s*(true|false);" %
                         "|".join(ENCODING_FLAGS + DECODE_FLAGS))
    for pos, name in marks:
        body_start = region.find(") {", pos)
        if body_start < 0:
            raise SystemExit("tok_pre_rules: could not find the body of the branch "
                             "for %r" % name)
        nxt = [x for x in (region.find("} else if (", body_start),
                           region.find("} else {", body_start)) if x > body_start]
        body = region[body_start:min(nxt) if nxt else len(region)]
        pt = re.search(r"pre_type = (LLAMA_VOCAB_PRE_TYPE_\w+);", body)
        if not pt:
            continue  # a condition that falls through to a later branch's body
        if name in out:
            raise SystemExit("tok_pre_rules: name %r appears twice in the map" % name)
        out[name] = {"pre_type": pt.group(1),
                     "flags": dict(flag_re.findall(body)),
                     "line": line_of(ref_text, a + pos)}
    return out


def parse_engine_table() -> dict[str, str]:
    """The engine's own `kPreNames`: name -> rule name (lowercased)."""
    path = os.path.join(ROOT, "src", "tokenizer", "tokenizer.cpp")
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as e:
        raise SystemExit("tok_pre_rules: cannot read %s: %s" % (path, e))
    m = re.search(r"kPreNames\[\]\s*=\s*\{(.*?)\};", text, re.S)
    if not m:
        raise SystemExit("tok_pre_rules: no kPreNames[] table in %s" % path)
    out: dict[str, str] = {}
    for name, rule in re.findall(r'\{\s*"([^"]+)"\s*,\s*PreRule::(\w+)\s*\}',
                                 m.group(1)):
        out[name] = rule.lower()
    if not out:
        raise SystemExit("tok_pre_rules: kPreNames[] parsed to zero entries")
    return out


def find_oracle(reference: str) -> str | None:
    """The shared library the oracle executable loads -- the one that contains
    the vocab implementation. Found by content, not by name, so a rebuild that
    renames it cannot make the provenance leg silently pass."""
    bin_dir = os.path.join(reference, "build", "bin")
    if not os.path.isdir(bin_dir):
        return None
    for fn in sorted(os.listdir(bin_dir)):
        if not fn.lower().endswith(".dll"):
            continue
        p = os.path.join(bin_dir, fn)
        try:
            with open(p, "rb") as fh:
                blob = fh.read()
        except OSError:
            continue
        if b"unknown pre-tokenizer type" in blob:
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference",
                    default=os.environ.get("KNJ_LLAMA_CPP", DEFAULT_REFERENCE),
                    help="a llama.cpp checkout (env: KNJ_LLAMA_CPP; default: %s)"
                         % DEFAULT_REFERENCE)
    ap.add_argument("--emit", metavar="PATH",
                    help="write the name/rule expectation the C++ dispatch test reads")
    args = ap.parse_args()

    failures: list[str] = []
    notrun: list[str] = []

    ref_src = os.path.join(args.reference, "src", "llama-vocab.cpp")
    if not os.path.exists(ref_src):
        print("NOT RUN: no reference source at %s" % ref_src)
        print("         (pass --reference <llama.cpp checkout>; without it this "
              "derivation cannot be made, and it is not a pass)")
        return 3

    with open(ref_src, "rb") as fh:
        ref_bytes = fh.read()
    ref_text = ref_bytes.decode("utf-8", errors="replace")
    ref_hash = sha256_bytes(ref_bytes)

    name_map = parse_name_map(ref_text)
    regex_table = parse_regex_table(ref_text)

    missing_types = sorted({v["pre_type"] for v in name_map.values()
                            if v["pre_type"] not in regex_table})
    if missing_types:
        failures.append("pre-type(s) referenced by the name map but absent from the "
                        "regex switch: %s" % ", ".join(missing_types))

    table = parse_engine_table()
    print("reference     : %s" % ref_src)
    print("                sha256 %s" % ref_hash)
    print("engine table  : %d name(s): %s" % (len(table), ", ".join(sorted(table))))
    print()

    # -- the anchors, and the derivation of every name's regex list ------------
    anchor_list: dict[str, str] = {}
    for rule, ptype in sorted(ANCHORS.items()):
        info = regex_table.get(ptype)
        if info is None:
            failures.append("anchor pre-type %s is not in the reference's regex "
                            "switch" % ptype)
            continue
        if len(info["regexes"]) != 1:
            failures.append("anchor %s has %d regex expressions; this engine emulates "
                            "ONE regex per rule, so a multi-expression list is a "
                            "different rule and cannot be compared" %
                            (ptype, len(info["regexes"])))
            continue
        anchor_list[rule] = info["regexes"][0]

    if len(anchor_list) != len(ANCHORS):
        print("no verdict: the anchors could not be derived")
        for f in failures:
            print("  FAIL %s" % f)
        return 1

    list_to_rule = {v: k for k, v in anchor_list.items()}
    for r in sorted(anchor_list):
        s = anchor_list[r]
        print("anchor regex  : %-7s len=%3d sha256 %s"
              % (r, len(s), sha256_bytes(s.encode())[:16]))

    derived: dict[str, str] = {}
    for name, info in sorted(name_map.items()):
        lists = regex_table[info["pre_type"]]["regexes"]
        if len(lists) == 1 and lists[0] in list_to_rule:
            derived[name] = list_to_rule[lists[0]]

    print("derived       : %d name(s) resolve to one of the two regexes: %s"
          % (len(derived), ", ".join(sorted(derived))))
    print()
    print("%-20s %-8s %-34s %s" % ("name", "rule", "pre-type", "branch flags"))
    for name, rule in sorted(derived.items(), key=lambda kv: (kv[1], kv[0])):
        info = name_map[name]
        fl = ", ".join("%s=%s" % (k, v) for k, v in sorted(info["flags"].items())) or "(none)"
        print("%-20s %-8s %-34s %s" % (name, rule, info["pre_type"], fl))
    print()

    # -- 1. every claimed name is known to the reference, with the right rule ---
    for name in sorted(table):
        if name not in name_map:
            failures.append("engine claims %r, which the reference's name map does "
                            "not contain (%s)" % (name, ref_src))
            continue
        info = name_map[name]
        lists = regex_table.get(info["pre_type"], {}).get("regexes", [])
        if len(lists) != 1 or lists[0] not in list_to_rule:
            failures.append("engine claims %r -> %s, whose regex list is not one of "
                            "the two the engine emulates (%d expression(s))"
                            % (name, info["pre_type"], len(lists)))
            continue
        derived_rule = list_to_rule[lists[0]]
        if table[name] != derived_rule:
            failures.append("engine maps %r to rule %r but the reference's %s runs "
                            "the %r regex" % (name, table[name], info["pre_type"],
                                              derived_rule))
            continue
        for flag in ENCODING_FLAGS:
            if info["flags"].get(flag) not in (None, "false"):
                failures.append("engine claims %r, but its reference branch sets "
                                "%s=%s, which changes what the tokenizer produces and "
                                "is not implemented" % (name, flag, info["flags"][flag]))

    # -- 2. completeness -------------------------------------------------------
    for name, rule in sorted(derived.items()):
        info = name_map[name]
        blocking = [f for f in ENCODING_FLAGS
                    if info["flags"].get(f) not in (None, "false")]
        if blocking:
            print("note          : %r -> %s runs the %s regex but sets %s; the engine "
                  "refuses it, correctly" % (name, rule, info["pre_type"],
                                             ", ".join(blocking)))
            continue
        if name not in table:
            failures.append("the reference maps %r to %s (line %d), whose regex is the "
                            "%s rule the engine already implements, and the engine's "
                            "table omits it -- a file declaring that pre is refused "
                            "for no reason" % (name, info["pre_type"], info["line"], rule))

    # -- 3. provenance: the oracle binary carries these exact strings ----------
    oracle = find_oracle(args.reference)
    oracle_ok = False
    if oracle is None:
        notrun.append("the oracle library (a build/bin/*.dll containing the vocab "
                      "implementation) was not found under %s, so the source->binary "
                      "provenance leg did NOT run" % args.reference)
    else:
        with open(oracle, "rb") as fh:
            blob = fh.read()
        oracle_ok = True
        for r in sorted(anchor_list):
            if anchor_list[r].encode("utf-8") + b"\x00" not in blob:
                oracle_ok = False
                failures.append("the %s regex parsed from the source is NOT present in "
                                "%s -- the source and the binary that measured this "
                                "machine's token ids disagree" % (r, oracle))
        print("oracle        : %s" % oracle)
        print("                both regex strings present verbatim: %s"
              % ("yes" if oracle_ok else "no"))

    # -- the expectation the C++ test reads ------------------------------------
    if args.emit:
        rel = os.path.relpath(args.emit, ROOT).replace(os.sep, "/")
        lines = [
            "# %s" % rel,
            "# Re-derived from the reference's ACTUAL regexes -- do not edit by hand.",
            "# generator  : python tools/tok_pre_rules.py --emit %s" % rel,
            "# reference  : %s" % ref_src,
            "#              sha256 %s" % ref_hash,
            "# anchors    : qwen2  sha256 %s" % sha256_bytes(anchor_list["qwen2"].encode()),
            "#              qwen35 sha256 %s" % sha256_bytes(anchor_list["qwen35"].encode()),
            "# oracle     : %s" % (oracle or "<not found: provenance leg NOT run>"),
            "#",
            "# The dispatch test asserts SET EQUALITY against knj::known_pre_names(),",
            "# so a name on one side without the other fails the suite.",
        ]
        for name, rule in sorted(derived.items(), key=lambda kv: (kv[1], kv[0])):
            info = name_map[name]
            if any(info["flags"].get(f) not in (None, "false") for f in ENCODING_FLAGS):
                continue
            lines.append("%s %s" % (name, rule))
        with open(args.emit, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        print("emitted       : %s (%d name(s))" % (args.emit, len(lines) - 11))

    print()
    for item in notrun:
        print("NOT RUN: %s" % item)
    for f in failures:
        print("FAIL: %s" % f)
    if failures:
        print("VERDICT: FAIL -- the engine's table and the reference's actual regexes "
              "disagree in %d place(s)." % len(failures))
        return 1
    if notrun:
        print("VERDICT: INCOMPLETE -- every configured check passed, but a leg did not "
              "run; this is not a pass.")
        return 3
    print("VERDICT: PASS -- every name the engine claims runs exactly the regex it "
          "says, the table is complete for those two regexes, and the oracle binary "
          "carries both strings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
