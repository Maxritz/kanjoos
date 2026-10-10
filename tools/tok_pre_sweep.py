#!/usr/bin/env python3
"""Sweep the non-whitespace pre-tokeniser alternatives against the reference.

Only the whitespace-run alternative (`\\s+(?!\\S)`) was ever swept against
llama.cpp's behaviour (the separator sweep). The other six alternatives of the
QWEN2/QWEN35 regex are emulated in `Tokenizer::pretokenize` from their prose.
This tool closes that by running the reference's ACTUAL pattern strings --
extracted mechanically from the llama.cpp source, never retyped -- through a
Unicode regex engine over crafted inputs, one per alternative and boundary.

Usage:
    python tools/tok_pre_sweep.py --reference <llama.cpp checkout>
        prints `input -> [chunks]` per rule for review; the C++ gate
        (tests/unit/test_tok_pre_sweep.cpp) hardcodes the reviewed splits.
    python tools/tok_pre_sweep.py --emit <dir>
        also writes the machine-readable expectation next to the log.

The patterns need no transcription: they are the string literals from the
QWEN2/QWEN35 cases of the `llm_tokenizer_bpe` constructor, unescaped once
(`\\\\p` -> `\\p`). A leg that cannot run exits 3 (never a pass).
"""
import argparse
import json
import os
import re
import sys

try:
    import regex as rx
except ImportError:
    raise SystemExit("tok_pre_sweep: the `regex` module is missing, NOT RUN (3)")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REFERENCE = "C:/Users/rr/OneDrive/Desktop/dxl/llama.cpp"

CASES = {
    "LLAMA_VOCAB_PRE_TYPE_QWEN2": "qwen2",
    "LLAMA_VOCAB_PRE_TYPE_QWEN35": "qwen35",
}

# (label, input, rules) -- each input targets one alternative or boundary.
# Rules are a subset of {"qwen2", "qwen35"}; both where they must agree.
INPUTS = [
    ("contr-lower", "don't", ("qwen2", "qwen35")),
    ("contr-upper", "DON'T", ("qwen2", "qwen35")),
    ("contr-ve", "they've", ("qwen2",)),
    ("contr-ll", "I'll", ("qwen2",)),
    ("contr-re", "we're", ("qwen2",)),
    ("contr-non", "rock 'n' roll", ("qwen2",)),
    ("contr-trail", "cats'", ("qwen2",)),
    ("contr-dbl", "''s", ("qwen2",)),
    ("contr-s", "'s", ("qwen2",)),
    ("letter-plain", "hello", ("qwen2", "qwen35")),
    ("letter-space", "Hello World", ("qwen2",)),
    ("letter-digit-split", "a1b", ("qwen2",)),
    ("letter-digit-run", "abc123", ("qwen2",)),
    ("digit-letter", "123abc", ("qwen2",)),
    ("digit-prefix-no", "1a", ("qwen2",)),
    ("letter-lead-space", " a", ("qwen2",)),
    ("underscore", "under_score", ("qwen2",)),
    ("hyphen", "co-op", ("qwen2",)),
    ("number-run", "123", ("qwen2",)),
    ("number-dot", "3.14", ("qwen2",)),
    ("punct-run", "!!!", ("qwen2",)),
    ("punct-lead-space", " !!!", ("qwen2",)),
    ("punct-crlf", "!?\n!\n", ("qwen2",)),
    ("punct-word", "a!b", ("qwen2",)),
    ("punct-emdash", "\u2014", ("qwen2",)),
    ("punct-cjk", "\u3002", ("qwen2",)),
    ("punct-cjk-word", "a\u3002b", ("qwen2",)),
    ("newline-mid", "a\nb", ("qwen2",)),
    ("newline-space", "a \n b", ("qwen2",)),
    ("newline-crlf", "a\r\n\r\nb", ("qwen2",)),
    ("cjk-letters", "\u65e5\u672c\u8a9e", ("qwen2", "qwen35")),
    ("emoji", "\U0001F680", ("qwen2",)),
    ("emoji-word", "a\U0001F680b", ("qwen2",)),
    ("mark-qwen2", "a\u0301b", ("qwen2",)),
    ("mark-qwen35", "a\u0301b", ("qwen35",)),
    ("trail-one", "hi ", ("qwen2",)),
    ("trail-two", "hi  ", ("qwen2",)),
    ("lead-two", "  hi", ("qwen2",)),
    ("mid-two", "a  b", ("qwen2",)),
]


def extract_pattern(src, case):
    """The regex string literal of one QWEN case, unescaped once.

    The block also carries the original tokenizer.json pattern in a `//`
    comment: comment lines are dropped first, so the literal found is the
    compiled one, never the comment. Verified by asserting the literal
    contains the case-sensitive `[sS]` classes the active patterns spell.
    """
    text = open(src, encoding="utf-8").read()
    i = text.find("case %s:" % case)
    if i < 0:
        raise SystemExit("tok_pre_sweep: %s not found in %s" % (case, src))
    j = text.find("regex_exprs = {", i)
    k = text.find("};", j)
    code = "\n".join(l for l in text[j:k].splitlines()
                      if not l.strip().startswith("//"))
    lit = re.search(r'"((?:[^"\\]|\\.)*)"', code)
    if not lit:
        raise SystemExit("tok_pre_sweep: no string literal in %s block" % case)
    pat = lit.group(1).encode().decode("unicode_escape")
    if "[sS]" not in pat:
        raise SystemExit("tok_pre_sweep: %s literal looks like the commented "
                         "original, not the compiled pattern" % case)
    return pat


def split(pat, s):
    """llama.cpp's loop: repeated search, first match wins, never stall."""
    rx_pat = rx.compile(pat)
    out, pos = [], 0
    while pos < len(s):
        m = rx_pat.search(s, pos)
        if not m or m.start() != pos:
            # llama.cpp advances one char on no-match at pos; every
            # alternative here consumes, so this is unreachable -- assert it.
            raise SystemExit("tok_pre_sweep: no match at %d of %r" % (pos, s))
        out.append(m.group(0))
        pos = m.end()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=os.environ.get("KNJ_LLAMA_CPP",
                                                           DEFAULT_REFERENCE))
    ap.add_argument("--emit", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    src = os.path.join(args.reference, "src", "llama-vocab.cpp")
    if not os.path.isfile(src):
        print("tok_pre_sweep: no reference checkout at %s: NOT RUN" % src)
        return 3
    pats = {rule: extract_pattern(src, case) for case, rule in CASES.items()}
    for rule, pat in pats.items():
        print("pattern[%s] = %r" % (rule, pat))
    print()
    rows = []
    for label, s, rules in INPUTS:
        for rule in rules:
            chunks = split(pats[rule], s)
            rows.append({"label": label, "rule": rule, "input": s,
                         "chunks": chunks})
            print("%-18s %-6s %r -> %r" % (label, rule, s, chunks))
    if args.emit:
        os.makedirs(args.emit, exist_ok=True)
        with open(os.path.join(args.emit, "pre_sweep.expected.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"patterns": pats, "rows": rows}, fh, indent=2,
                      ensure_ascii=False)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
