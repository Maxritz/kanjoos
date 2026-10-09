#!/usr/bin/env python3
"""Write the tokenizer.ggml.pre fixtures under tests/fixtures/tok/.

Why these files exist
---------------------
The engine's tokenizer dispatches on `tokenizer.ggml.pre`. Until it did, every
file was tokenized with the qwen2 rule -- including files that declare a
different one -- and that does not crash: it changes token ids and nothing else.
The two rules this engine implements differ in exactly one place (a combining
mark joins the letter run under qwen35 and is punctuation under qwen2), so the
difference is invisible except on text that carries a mark and a merge that
would cross the resulting boundary.

These fixtures are that text, and nothing else. Every fixture is byte-for-byte
the same vocabulary and merge table, and differs ONLY in the value of
`tokenizer.ggml.pre`, so a test that reads two of them isolates the dispatch.

The vocabulary
--------------
256 byte tokens in GPT-2 byte order, then two merged tokens:

    id 256 = "a" + the two bytes of U+0301, first half
    id 257 = id 256 + the second byte of U+0301

    merges[0] = "a <first half>"    rank 0
    merges[1] = "<id256> <second half>"  rank 1

Together they merge the byte sequence 61 CC 81 down to a single symbol. Whether
that merge may fire depends on the pre-tokeniser: under qwen35 the mark joins
the letter run, so "a" and the mark are in ONE chunk and both merges fire;
under qwen2 the mark is its own chunk, so the boundary sits between them and
neither merge can cross it. Same file, same merges, different ids.

    qwen2  "a\u0301b" -> [97, 204, 129, 98]   (4 tokens, no merge fired)
    qwen35 "a\u0301b" -> [257, 98]            (2 tokens, both merges fired)

The whitespace-run cases
------------------------
Four more merged tokens, and this part has nothing to do with `pre` -- it is the
same under both rules, and the fixtures carry it because they are the only
vocabularies in the tree with a hand-chosen merge table:

    id 258 = the two bytes of U+00A0 as one symbol        merges[2] = "Â ł"
    id 259 = the two bytes of U+00B4 as one symbol        merges[3] = "Â ´"
    id 260 = U+3000's first two bytes                     merges[4] = "ã Ģ"
    id 261 = the three bytes of U+3000 as one symbol      merges[5] = "ãĢ Ģ"

A whitespace run is where the reference regex's `\\s+(?!\\S)` gives back ONE
CHARACTER so that a leading space can still be taken by the next alternative.
This engine gave back one BYTE, which for a character wider than ASCII cuts the
character in half and forbids its own merge:

    "U+00A0 U+00B4"  engine was [194, 160, 259], must be [258, 259]
    "U+3000 U+00B4"  engine was [260, 128, 259], must be [261, 259]

Measured on the real file (Qwen3-MOE-4x0.6B) against llama.cpp 2026-10-08 as
[126, 254, 28111] vs [4102, 28111] and [1277, 222, 28111] vs [22441, 28111];
this fixture is the same case with ids small enough for a test to state exactly.

The generator computes the byte map with the same algorithm the C++ side uses
(tools/../src/tokenizer/tokenizer.cpp, ByteMap), so the ids above are derived
rather than guessed, and the test asserts them exactly.

Usage
-----
    python tools/make_tok_fixtures.py            # write tests/fixtures/tok/
    python tools/make_tok_fixtures.py --check    # verify on disk, exit 1 on drift
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

try:
    import gguf
except ImportError:  # pragma: no cover - environment problem, not a test failure
    sys.exit("error: the 'gguf' python package is required (pip install gguf)")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "tests", "fixtures", "tok")

# The `pre` names this engine honours, and the file each one gets. The C++ side
# exposes the same list (knj::known_pre_names()) and derives the filename from
# the name, so a name added there without a fixture here fails the test with a
# "run the generator" message instead of silently going untested.
KNOWN_PRE_NAMES = [
    "qwen2",
    "deepseek-r1-qwen",
    "kormo",
    "f2llmv2",
    "megrez",
    "grok-2",
    "stablelm2",
    "hunyuan",
    "solar-open",
    "qwen35",
]

# The refusals. `absent` writes no key at all. Measured against llama.cpp
# 8b4b355 (2026-10-08): it warns "missing pre-tokenizer type, using: 'default'"
# and runs the DEFAULT pre-type, whose BPE `regex_exprs` is a FOUR-expression
# list -- a punctuation run, the GPT-2 word rule, `\p{N}+` and
# `[0-9][0-9][0-9]` -- not the empty list an older llama.cpp revision used (the
# comment here said so until tools/tok_pre_rules.py read the actual source). It
# is a third rule all the same, which is why the engine refuses it by name. An
# unknown NAME is a hard error in llama.cpp ("unknown pre-tokenizer type").
#
# There is NO empty-string fixture. `GGUFWriter.add_tokenizer_pre("")` is silently
# dropped, so that file comes out with no key at all -- byte-identical to
# pre_absent (measured: same sha256). A present-but-empty `pre` is one branch in
# resolve_pre() together with an absent key, and the test reaches it through
# pre_absent rather than through a fixture that only pretends to be a second case.
NEGATIVE_FIXTURES = {
    "pre_absent": None,
    "pre_unknown": "llama3",
}

MARK = "\u0301"  # COMBINING ACUTE ACCENT: Mn, so a mark under either rule
LETTER_A = b"a"
LETTER_B = b"b"


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte -> codepoint permutation. Mirrors ByteMap in tokenizer.cpp."""
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapped: dict[int, str] = {}
    for b in printable:
        mapped[b] = chr(b)
    n = 0
    for b in range(256):
        if b not in mapped:
            mapped[b] = chr(256 + n)
            n += 1
    return mapped


BM = bytes_to_unicode()


def enc(bs: bytes) -> str:
    """The byte-level-BPE spelling of a raw byte string."""
    return "".join(BM[b] for b in bs)


# The whitespace-run characters. U+00A0 (2 bytes) and U+3000 (3 bytes) are the
# two that the real Qwen vocabulary has a merge across, which is what makes the
# difference visible at all: see the docstring.
NBSP = "\u00a0".encode("utf-8")
ACUTE = "\u00b4".encode("utf-8")
IDEO = "\u3000".encode("utf-8")


def build_vocab() -> tuple[list[str], list[str]]:
    """Return (tokens, merges): 256 byte tokens then the merged results."""
    tokens = [BM[b] for b in range(256)]
    a = enc(LETTER_A)
    mark = MARK.encode("utf-8")
    assert len(mark) == 2, "U+0301 must be two UTF-8 bytes"
    half1, half2 = enc(bytes(mark[:1])), enc(bytes(mark[1:]))

    # id 256: 'a' + the mark's first byte.  id 257: that + the mark's second byte.
    t256 = a + half1
    t257 = t256 + half2
    # ids 258-261: one whitespace character as a single symbol, so a chunk that
    # holds the whole character merges to ONE token and a chunk that holds half
    # of it cannot.
    t258 = enc(NBSP)
    t259 = enc(ACUTE)
    t260 = enc(bytes(IDEO[:2]))
    t261 = enc(IDEO)
    tokens += [t256, t257, t258, t259, t260, t261]

    merges = [
        f"{a} {half1}",                        # rank 0 -> id 256
        f"{t256} {half2}",                     # rank 1 -> id 257
        f"{enc(bytes(NBSP[:1]))} {enc(bytes(NBSP[1:]))}",    # rank 2 -> id 258
        f"{enc(bytes(ACUTE[:1]))} {enc(bytes(ACUTE[1:]))}",  # rank 3 -> id 259
        f"{enc(bytes(IDEO[:1]))} {enc(bytes(IDEO[1:2]))}",   # rank 4 -> id 260
        f"{t260} {enc(bytes(IDEO[2:]))}",                    # rank 5 -> id 261
    ]
    return tokens, merges


def expected_ids() -> dict[str, list[int]]:
    """The ids each rule must produce for "a" + U+0301 + "b", derived from the
    vocabulary above rather than transcribed from a run."""
    tokens, _ = build_vocab()
    idx = {t: i for i, t in enumerate(tokens)}
    a = enc(LETTER_A)
    mark = MARK.encode("utf-8")
    half1, half2 = enc(bytes(mark[:1])), enc(bytes(mark[1:]))
    b = enc(LETTER_B)
    # The whitespace-run cases, derived from the same vocabulary. `rule` does not
    # enter these: the whitespace alternatives are identical under qwen2 and
    # qwen35, so they are asserted on the qwen2 fixture and must hold for both.
    nbsp, acute, ideo = enc(NBSP), enc(ACUTE), enc(IDEO)
    return {
        # qwen2: three chunks -- "a", the mark alone, "b". No merge may cross a
        # chunk boundary, so the mark stays two byte tokens.
        "qwen2": [idx[a], idx[half1], idx[half2], idx[b]],
        # qwen35: one letter run -- both merges fire inside it.
        "qwen35": [idx[a + half1 + half2], idx[b]],
        # One whitespace character + acute: the character's own merge fires, so
        # two tokens -- never the three a byte-counted backtrack produced.
        "nbsp_acute": [idx[nbsp], idx[acute]],
        "ideo_acute": [idx[ideo], idx[acute]],
        # Controls: the ASCII cases the byte count got right all along. A run of
        # two spaces followed by a word gives [' '] then [' hi'], and a single
        # space followed by the acute gives [' '] then the merged acute.
        "two_spaces_hi": [idx[enc(b" ")], idx[enc(b" ")], idx[enc(b"h")], idx[enc(b"i")]],
        "space_acute": [idx[enc(b" ")], idx[acute]],
    }


def write_fixture(path: str, pre: "str | None") -> None:
    tokens, merges = build_vocab()
    w = gguf.GGUFWriter(path, arch="kanjoos-tok-fixture", use_temp_file=False)
    w.add_name("kanjoos tokenizer.ggml.pre fixture")
    w.add_tokenizer_model("gpt2")
    if pre is not None:
        w.add_tokenizer_pre(pre)
    w.add_token_list(tokens)
    w.add_token_types([gguf.TokenType.NORMAL] * len(tokens))
    w.add_token_merges(merges)
    # Additive, and not part of what these fixtures pin; present so the file is
    # shaped like a real tokenizer if it is opened by other tools.
    w.add_bos_token_id(0)
    w.add_eos_token_id(1)
    w.add_add_bos_token(False)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def fixture_plan() -> dict[str, "str | None"]:
    """filename -> the `pre` value to write (None = no key at all)."""
    plan = {f"pre_{name}.gguf": name for name in KNOWN_PRE_NAMES}
    for stem, pre in NEGATIVE_FIXTURES.items():
        plan[f"{stem}.gguf"] = pre
    return plan


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="do not write; verify the on-disk fixtures and exit 1 on drift")
    args = ap.parse_args()

    plan = fixture_plan()
    if not args.check:
        os.makedirs(OUT_DIR, exist_ok=True)
        # Prune: a renamed or dropped case must not leave a stale .gguf behind for
        # a test to keep passing against.
        for stale in sorted(os.listdir(OUT_DIR)):
            if stale.endswith(".gguf") and stale not in plan:
                os.remove(os.path.join(OUT_DIR, stale))
                print(f"  pruned   {stale}  (no longer in the plan)")

    eids = expected_ids()
    print(f"byte map        : {len(BM)} entries, 256 byte tokens then 6 merged")
    print(f"expected ids    : qwen2 {eids['qwen2']}   qwen35 {eids['qwen35']}")
    for case in ("nbsp_acute", "ideo_acute", "two_spaces_hi", "space_acute"):
        print(f"                  {case:15s} {eids[case]}")
    print(f"fixtures        : {len(plan)} in {os.path.relpath(OUT_DIR, ROOT)}")

    if args.check:
        # Deterministic writer + identical inputs => identical bytes, so a hash
        # mismatch is drift, not noise. The vocabulary is rebuilt in memory, so
        # the check needs no scratch file.
        import tempfile
        bad = 0
        for name, pre in sorted(plan.items()):
            path = os.path.join(OUT_DIR, name)
            if not os.path.exists(path):
                print(f"  MISSING  {name}  (run without --check)")
                bad += 1
                continue
            with tempfile.TemporaryDirectory() as td:
                fresh = os.path.join(td, name)
                write_fixture(fresh, pre)
                if sha256(fresh) != sha256(path):
                    print(f"  DRIFT    {name}  (run without --check)")
                    bad += 1
                else:
                    print(f"  ok       {name}  {sha256(path)[:16]}")
        if bad:
            print(f"FAILED: {bad} fixture(s) out of date")
            return 1
        print("all fixtures match what this generator would write")
        return 0

    for name, pre in sorted(plan.items()):
        path = os.path.join(OUT_DIR, name)
        write_fixture(path, pre)
        shown = "<no key>" if pre is None else (repr(pre) if pre == "" else f"'{pre}'")
        print(f"  {name:32s} pre={shown:24s} {os.path.getsize(path):6d} B  {sha256(path)[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
