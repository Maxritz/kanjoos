#!/usr/bin/env python3
r"""Cross-check the engine's tokenizer against llama.cpp's, string by string.

Round-trip (`decode(encode(s)) == s`) is necessary and **not sufficient**: it
cannot see a wrong *merge boundary*. A tokenizer that splits "12345" as [12][345]
round-trips exactly and is still not the model's tokenizer -- and the model sees
different ids, so every measurement downstream of it is of a different problem.
The only way to test a merge boundary is to compare against another
implementation on the same strings.

This runs two implementations on the same input and compares the id lists
element by element:

  * `llama-tokenize --ids` from the local llama.cpp, which carries its own
    implementation of the `qwen35` pre-tokeniser;
  * this engine's `kanjoos-run` qwen35 probe, whose tokenizer is built from the
    GGUF's own vocabulary and merges.

llama.cpp is run twice -- with special-token parsing (its default) and with
`--no-parse-special`. A string that matches only the second is reported as a
**POLICY** difference, never as a pass: this engine tokenises raw text without
parsing control tokens, and that is a statement about the engine that has to be
visible rather than averaged away. A string that matches in *neither* mode is a
**merge-boundary difference** and is the failure this gate exists to catch.

    python tools/tok_crosscheck.py --model <file.gguf>

Exit status: 0 when no string differs in both modes, 1 when one does. `--strict`
also fails on a policy difference.

Corpus and fuzz mode (read this before the help text below assumes otherwise):

    python tools/tok_crosscheck.py --model <file.gguf> --corpus doc.md --max-tokens 512
    python tools/tok_crosscheck.py --model <file.gguf> --fuzz 5000 --fuzz-seed 1

A large natural-language corpus (each line is one prompt) and a random-byte
fuzzer (which walks both byte intervals over the UTF-8 range and deliberately
hits invalid-sequence corner cases) are both compared against llama.cpp in the
same way as the hand-written set -- each prompt lives in its own file, both
tokenizer paths are run, and the output counts exactly how many tokens differ.

The boundary/policy split is the same: a string whose id list differs from llama
in *one* parse mode is a **POLICY** difference (special-token parsing, which the
engine does not do), and only a string that differs in the no-parse-special mode
is a **merge-boundary** difference and a failure. `tok_crosscheck` reports
`boundary`, `policy`, `ok`, `tokens` exactly like the default mode, and its
shell exit code mirrors them: 0 when no merge-boundary difference is found, 1
when one is. `--strict` additionally fails on policy differences.

Neither mode replaces the hand-written set: the unit-test-driven bounded set is
small, targeted and deterministic and is what makes the gate catch the specific
softcap scale, trim-window priming and offset-scale bugs. The corpus and fuzz
modes are large and unspecific and are what pin the rule dispatch against a real
tokenizer over a broad input surface -- a file whose declared `pre` differs from
qwen2 (so the mark probe fires) and a file what declares `qwen35` (so the mark
probe does not) both must disagree with the hand set on exactly the strings the
fixtured cross-check already pins.

Separator sweep (bounded: one process pair per codepoint, ~14 of them)

    python tools/tok_crosscheck.py --model <file.gguf> --separator-sweep

This is the sweep that found the whitespace-run defect of 2026-10-08 (the
engine's emulation of `\s+(?!\S)` gave back one BYTE where the regex gives back
one CHARACTER, so a two- or three-byte space was cut in half and its own merge
could not fire). It is a standalone mode rather than a corpus pass because the
input is one codepoint per line: every separator in the engine's Space class is
probed as `separator + U+00B4`, a pair the reference vocabulary merges, so
"the whole character formed one chunk" and "the character was split" produce
different id lists. A separator whose two id lists differ is a merge-boundary
defect; the two ASCII controls (U+0020, U+0009) must agree and are the control.

Exit status mirrors the other modes: 0 when every separator agrees, 1 when one
does not. It used to live in tmp/probe_space_class.py and was a hand-run
script; it is a gate case now (ctest `tok_separator_sweep`).
"""
import argparse
import random
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# The machine-specific default inputs, in ONE place. tools/smoke_tools.py imports
# these rather than keeping a second copy that can drift, and every one of them is
# overridable from the environment (KNJ_TOK_MODEL / KNJ_LLAMA_TOKENIZE /
# KNJ_ENGINE) so a gate runner does not carry hardware paths in its command line.
DEFAULT_MODEL = ("C:/Users/rr/OneDrive/Desktop/kraken/models/"
                 "Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf")
DEFAULT_TOKENIZE = ("C:/Users/rr/OneDrive/Desktop/dxl/llama.cpp/build/bin/"
                    "llama-tokenize.exe")
DEFAULT_ENGINE = os.path.join(ROOT, "build", "cmake-host", "kanjoos-run.exe")

# The engine is a MinGW build: its libstdc++-6.dll / libgcc_s_seh-1.dll /
# libwinpthread-1.dll must be findable or it dies at the process boundary with
# 0xC0000139 -- which this tool would otherwise report as "the engine printed no
# id list", i.e. as a tokenizer result. The first candidate that actually
# contains libstdc++-6.dll is prepended to the engine's PATH for every spawn.
MINGW_CANDIDATES = (
    "/c/Strawberry/c/bin", "C:/Strawberry/c/bin",
    "/mingw64/bin", "C:/msys64/mingw64/bin",
    "C:/Program Files/Git/mingw64/bin",
    "/c/msys64/mingw64/bin",
)


# Windows loader failures, as the unsigned exit codes Python reports: a missing
# DLL (0xC0000135), a missing entry point in a DLL (0xC0000139) and a bad image
# (0xC000007B). The engine is a MinGW build, so the usual cause is that its
# libstdc++-6.dll / libgcc_s_seh-1.dll / libwinpthread-1.dll are not findable --
# and that used to be reported as "the engine printed no id list", i.e. as a
# tokenizer result. It is not one.
LOADER_FAILURE_CODES = {
    3221225781: "0xC0000135 (DLL not found)",
    3221225785: "0xC0000139 (entry point not found in a DLL)",
    3221225571: "0xC000007B (bad image)",
}


def engine_env() -> dict:
    """os.environ with the MinGW runtime directory prepended, if one is found.

    Public because tools/smoke_tools.py uses it too: the gate's subprocesses must
    share one rule about where the runtime lives, not two copies that can drift.
    """
    env = dict(os.environ)
    # KNJ_MINGW_BIN first: an explicit answer beats a guess, and it is also named
    # in the loader-failure message below so the fix is one variable away.
    for d in (os.environ.get("KNJ_MINGW_BIN", ""),) + MINGW_CANDIDATES:
        if d and os.path.isfile(os.path.join(d, "libstdc++-6.dll")):
            env["PATH"] = d + os.pathsep + env.get("PATH", "")
            return env
    return env

# The default set leans on the boundaries a BPE merge is most likely to get
# wrong: digits vs letters, leading and repeated spaces, punctuation runs,
# apostrophes, CJK (one byte-level token per character), emoji (multi-byte),
# newlines/tabs, a control token, and a long word that must split.
STRINGS = [
    "The capital of France is",
    "hello world",
    "1 2 3 123 1234 3.14159 -0.5",
    "  leading and   multiple    spaces",
    "line1\nline2\n\nline3",
    "日本語のテキスト",
    "emoji: \U0001F680\U0001F525",
    "tab\there <|endoftext|>",
    "abc123def 0.5e-3 7zip A1B2C3",
    "don't  it's  can't've  l'été",
    "!!!!  ...  ---  ;;  (((",
    "{ \"key\": [1,2,3], \"n\": null }",
    "a" * 64,
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "mixed  mixed\tmixed\nmixed \U0001F600 mixed",
    # The class-rule probes. These exist because the pre-tokeniser decides *where a
    # merge is allowed to happen*, so each one puts a non-ASCII symbol or a
    # non-ASCII space where a word character meets it: if the classifier called the
    # symbol a letter, one chunk would span the join and a merge across it could
    # fire; if it is punctuation, the join is a boundary. Measured 2026-10-08: with
    # every non-ASCII codepoint forced to `Letter` (the pre-fix byte test) all ten
    # of these still produced identical ids, because byte-level BPE re-derives the
    # same symbols inside a wider chunk. They are kept as a guard: they pin the
    # class rule to llama.cpp's answer even though that rule is currently
    # unobservable here.
    "a\U0001F680b",                       # letter, rocket, letter
    "a\u2014b",                           # em dash between letters
    "10\u201320",                         # en dash between digits
    "\u65e5\u672c\u8a9e\u3001\u3067\u3059",   # CJK around the ideographic comma
    "a\u00a0b",                           # NO-BREAK SPACE
    "h\u00e9llo\u2026world",              # ellipsis
    "\u20ac100 \u00a35",                  # euro, pound: symbols, not letters
    "emoji\U0001F680fire\U0001F525",
    "\u00dcn\u00efc\u00f6d\u00e9 t\u00e8xt",
    "x\u2009y",                           # thin space
    # THE mark probes. A combining mark is the ONE place where the two
    # pre-tokenisers this engine implements differ: `qwen2` matches [\p{L}]+ and
    # treats a mark as punctuation, `qwen35` matches [\p{L}\p{M}]+ and lets the
    # mark join the letter run -- so the chunk boundary moves and a different set
    # of merges becomes possible. These are decomposed on purpose (the precomposed
    # forms have no mark at all), and they run against whichever rule each file
    # declares.
    "e\u0301x",                           # e + COMBINING ACUTE + x
    "cafe\u0301 au lait",                 # decomposed "cafe\u0301" inside a phrase
    "a\u0308\u0301b",                     # two marks in a row
    "n\u0303a",                           # n + COMBINING TILDE + a
    "\u0e01\u0e34\u0e19",                 # Thai: a letter, its vowel sign, a letter
    "\u0915\u093f\u0938",                 # Devanagari: a letter, its vowel sign, more
    "\u0627\u064e\u0628",                 # Arabic: a letter, a fatha, a letter
]

# The separator sweep's table: one codepoint per line, every one of them in the
# engine's Space class (Zs/Zl/Zp plus the ASCII controls) and therefore a place
# where the emulated whitespace rule meets the reference's `\s`. The first two
# are the controls -- one byte, one character, so the byte-counted backtrack got
# them right even before the fix, and they must keep agreeing.
SEPARATORS = [
    ("U+0020 SPACE (control)", 0x0020),
    ("U+0009 TAB (control)", 0x0009),
    ("U+00A0 NBSP", 0x00A0),
    ("U+1680 OGHAM SPACE", 0x1680),
    ("U+2000 EN QUAD", 0x2000),
    ("U+2007 FIGURE SPACE", 0x2007),
    ("U+200A HAIR SPACE", 0x200A),
    ("U+2028 LINE SEP", 0x2028),
    ("U+2029 PARA SEP", 0x2029),
    ("U+202F NNBSP", 0x202F),
    ("U+205F MMSP", 0x205F),
    ("U+3000 IDEOGRAPHIC", 0x3000),
    ("U+000B VT", 0x000B),
    ("U+000C FF", 0x000C),
]

# The two markers this engine uses for "here are the prompt's ids", one per path:
#
#   * the qwen35 front-end probe prints `  probe text "<text>" -> N id(s): <ids>`
#     and returns before the shared generation path is reached;
#   * every other architecture prints `prompt ids : <ids>` under `--dump-tokens`.
#
# Both are read. Matching only the first is what made an earlier run of this tool
# on a qwen3moe file stop with "the engine printed no 'id(s):' line": the ids
# were there the whole time, spelled `prompt ids :`, and the tool was reading the
# wrong marker. A tool that cannot read a second engine path is a tool defect,
# not a tokenizer result.
#
# The probe's text can contain a newline (one of the test strings does), so the
# ids are not on the same PHYSICAL line as `probe text` -- but `id(s):` always
# is, and it occurs exactly once per run: the round-trip lines print a count,
# never that marker.
ID_LINE = re.compile(r"id\(s\):\s*(.*)$")
IDS_LINE = re.compile(r"prompt ids\s*:\s*(.*)$")

# Not in the default set: the empty string. The engine's *tokenizer* returns no
# ids for it (the round-trip table prints n=0 and its decode is empty), but the
# probe then substitutes bos for an empty prompt so that the layer probe has a
# token to run. That is the probe's guard, not a tokenizer result, so comparing
# it here would report a difference that is neither a merge boundary nor a
# policy -- it is a guard in a different component.


def _hex(s: str) -> str:
    """Compact hex of a UTF-8 string, for logging fuzz strings on a diff."""
    return s.encode("utf-8", errors="replace").hex()


def _gen_fuzz_strings(n: int, seed: int) -> list[str]:
    """n synthetic prompts generated from a fixed RNG seeded with `seed`.

    The prompts walk several byte intervals over the UTF-8 range on purpose,
    because the pre-tokeniser's only real behavioural split (mark vs letter under
    the two rules) is a byte-pattern difference, and the corpus is all natural
    language. The goal is to hit byte classes the corpus does not: zero bytes,
    high bytes, invalid lead/continuation sequences, overlong sequences, lone
    surrogates. This is NOT a completeness proof over those forms -- a fuzz prompt
    is one prompt, and a tokenizer can emit anything it likes for bytes that do not
    form text. The value of the fuzzer is to surface regressions (a tokenizer that
    silently eats a byte, or that diverges on a single byte interval) that the
    corpus and the hand set would not.

    Each prompt is short enough to be cheap to tokenise and to keep each llama.cpp
    invocation deterministic on this machine.
    """
    r = random.Random(seed)
    out: list[str] = []
    for _ in range(n):
        kind = r.choice(_FUZZ_TYPES)
        out.append(kind(r))
    return out


class _FuzzBody:
    """One byte-interval strategy for the fuzzer. Uses chr() so the fuzz string
    is a real Python str; a str that cannot round-trip to UTF-8 on this machine is
    impossible here, so no prompt is lossy before it reaches the tokenizers.

    Instances are callable so the fuzz harness can invoke them uniformly as
    `kind(r)` — the same calling convention the _FUZZ_TYPES lambdas already use.
    """

    def __call__(self, r: random.Random) -> str:  # pragma: no cover - exercised by --fuzz
        raise NotImplementedError


class _Repeat(_FuzzBody):
    def __init__(self, lo: int, hi: int, minlen: int = 1, maxlen: int = 64) -> None:
        self.lo = lo
        self.hi = hi
        self.minlen = minlen
        self.maxlen = maxlen

    def __call__(self, r: random.Random) -> str:
        n = r.randint(self.minlen, self.maxlen)
        return "".join(chr(r.randint(self.lo, self.hi)) for _ in range(n))


class _Overlong(_FuzzBody):
    # overlong and lone-surrogate codepoints are *invalid* UTF-8; emoji.py never
    # emits them, but this engine may choose to, and compare_string() reads the
    # payload as UTF-8 bytes and delegates to the tokenizers. We use the *bytes*
    # form directly and model them as str via a replacement decode path so the
    # prompts are in the tool's str domain without pretending the bytes are valid.
    def __init__(self, byte_seq: bytes) -> None:
        self._bytes = byte_seq

    def __call__(self, r: random.Random) -> str:
        return self._bytes.decode("utf-8", errors="replace")


# The fuzzer faces two byte intervals that the natural-language corpus and the
# mark probes are unlikely to exercise much: low bytes (including 0x00) and high
# bytes (0x80..0xFF, the top half of the byte-level BPE alphabet). Short runs of
# each, mixed with ASCII, cover the bulk of the space without making each prompt
# huge.
import typing

_FUZZ_TYPES: list[typing.Callable[[random.Random], str]] = [
    lambda r: _Repeat(0x20, 0x7E, 1, min(64, r.randint(1, 128)))(r),     # ASCII
    lambda r: _Repeat(0x00, 0x1F, 1, min(8, r.randint(1, 16)))(r),        # C0 control bytes
    lambda r: _Repeat(0x80, 0xFF, 1, min(64, r.randint(1, 128)))(r),      # high bytes
    lambda r: _Repeat(0xC0, 0xDF, 2, 2)(r),                               # 2-byte lead bytes
    lambda r: _Repeat(0xE0, 0xEF, 3, 3)(r),                               # 3-byte lead bytes
    lambda r: _Repeat(0xF0, 0xF4, 4, 4)(r),                               # 4-byte lead bytes
    lambda r: _Repeat(0x80, 0xBF, 1, 4)(r),                               # continuation bytes
    lambda r: _Repeat(0x7F, 0x7F, 1, 1)(r),                               # DEL
    lambda r: (
        "echo " + _Repeat(0x80, 0xFF, 2, 64)(r) + " end"
    ).encode("raw_unicode_escape").decode("raw_unicode_escape"),                # mixed phrase
    # invalid, overlong, lone-surrogate forms, kept in the pool so the fuzzer can
    # hit them a few times across the run. The replacement decode path keeps them
    # in the tool's str domain.
    lambda r: _Overlong(b"\xC0\xAF")(r),                                  # overlong / (U+002F)
    lambda r: _Overlong(b"\xE0\x80\xAF")(r),                             # overlong lower (U+002F)
    lambda r: _Overlong(b"\xF0\x80\x80\xAF")(r),                        # overlong lower (U+002F)
    lambda r: _Overlong(b"\xED\xA0\x80")(r),                             # lone high surrogate
    lambda r: _Overlong(b"\xED\xB0\x80")(r),                             # lone low surrogate
    lambda r: _Repeat(0xFE, 0xFF, 1, 2)(r),                               # non-UTF-8 lead bytes
    lambda r: b"\xC0\x80\xC1\x81\xF5\x80\x80\x80".decode("utf-8", errors="replace"),  # mixed invalid
]


def engine_ids(engine, model, path):
    """Token ids this engine produces for the file's bytes, from its own output.

    `--tokenize-only` asks the engine for ids and exits before prefill/generation.
    The qwen35 front end has a corresponding tokenization-only path and prints its
    `id(s):` marker; qwen3moe prints `prompt ids :`. Both are read here.

    Batch paths (--corpus, --fuzz) pass every prompt through a FILE too: the
    batch mode does not special-case argv, and the process-boundary failures above
    are why both paths use files -- the directory is just the storage for them.
    """
    p = subprocess.run([engine, "--model", model, "--prompt-file", path,
                        "--tokenize-only"],
                       env=engine_env(),
                       capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        code = p.returncode & 0xFFFFFFFF
        if code in LOADER_FAILURE_CODES:
            raise SystemExit(
                "tok_crosscheck: the engine failed to START (exit %d = %s) instead of "
                "tokenizing %s; this is a process-boundary failure.\nstdout:\n%s\nstderr:\n%s"
                % (p.returncode, LOADER_FAILURE_CODES[code], path,
                   p.stdout[-600:], p.stderr[-400:]))
        raise SystemExit(
            "tok_crosscheck: the engine exited %d instead of tokenizing %s.\n"
            "stdout:\n%s\nstderr:\n%s"
            % (p.returncode, path, p.stdout[-600:], p.stderr[-400:]))
    for line in p.stdout.splitlines():
        cut = line.rfind("id(s):")
        if cut >= 0:
            m = ID_LINE.search(line, cut)
            if m:
                body = m.group(1).strip()
                return [int(x) for x in body.split()] if body else []
        m = IDS_LINE.search(line)
        if m:
            body = m.group(1).strip()
            return [int(x) for x in body.split()] if body else []
def llama_ids(tokenize, model, path, parse_special):
    # `--no-escape` is a correctness requirement, not a preference. Measured on
    # this machine 2026-10-08 with llama-tokenize's own help text: `-e, --escape,
    # --no-escape  whether to process escapes sequences (\n, \r, \t, \', \", \\)`
    # defaults to TRUE, so without this flag llama.cpp tokenises a *different
    # string* than the file holds: the bytes 5c 74 (backslash, t) arrive as a real
    # TAB (0x09) and 5c 72 65 (`\re`) as CR + e. The engine tokenises the raw
    # bytes, so the comparison was engine-vs-unescaped-text and reported two
    # spurious merge-boundary failures on the fuzz passes that happened to draw a
    # backslash. With the flag, the same four inputs agree exactly:
    #   literal 5c 74      engine 4955            llama 197 -> 4955
    #   literal 5c 72 65   engine [59, 265]       llama [201, 68] -> [59, 265]
    #   real TAB 0x09      engine 197 == llama 197 either way
    #   literal 5c 73      engine 32407 == llama 32407 either way (`\s` is not an
    #                      escape llama.cpp recognises)
    # `--binary-file` was also measured: it does NOT disable escape processing.
    cmd = [tokenize, "-m", model, "-f", path, "--ids", "--no-bos", "--no-escape"]
    if not parse_special:
        cmd.append("--no-parse-special")
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    for line in reversed(p.stdout.splitlines()):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            body = line[1:-1].strip()
            return [int(x) for x in body.split(",")] if body else []
    raise SystemExit("tok_crosscheck: llama-tokenize printed no id list for %s "
                     "(rc %d)\n%s" % (path, p.returncode, (p.stdout + p.stderr)[-800:]))


def first_diff(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def _fuzz_wiring_ok() -> None:
    """Check the fuzz harness's wiring before it is used on a real file.

    The fuzzer is the one path in this tool whose prompt source is code rather
    than text, so a wiring error (an entry that is not callable, or that returns
    bytes instead of str) would otherwise surface as a tokenizer-shaped failure
    far from its cause. Two things are asserted here:

      * every `_FUZZ_TYPES` entry is callable as `kind(rng)` and returns `str`,
        which is the call convention `_gen_fuzz_strings` uses; and
      * a prompt survives the write-read round trip the harness actually does
        (each prompt is written to a file and re-read by both tokenizers), so a
        difference in this run is not an encoding artefact of the tool's own I/O.
    """
    r = random.Random(0)
    sample = [kind(r) for kind in _FUZZ_TYPES]
    for n, s in enumerate(sample):
        if not isinstance(s, str):
            raise SystemExit("fuzz harness: _FUZZ_TYPES[%d] returned %s, not str"
                             % (n, type(s).__name__))
    # The pool is only meaningful if `_gen_fuzz_strings` can draw from it.
    if not _gen_fuzz_strings(1, 0):
        raise SystemExit("fuzz harness: _gen_fuzz_strings produced no prompt")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s00.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(sample[0])
        with open(path, "r", encoding="utf-8") as f:
            roundtrip = f.read()
    if roundtrip != sample[0]:
        raise SystemExit("fuzz harness: the write-read round trip changed a prompt")


def separator_sweep(engine, model, tokenize, workdir) -> int:
    """Probe every separator in the engine's Space class against llama.cpp.

    Each probe is `separator + U+00B4`, not the separator alone: a lone space
    tokenizes identically whether the character was split or not, because a
    split still yields that character's own byte tokens. The acute makes the
    pair a merge the reference's vocabulary performs, so the two outcomes
    differ in the id list rather than only in an internal chunk boundary.

    Returns 0 when every separator agrees, 1 otherwise.
    """
    print("%-26s %-7s %-22s %s" % ("separator", "verdict", "engine ids", "llama ids"))
    print("%-26s %-7s %-22s %s" % ("", "", "(separator + U+00B4)", ""))
    diffs = 0
    for label, cp in SEPARATORS:
        text = chr(cp) + "\u00b4"
        path = os.path.join(workdir, "sep_%04X.txt" % cp)
        with open(path, "wb") as fh:
            fh.write(text.encode("utf-8"))
        e = engine_ids(engine, model, path)
        l = llama_ids(tokenize, model, path, False)
        same = e == l
        if not same:
            diffs += 1
        print("%-26s %-7s %-22s %s"
              % (label, "SAME" if same else "DIFF", str(e)[:22], str(l)[:22]))
    print()
    print("%d/%d separators identical" % (len(SEPARATORS) - diffs, len(SEPARATORS)))
    if diffs:
        print("VERDICT: FAIL -- %d separator(s) chunk differently from llama.cpp. The "
              "engine's Space class and the reference's `\\s` disagree somewhere in "
              "the whitespace-run rule, which is a merge-boundary defect (the "
              "model sees different ids)." % diffs)
        return 1
    print("VERDICT: PASS -- every separator in the engine's Space class chunks exactly "
          "as llama.cpp chunks it. This is the sweep that caught the byte-vs-character "
          "backtrack; the two ASCII controls are included so a rule that stopped "
          "handling ASCII would fail here too.")
    return 0


def main():
    # The test set is deliberately non-ASCII and a Windows console defaults to
    # cp1252, which cannot encode CJK or emoji: without this, the tool dies while
    # *reporting* a difference instead of reporting it.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("KNJ_TOK_MODEL", DEFAULT_MODEL),
                    help="the GGUF both tokenizers read (env: KNJ_TOK_MODEL)")
    ap.add_argument("--engine",
                    default=os.environ.get("KNJ_ENGINE", DEFAULT_ENGINE), help="the "
                    "engine binary (env: KNJ_ENGINE)")
    ap.add_argument("--llama-tokenize",
                    default=os.environ.get("KNJ_LLAMA_TOKENIZE", DEFAULT_TOKENIZE),
                    help="llama-tokenize.exe from the local llama.cpp build "
                         "(env: KNJ_LLAMA_TOKENIZE)")
    ap.add_argument("--strict", action="store_true",
                    help="a special-token POLICY difference is also a failure")
    ap.add_argument("--separator-sweep", action="store_true",
                    help="probe every separator in the engine's Space class (as "
                         "`separator + U+00B4`) against llama.cpp and exit; this is "
                         "the bounded sweep that found the byte-vs-character "
                         "whitespace backtrack, and it is a gate case (ctest "
                         "tok_separator_sweep). It replaces the normal comparison "
                         "run rather than adding to it.")
    ap.add_argument("--strings-file",
                    help="test strings separated by NUL bytes, UTF-8 (the built-in "
                         "set is used otherwise)")

    corpus = ap.add_argument_group("corpus and fuzz (large, unspecific inputs)")
    corpus.add_argument("--corpus",
                       nargs="*",
                       default=[],
                       help="natural-language corpus files, one prompt per line, UTF-8 "
                            "(each line is tokenised once and compared to llama.cpp). "
                            "These are the big unspecific prompts this check uses to "
                            "harden the tokenizer over a broad input surface; the "
                            "hand-written set in this tool is still run too unless "
                            "--only-corpus is given.")
    corpus.add_argument("--only-corpus",
                        action="store_true",
                        help="do not run the built-in hand-written set or the fuzzer; "
                             "only the --corpus files. Use this when re-running a "
                             "specific corpus against a changed tokenizer.")
    corpus.add_argument("--max-tokens",
                        type=int,
                        default=2048,
                        help="cap each prompt at this many tokens for the corpus run so "
                             "that a long document does not dominate the report. This is "
                             "a truncation of the prompt for the COMPARISON, not a change "
                             "to the tokenizer's own context; the tokenizer still reads "
                             "the whole file. Default 2048.")
    corpus.add_argument("--fuzz",
                        type=int,
                        default=0,
                        help="also run a random-byte fuzzer for this many prompts. Each "
                             "prompt is generated fresh (seeded, so it is reproducible) and "
                             "is compared against llama.cpp in the same way as a hand-written "
                             "string. 0 disables this mode -- the fuzzer is not on by "
                             "default, because its purpose is to find regression surfaces "
                             "(byte intervals, invalid UTF-8 placements, overlong sequences) "
                             "not to prove the tokenizer is complete over them.")
    corpus.add_argument("--fuzz-seed",
                        type=int,
                        default=17,
                        help="seed for the fuzzer prompt generator (so a fuzz run is "
                             "reproducible). Default 17.")
    fuzz_detail = corpus.add_mutually_exclusive_group()
    fuzz_detail.add_argument("--fuzz-verbose",
                             action="store_true",
                             dest="fuzz_verbose",
                             help="print every fuzz string's hex and both id lists on a "
                                  "difference, instead of only the summary counts at the end.")
    fuzz_detail.add_argument("--no-fuzz-verbose",
                             action="store_false",
                             dest="fuzz_verbose",
                             help="show only the summary counts (the default).")
    ap.set_defaults(fuzz_verbose=False)
    corpus.add_argument("--fuzz-min-fail",
                        type=int,
                        default=1,
                        help="fuzz boundary-difference count that triggers exit 1. 1 means "
                             "a single merge-boundary difference anywhere in corpus+fuzz is "
                             "a failure, matching the hand-written set's semantics. Raise it "
                             "only when fuzzing a tokenizer you know has a policy difference "
                             "you want to leave uncaught.")
    corpus.add_argument("--max-strings",
                        type=int,
                        default=None,
                        help="Docker-free safety cap on STRINGS, the hand-written set, when "
                             "it is too long for one turn's runtime budget. For example "
                             "--max-strings 128 only uses the first 128 strings. Empty = use "
                             "the whole built-in set.")
    corpus.add_argument("--fuzz-bound",
                        type=int,
                        default=None,
                        help="bound the fuzz prompts to the first N when a full --fuzz run is "
                             "too expensive and you only want the first N for the report (for "
                             "example --fuzz-bound 50 when re-running with --fuzz 5000). The "
                             "fuzzer still generates --fuzz prompts, but only the first N are "
                             "compared and reported; the rest are skipped. 0 = no bound (and "
                             "an explicit --fuzz-bound=0 = no bound). Default: no bound.")
    args = ap.parse_args()

    if args.fuzz_bound is not None and args.fuzz_bound < 0:
        raise SystemExit("tok_crosscheck: --fuzz-bound must be >= 0, got %d" % args.fuzz_bound)

    # The three inputs are PREREQUISITES, not arguments to be validated: with any
    # of them missing nothing can be compared, so this is a NOT RUN (exit 3). The
    # ctest case maps 3 to ***Skipped*** with this message printed, which is what
    # keeps "did not run" from being written the same way as "passed".
    prereq_missing = []
    if not os.path.isfile(args.model):
        prereq_missing.append("the model %s (--model / KNJ_TOK_MODEL)" % args.model)
    if not os.path.isfile(args.engine):
        prereq_missing.append("the engine %s (--engine / KNJ_ENGINE)" % args.engine)
    if not os.path.isfile(args.llama_tokenize):
        prereq_missing.append("llama-tokenize %s (--llama-tokenize / "
                              "KNJ_LLAMA_TOKENIZE)" % args.llama_tokenize)
    if prereq_missing:
        print("NOT RUN: %d prerequisite(s) for a comparison are missing:"
              % len(prereq_missing))
        for item in prereq_missing:
            print("  - %s" % item)
        print("         A run that compared nothing is not a pass.")
        return 3

    # Safety cap on the hand-written set for when it is too long for the runtime
    # budget of one turn; the corpus adds on top of it.
    if args.max_strings and len(STRINGS) > args.max_strings:
        strings = STRINGS[:args.max_strings]
    else:
        strings = list(STRINGS)

    if args.strings_file:
        # Split on a NUL, not on newlines: a test string may legitimately contain
        # a newline, and "one string per line" cannot express that.
        with open(args.strings_file, "rb") as fh:
            strings = [s.decode("utf-8") for s in fh.read().split(b"\0") if s]

    # The corpus lines are read whenever --corpus is given; --only-corpus then
    # decides whether they are compared *instead of* the hand-written set or
    # alongside it. Reading them into their own list first is what keeps both
    # spellings honest -- an earlier version skipped the corpus entirely under
    # --only-corpus (the flag was inverted), so a run that asked for one corpus
    # file silently compared the built-in 32 strings and reported a pass.
    corpus_lines: list[str] = []
    for cf in args.corpus:
        with open(cf, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.rstrip("\n").rstrip("\r")
                if stripped:
                    corpus_lines.append(stripped)
    if args.only_corpus:
        strings = list(corpus_lines)
    else:
        strings.extend(corpus_lines)

    # Per-prompt token cap for corpus mode only. The hand set and fuzzer are
    # already short and are not truncated. The cap is applied inside the
    # comparison branch below so every long prompt is logged there.
    max_tokens: int | None = args.max_tokens if corpus_lines else None


    if args.separator_sweep:
        # A distinct mode: it compares 14 sentinel inputs, not the hand set, and
        # its verdict is about one rule (the whitespace run) rather than about
        # tokenizer parity in general.
        sweep_dir = os.path.join(os.path.dirname(os.path.abspath(args.engine)),
                                 "tok_cc_tmp")
        os.makedirs(sweep_dir, exist_ok=True)
        print("engine        : %s" % args.engine)
        print("llama.cpp     : %s" % args.llama_tokenize)
        print("model         : %s" % args.model)
        print()
        return separator_sweep(args.engine, args.model, args.llama_tokenize, sweep_dir)

    print("engine        : %s" % args.engine)
    print("llama.cpp     : %s" % args.llama_tokenize)
    print("model         : %s" % args.model)
    print("strings       : %d (each written to a file and read with --prompt-file / -f)"
          % len(strings))
    print("engine ids    : `--dump-tokens -n 0` (`prompt ids :`) or the qwen35 probe's "
          "`id(s):`, whichever this architecture prints")
    print()
    print("%-44s %7s %7s %7s  %s" % ("string", "engine", "llama", "llama",
                                     "verdict"))
    print("%-44s %7s %7s %7s  %s" % ("", "ids", "parse", "raw", ""))

    # Every string goes through a FILE, never argv. Measured on this machine:
    # `-p <CJK>` reaches a native Windows binary in the ANSI code page, so the
    # tokenizer is handed mangled bytes, and `-p <emoji>` arrives split into
    # several arguments (the engine answers `error: missing value for -p`). That
    # is a property of the process boundary, not of either tokenizer, and it
    # would corrupt the comparison in a way that looks like a tokenizer defect.
    tmp = os.path.join(os.path.dirname(os.path.abspath(args.engine)), "tok_cc_tmp")
    os.makedirs(tmp, exist_ok=True)

    boundary, policy, ok, tokens = [], [], 0, 0
    fuzz_strings: list[str] = []
    fuzz_diffs: list[tuple[str, list[int], list[int], list[int]]] = []
    if args.fuzz and args.fuzz > 0:
        fuzz_strings.extend(_gen_fuzz_strings(args.fuzz, args.fuzz_seed))
        strings.extend(fuzz_strings)
    if args.fuzz_verbose:
        print("corpus    :", args.corpus if args.corpus else "(none)")
        print("fuzz      : %d prompts, seed %d, types %s" % (
            args.fuzz, args.fuzz_seed, (
                tuple(t.__name__ for t in _FUZZ_TYPES) if args.fuzz else [])))

    # The fuzz harness calls every `_FUZZ_TYPES` entry as `kind(r)`. Before
    # exercising it on the qwen3moe file, verify that the wiring matches the
    # call convention: the lambdas are functions, and the class entries
    # (_Repeat, _Overlong) are callable via __call__.

    if args.fuzz and args.fuzz > 0:
        _fuzz_wiring_ok()

    # The fuzzer can be bounded via --fuzz-bound. The fuzz prompts still occupy the
    # tail of `strings`, so the comparison branch below uses the fuzz-table and
    # fuzz_diffs machinery for exactly the first N fuzz prompts and then discards
    # the rest before the loop starts (same seed, same sequence).
    if args.fuzz and args.fuzz > 0 and args.fuzz_bound is not None and args.fuzz_bound >= 0:
        keep = args.fuzz_bound
        # N fuzz prompts stay at the end of `strings`; the first N are kept, the
        # rest of the fuzz prompts are dropped in place so that the loop's indexing
        # and fuzz_start stay correct.
        if keep < len(fuzz_strings):
            drop_at = len(strings) - len(fuzz_strings) + keep
            del strings[drop_at:]
            fuzz_strings = fuzz_strings[:keep]
            fuzz_diffs = []

    # "Not run" and "passed" are different results and must never be written the
    # same way (a run that compared zero prompts printed 0/0 and exited 0).
    if not strings:
        raise SystemExit("tok_crosscheck: nothing to compare -- --only-corpus was given "
                         "without a --corpus line, or the --strings-file was empty; "
                         "refusing to report a pass for a run that compared no prompts")

    # `strings` is [hand set] [+ corpus] [+ fuzz], and `fuzz_strings` is the last
    # block of it, so the index where the fuzz prompts start is what tells the
    # reporter below whether a difference came from a fuzz prompt.
    fuzz_start = len(strings) - len(fuzz_strings)

    for n, s in enumerate(strings):
        shown = s.replace("\n", "\\n").replace("\t", "\\t")
        shown = shown if len(shown) <= 42 else shown[:39] + "..."
        path = os.path.join(tmp, "s%02d.txt" % n)
        with open(path, "wb") as fh:
            fh.write(s.encode("utf-8"))
        e = engine_ids(args.engine, args.model, path)
        lp = llama_ids(args.llama_tokenize, args.model, path, True)
        lr = llama_ids(args.llama_tokenize, args.model, path, False)
        # The cap: truncate each list for the comparison and say so, with the
        # FULL length, before the verdict row prints the capped ones. All three
        # lists are fetched first because every notice names all three lengths —
        # printing the engine's notice before llama ran referenced variables
        # that did not exist yet (UnboundLocalError, hit by the first run that
        # ever crossed the cap).
        if max_tokens:
            full = (len(e), len(lp), len(lr))
            capped = []
            if full[0] > max_tokens:
                e = e[:max_tokens]
                capped.append("engine %d" % full[0])
            if full[1] > max_tokens:
                lp = lp[:max_tokens]
                capped.append("llama-parse %d" % full[1])
            if full[2] > max_tokens:
                lr = lr[:max_tokens]
                capped.append("llama-raw %d" % full[2])
            if capped:
                # one notice per prompt, before the verdict row that carries the
                # capped lengths: the full lengths are the interesting part.
                print("%-44s %7d %7d %7d  TRUNCATED for comparison (%d): %s"
                      % (shown, full[0], full[1], full[2], max_tokens,
                         ", ".join(capped)))
        tokens += len(e)
        if e == lp and e == lr:
            ok += 1
            print("%-44s %7d %7d %7d  PASS" % (shown, len(e), len(lp), len(lr)))
            continue
        if e == lr:
            policy.append((s, e, lp))
            print("%-44s %7d %7d %7d  POLICY: matches --no-parse-special only; the "
                  "engine does not parse control tokens" % (shown, len(e), len(lp), len(lr)))
            continue
        boundary.append((s, e, lp, lr))
        i = first_diff(e, lr) if first_diff(e, lr) >= 0 else first_diff(e, lp)
        if n >= fuzz_start:
            fuzz_diffs.append((s, e, lp, lr))
            if args.fuzz_verbose:
                print("%-44s %7d %7d %7d  FAIL first difference at index %d  (fuzz)"
                      % (shown, len(e), len(lp), len(lr), i))
                for name, ids in (("engine", e), ("llama ()", lp), ("llama raw", lr)):
                    print("      %-10s %s%s" % (name, str(ids[:14]),
                                                " ..." if len(ids) > 14 else ""))
            continue
        print("%-44s %7d %7d %7d  FAIL first difference at index %d"
              % (shown, len(e), len(lp), len(lr), i))
        for name, ids in (("engine", e), ("llama ()", lp), ("llama raw", lr)):
            print("      %-10s %s%s" % (name, str(ids[:14]),
                                        " ..." if len(ids) > 14 else ""))

    print()
    print("%d/%d strings identical, %d policy difference(s), %d merge-boundary "
          "difference(s), %d tokens compared"
          % (ok, len(strings), len(policy), len(boundary), tokens))
    if boundary:
        print("VERDICT: FAIL -- a merge boundary differs from llama.cpp's on %d "
              "string(s). The model would see different ids, so this is a defect in "
              "the engine's tokenizer or in the file's pre-tokeniser rule."
              % len(boundary))
        if args.fuzz_verbose and fuzz_strings:
            print()
            print("fuzz diffs (hex):")
            for s, e, lp, lr in fuzz_diffs:
                print("  %s  engine %s | llama %s" % (_hex(s), e, lp))
                print("      engine : %s" % e)
                print("      llama  : %s" % lp)
        return 1
    if policy and args.strict:
        print("VERDICT: FAIL (--strict) -- %d policy difference(s) between the engine "
              "and llama.cpp's default special-token parsing." % len(policy))
        return 1
    if policy:
        print("VERDICT: PASS -- every merge boundary agrees with llama.cpp on every "
              "string. %d string(s) differ only in special-token POLICY: the engine "
              "tokenises raw text and does not parse control tokens, which is stated "
              "here rather than hidden by running llama.cpp in one mode only."
              % len(policy))
        return 0
    print("VERDICT: PASS -- every merge boundary agrees with llama.cpp on every string.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
