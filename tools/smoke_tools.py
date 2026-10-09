#!/usr/bin/env python3
"""Standing functional smoke for the tokenizer tools -- a gate case, not a scratch file.

It asserts the *behaviour* of the tools it runs, not the exit code alone, because
the defect it was written for was a run that exited 0 while comparing nothing:

    * a run must have compared a non-zero number of prompts, and must say so;
    * `--only-corpus` must compare exactly the corpus lines;
    * `--corpus` alone must compare the hand set plus the corpus lines;
    * a POLICY difference (special-token parsing) is the documented, permitted
      result -- a MERGE-BOUNDARY difference is the failure;
    * `--only-corpus` with no corpus must refuse rather than print a 0/0 pass;
    * the imrope fixture must cover all four axes and emit valid JSON.

This lived in `tmp/` and was run by hand. It is now the ctest case `tools_smoke`
(`ctest --test-dir build/cmake-host -R tools_smoke --output-on-failure`), so it
runs with the rest of the suite. The separator sweep it used to be paired with is
now a first-class mode of tools/tok_crosscheck.py (`--separator-sweep`) and is
its own gate case, `tok_separator_sweep`; this file does not duplicate it.

Prerequisites, and what happens without them
--------------------------------------------
Three things are machine-specific: the engine binary, `llama-tokenize.exe` from
the local llama.cpp build, and a model file both can tokenize. Each is
discovered (env override first, then the default path used in AGENTS.md), and if
a prerequisite is missing the smoke prints which one and exits 3 -- an explicit
NOT RUN. The ctest case sets SKIP_RETURN_CODE 3, so a run without the reference
shows up as ***Skipped*** with the reason printed, never as a pass.

The engine is a MinGW build, so its `libstdc++-6.dll`, `libgcc_s_seh-1.dll` and
`libwinpthread-1.dll` must be findable. This script prepends the first candidate
directory that actually contains `libstdc++-6.dll` to the subprocess PATH --
without that, the engine fails at the process boundary (0xC0000139) and the
failure reads as a tokenizer result.

    python tools/smoke_tools.py
    KNJ_TOK_MODEL=<file.gguf> KNJ_LLAMA_TOKENIZE=<exe> python tools/smoke_tools.py

Exit codes:
    0   every case passed
    1   at least one case failed
    3   a prerequisite is missing, so the cases did NOT run (never a pass)
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOK = os.path.join(HERE, "tok_crosscheck.py")
FIX = os.path.join(HERE, "imrope_axis_fixture.py")

sys.path.insert(0, HERE)
# One source of truth for the machine-specific defaults and for where the MinGW
# runtime lives: the tool that owns them.
from tok_crosscheck import (DEFAULT_MODEL, DEFAULT_TOKENIZE,  # noqa: E402
                           MINGW_CANDIDATES, engine_env)

SUMMARY = re.compile(
    r"(\d+)/(\d+) strings identical, (\d+) policy difference\(s\), "
    r"(\d+) merge-boundary difference\(s\), (\d+) tokens compared")


def discover_engine() -> str | None:
    """The engine binary: explicit override, then build/cmake-host, then the
    newest kanjoos-run.exe under any build/*/ -- printed with its hash so a stale
    binary in a second build tree cannot be measured silently."""
    cands: list[str] = []
    if os.environ.get("KNJ_ENGINE"):
        cands.append(os.environ["KNJ_ENGINE"])
    cands.append(os.path.join(ROOT, "build", "cmake-host", "kanjoos-run.exe"))
    cands.extend(sorted(glob.glob(os.path.join(ROOT, "build", "*", "kanjoos-run.exe")),
                        key=os.path.getmtime, reverse=True))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def run(env: dict, *argv: str) -> tuple[int, str]:
    # errors="replace" for the same reason the tool itself uses it: the prompts
    # are raw bytes and a Windows console codec cannot decode all of them. A
    # harness that dies while *reading* a report must not be mistaken for the
    # tool failing.
    p = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, errors="replace")
    return p.returncode, p.stdout


def report(out: str, want_total: int | None = None) -> tuple[bool, str]:
    """Check the tool's summary against the contract it documents."""
    m = SUMMARY.search(out)
    if not m:
        return False, "no comparison summary -- nothing was compared"
    ok, total, policy, boundary, tokens = (int(g) for g in m.groups())
    if total == 0:
        return False, "compared 0 prompts"
    if want_total is not None and total != want_total:
        return False, "compared %d prompts, expected %d" % (total, want_total)
    if boundary:
        return False, "%d merge-boundary difference(s)" % boundary
    if tokens == 0:
        return False, "0 tokens compared"
    return True, ("%d/%d identical, %d policy, 0 boundary, %d tokens"
                  % (ok, total, policy, tokens))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    model = os.environ.get("KNJ_TOK_MODEL", DEFAULT_MODEL)
    tokenize = os.environ.get("KNJ_LLAMA_TOKENIZE", DEFAULT_TOKENIZE)
    engine = discover_engine()

    missing = []
    if engine is None:
        missing.append("the engine binary (set KNJ_ENGINE, or build kanjoos-run "
                       "into build/cmake-host)")
    if not os.path.isfile(tokenize):
        missing.append("llama-tokenize.exe at %s (set KNJ_LLAMA_TOKENIZE)" % tokenize)
    if not os.path.isfile(model):
        missing.append("a model file at %s (set KNJ_TOK_MODEL)" % model)
    # The imrope fixture case needs the `gguf` package (it WRITES a GGUF). That is
    # a prerequisite of this interpreter, and checking it here is what keeps a
    # missing package from being reported as a broken fixture -- a NOT RUN, not a
    # failure. Interpreter choice is the cmake variable KNJ_PYTHON.
    try:
        import gguf  # noqa: F401
    except ImportError:
        missing.append("the `gguf` python package in %s (pip install gguf; or "
                       "point -DKNJ_PYTHON at an interpreter that has it)"
                       % sys.executable)
    if missing:
        print("NOT RUN: this smoke measures two tokenizers against each other and "
              "a prerequisite is missing:")
        for m in missing:
            print("  - %s" % m)
        print("         A run that did not compare anything is not a pass.")
        return 3

    env = engine_env()
    with open(engine, "rb") as fh:
        eng_hash = hashlib.sha256(fh.read()).hexdigest()[:16]
    print("engine        : %s  sha256 %s" % (engine, eng_hash))
    print("llama-tokenize: %s" % tokenize)
    print("model         : %s" % model)
    mingw = next((d for d in MINGW_CANDIDATES
                  if os.path.isfile(os.path.join(d, "libstdc++-6.dll"))), None)
    print("mingw runtime : %s" % (mingw if mingw else
                                  "not found in the known locations (the engine may "
                                  "still run if its DLLs are on PATH already)"))
    print()

    failures: list[str] = []
    corpus = os.path.join(tempfile.gettempdir(), "smoke_corpus.txt")
    with open(corpus, "w", encoding="utf-8") as fh:
        fh.write("hello world\nThe capital of France is\n")

    def case(name: str, argv: list[str], want_total: int | None = None) -> None:
        rc, out = run(env, *argv)
        ok, why = report(out, want_total)
        # rc 0 <=> no merge-boundary difference: one contract, checked both ways.
        rc_ok = (rc == 0) == ok
        print("%-28s rc=%d  %s%s"
              % (name, rc, why, "" if rc_ok else "   <-- rc disagrees with the report"))
        if not (ok and rc_ok):
            failures.append("%s: rc=%d %s" % (name, rc, why))
            print(out)

    # Every case passes the engine explicitly, so the smoke cannot measure a
    # different binary than the one it printed above.
    case("--corpus --only-corpus",
         [TOK, "--engine", engine, "--model", model, "--corpus", corpus,
          "--only-corpus"], want_total=2)
    case("--corpus (hand + corpus)",
         [TOK, "--engine", engine, "--model", model, "--corpus", corpus])
    case("--fuzz 5 --fuzz-seed 7",
         [TOK, "--engine", engine, "--model", model,
          "--llama-tokenize", tokenize, "--fuzz", "5", "--fuzz-seed", "7",
          "--no-fuzz-verbose"])

    # --only-corpus with no corpus is a refusal, never a 0/0 pass.
    rc, out = run(env, TOK, "--engine", engine, "--model", model, "--only-corpus")
    refused = rc != 0 and "nothing to compare" in out
    print("%-28s rc=%d  %s" % ("--only-corpus, no corpus", rc,
                               "refused" if refused else "DID NOT REFUSE"))
    if not refused:
        failures.append("an empty run was not refused: rc=%d" % rc)
        print(out)

    # The CLI's tokenizer-only mode must exit 0 before prefill. Use an impossible
    # context size as a trap: the ordinary generation path refuses this prompt
    # after tokenizing it, but tokenization itself must remain available.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as prompt:
        prompt.write("hello world")
        prompt_path = prompt.name
    try:
        p = subprocess.run(
            [engine, "--model", model, "--prompt-file", prompt_path,
             "--tokenize-only", "--ctx", "1"],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace")
        rc, out = p.returncode, p.stdout
    finally:
        os.unlink(prompt_path)
    token_only_ok = rc == 0 and "prompt ids :" in out and "prefill    :" not in out
    print("%-28s rc=%d  %s" % ("--tokenize-only", rc,
                               "ids printed without prefill" if token_only_ok else "FAILED"))
    if not token_only_ok:
        failures.append("tokenize-only path: rc=%d" % rc)
        print(out)

    # The fixture: comma-separated positions, all four axes, valid JSON.
    rc, out = run(env, FIX, "--sections", "11", "11", "10", "224",
                  "--positions", "t=1,h=2,w=3,extra=7")
    fixture_ok = rc == 0 and "over axes [0, 1, 2, 3]" in out
    print("%-28s rc=%d  %s" % ("imrope fixture", rc,
                               "four axes covered" if fixture_ok else "axes NOT covered"))
    if not fixture_ok:
        failures.append("fixture: rc=%d (axes not all covered)" % rc)
        print(out)

    if failures:
        print("\nFAIL (%d):" % len(failures))
        for f in failures:
            print("  - " + f)
        return 1
    print("\nall smokes pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
