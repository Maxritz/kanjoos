# records/README.md
# Records — raw measured evidence
#
# This tree holds the raw evidence behind the project's measured numbers.
# It is the honest "where the number came from" layer.  It is intentionally
# messy and dated: corpus-level curation happens in the docs/tools/baselines
# files, and this tree is the scratch space behind them.
#
# Reading order:
#   1. tools/bench/baselines.txt          — first-class pinned benchmarks
#   2. tools/bench/tok-fuzz-baseline.txt  — tok-fuzz result + other-scores context
#   3. docs/*.md                          — the design evidence built from these
#   4. this tree                          — the raw supporting artifacts
#
# What lives here vs what is curated:
#   - first-class pinned results are in the baseline files above
#   - run logs, profile dumps, tensor dumps, ref recordings, and session notes
#     live here as their raw form
#
# Sizes to know:
#   records/c21-baseline-2026-10-08/        — CSV/JSON/TXT profile runs + logs (~276K)
#   records/qwen35-imrope-fixture/          — imrope axis fixture tables (~17K)
#   records/qwen35-probe-2026-10-08/        — fp32 probe tensor dumps + logs (~19M)
#   records/2026-10-07_*.txt                — device/host/stream/ref evidence
#   records/2026-10-08_*.txt                — 10-08 bringup + streamed ffn smoke notes
#   records/tok-fuzz-5000/                  — 5000-prompt tok-fuzz campaign artifacts
#   records/tok-crosscheck-fuzz-*.log       — fuzz cross-check logs
#   records/tok-pre-dispatch-*.log          — pre-tokeniser dispatch mutation logs
#
# Regenerable artifacts
#   Several dirs here are regenerable from the harness or the fixture generator.
#   Do not treat them as canonical source.  Treat them as supporting evidence
#   unless a baseline file explicitly cites the file inside.
#
# Scratch / temp
#   tmp/ and .freebuff/ remain excluded per the root .gitignore.  This tree
#   is not a dump for everything the workspace touches — only measured evidence
#   worth keeping on disk.

