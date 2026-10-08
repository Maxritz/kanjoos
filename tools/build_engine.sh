#!/usr/bin/env bash
# tools/build_engine.sh -- build the host-side engine (loader, tokenizer, model, CLI).
#
# This is deliberately a script and not a CMake target. The engine's host half
# is plain C++17 with no ROCm dependency, and the project's CMake tree is
# entangled with the HIP toolchain (cmake/rocm.cmake drives hipcc through
# enable_language(HIP), which this MinGW host cannot satisfy). A three-line
# compile keeps the reference path buildable today; folding it into CMake is a
# separate decision recorded in docs/BUILD-OUTLINE.md section 21.
#
# Usage: tools/build_engine.sh [debug]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CXX="${CXX:-g++}"
OUT_DIR="${OUT_DIR:-build/host}"
BIN="$OUT_DIR/kanjoos-run"

MODE="${1:-release}"
case "$MODE" in
  release) OPT="-O2 -DNDEBUG" ;;
  debug)   OPT="-O0 -g" ;;
  *) echo "unknown mode: $MODE (use release|debug)" >&2; exit 2 ;;
esac

mkdir -p "$OUT_DIR"

SRC=(
  src/loader/gguf.cpp
  src/loader/dequant.cpp
  src/tokenizer/tokenizer.cpp
  src/util/parallel.cpp
  src/model/model.cpp
  src/cli/main.cpp
)

echo "cxx     : $($CXX --version | head -1)"
echo "mode    : $MODE"
echo "sources : ${#SRC[@]}"

# -I. so the repo-rooted "src/..." include style resolves from any cwd.
"$CXX" -std=c++17 $OPT -pthread -I. -o "$BIN" "${SRC[@]}"

echo "binary  : $BIN"
ls -la "$BIN" | awk '{print "bytes   : " $5}'
