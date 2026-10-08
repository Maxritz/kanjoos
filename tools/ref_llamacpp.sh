#!/usr/bin/env bash
# tools/ref_llamacpp.sh -- the llama.cpp reference, CPU-only and GPU, for the
# smallest MoE on this machine.
#
# Why CPU-only matters: the small MoE is 964 MB, so `-ngl 99` puts the entire
# model in 16 GiB of VRAM and measures VRAM bandwidth. That number (214 t/s) is
# real but it is not the baseline the engine's CPU path competes against. The
# baseline for a CPU forward pass is `-ngl 0`, and that is what this records.
set -uo pipefail

MODEL="${MODEL:-/c/Users/rr/OneDrive/Desktop/kraken/models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf}"
BIN=/c/Users/rr/OneDrive/Desktop/dxl/llama.cpp-rdnax/build-hip-final/bin
PROMPT="${PROMPT:-The capital of France is}"
NL="${NL:-24}"
OUT="${OUT:-records/2026-10-07_llamacpp_reference.txt}"

export HIP_PATH='G:\ROCM10RT-gfx1201'
export HIP_VISIBLE_DEVICES=0
export PATH="/g/ROCM10RT-gfx1201/bin:/c/Windows/System32:$PATH"

mkdir -p "$(dirname "$OUT")"
{
  echo "# llama.cpp reference -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# model: $MODEL"
  echo "# prompt: $PROMPT"
  echo

  for ngl in 0 99; do
    echo "=== llama-cli -ngl $ngl --temp 0 -n $NL ==="
    echo "--- PROMPT TOKENS ---"
    ( cd "$BIN" && ./llama-cli.exe -m "$MODEL" -ngl "$ngl" -c 512 -t 32 \
        -p "$PROMPT" -n "$NL" -no-cnv --temp 0 --seed 1 --verbose-prompt 2>&1 )
    echo "=== end llama-cli -ngl $ngl ==="
    echo
  done

  echo "=== llama-bench pp512/tg128, cpu and gpu ==="
  ( cd "$BIN" && ./llama-bench.exe -m "$MODEL" -p 512 -n 128 -t 32 -ngl 0 2>&1 )
  ( cd "$BIN" && ./llama-bench.exe -m "$MODEL" -p 512 -n 128 -t 32 -ngl 99 2>&1 )
} > "$OUT" 2>&1
echo "REF_DONE rc=$?"
