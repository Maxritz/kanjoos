#!/usr/bin/env bash
# Compile every Kanjoos ISA probe against each target and print a matrix.
#
#   tools/isa_probe/run_isa_probe.sh [compiler] [arch ...]
#
#   compiler: AMD device clang. Default: auto-detect under /g/ROCM10RT-*/lib/llvm/bin,
#             else $KNJ_CLANG, else hipcc on PATH.
#   arch ...: default "gfx1031 gfx1201"
#
# Exit codes:
#   0  toolchain healthy, matrix printed (REFUSED/CRASH/EMPTY cells are data,
#      not errors)
#   2  no compiler found
#   3  compiler found but UNHEALTHY — the preflight kernel did not compile.
#      No matrix is printed, because a broken compiler otherwise reports a
#      uniform wall of REFUSED that is indistinguishable from "no features".
#
# Why the preflight exists: an earlier revision drove these probes through
# hipcc with the HIP headers included. When the ROCm install was incomplete
# (no amdgcn/bitcode) or hipcc pointed at a deleted toolchain, every probe
# failed for reasons that had nothing to do with the ISA and the script printed
# "REFUSED" everywhere. It looked like a hardware verdict and it was a lie.
# A capability probe must be able to say "I could not measure".
#
# Four distinct cell values, because they need four different fixes:
#   OK       the instruction named below is present in the emitted ISA
#   REFUSED  the frontend rejected the source (bad spelling or a feature gate)
#   CRASH    the backend crashed (a compiler bug, not a verdict)
#   EMPTY    exit 0 but no kernel in the code object (a silent no-op — the
#            one that would fool a harness which only checked exit codes)

set -uo pipefail

CLANG="${1:-}"
shift || true
if [ -z "$CLANG" ] && [ -n "${KNJ_CLANG:-}" ]; then CLANG="$KNJ_CLANG"; fi
if [ -z "$CLANG" ]; then
  for cand in /g/ROCM10RT-gfx1201/lib/llvm/bin/clang.exe \
              /g/ROCM10RT-gfx1031/lib/llvm/bin/clang.exe \
              /opt/rocm/llvm/bin/clang; do
    if [ -x "$cand" ]; then CLANG="$cand"; break; fi
  done
fi
if [ -z "$CLANG" ] || ! command -v "$CLANG" >/dev/null 2>&1; then
  echo "device clang not found; set KNJ_CLANG or pass it as argv[1]" >&2
  exit 2
fi

read -r -a ARCHS <<< "${*:-gfx1031 gfx1201}"

NAMES=(sdot4 sdot2 pk_add_f16 pk_fma_f16 pk_mad_f32 dot4_asm dot8_i4_asm dot2c_f16_asm
       wmma_f16_w32 wmma_f16_gfx12 wmma_bf16_w32 wmma_bf16_gfx12
       dot2_i16_asm dot2c_i16_asm)
PROBE_ID=(0 1 2 3 4 5 6 7 8 9 10 11 12 13)
PREFLIGHT=90

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/isa_probe.hip"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# gfx1201 is natively wave32. The packed-math and WMMA operand layouts are
# selected by this target feature; it is a build flag, not a capability.
arch_flags() {
  case "$1" in
    gfx12*|gfx9[0-9]4*) echo "-Xclang -target-feature -Xclang +wavefrontsize32" ;;
    *)                   echo "" ;;
  esac
}

compile() {  # arch probe_id  -> writes $TMP/out.s, returns clang status
  local a="$1" id="$2" f
  f=($(arch_flags "$a"))
  "$CLANG" --offload-arch="$a" -nogpuinc -nogpulib --cuda-device-only \
           -x hip -std=c++17 -DKNJ_PROBE="$id" "${f[@]}" \
           -S "$SRC" -o "$TMP/out.s" 2>"$TMP/err.txt"
}

# ---------------------------------------------------------------------------
# LINK stage -- the matrix the upstream -S run does NOT produce.
# ---------------------------------------------------------------------------
# `-S` (text assembly) does not invoke the integrated assembler's operand
# validator. MEASURED on this machine: the 3-operand dot spelling
#     v_dot4_i32_i8 D, A, B      (missing the accumulator)
# assembles through -S with exit 0 and is emitted verbatim into the .s, but
# -c (assemble + link the device object) fails with
#     ld.lld: error: too few operands for instruction
# This is the exact defect that produced a "WORKS ON BOTH ARCHES" claim in an
# earlier docs version and then a build that would not link (docs/00-verified-
# facts.md section 7.2: assemble != link). So every OK cell gets a second
# verdict: L-FAIL if -c rejects it, OK if it links.
#
# -c here needs no device library (it is --cuda-device-only, -nogpulib) and no
# HIP headers (the probe sources its own __global__ shim), so it runs on every
# tree the compile stage runs on, including ones that cannot link a full binary.
linkstage() {  # arch probe_id -> $TMP/link.o, returns clang status
  local a="$1" id="$2" f
  f=($(arch_flags "$a"))
  "$CLANG" --offload-arch="$a" -nogpuinc -nogpulib --cuda-device-only \
           -x hip -std=c++17 -DKNJ_PROBE="$id" "${f[@]}" \
           -c "$SRC" -o "$TMP/link.o" 2>"$TMP/linkerr.txt"
}

# Did the backend actually EMIT a kernel? Clang returns 0 in a case that looks
# like success and is not: forcing the sdot4 feature gate open on gfx1201
# (-Xclang -target-feature -Xclang +dot1-insts) passes the frontend and then
# produces a code object with `amdhsa.kernels: []` — exit 0, zero instructions,
# no kernel. A harness that only watched exit codes would record that as a
# working capability. Any cell whose .s carries no kernel metadata is EMPTY,
# which is a different bug from REFUSED and needs a different fix.
kernel_emitted() { grep -q '\.amdhsa_next_free_vgpr' "$TMP/out.s" 2>/dev/null; }

# The instruction this probe exists to test. We read it back out of the
# emitted ISA rather than trusting an exit code: a compiler can accept a
# builtin and then lower it to something else, and that is exactly the kind
# of silent substitution a capability matrix must not contain.
interesting_ins() {
  grep -oE '(v_dot[0-9a-z_]*|v_wmma_[a-z0-9_]+|v_pk_[a-z0-9_]+|v_fma_mix[a-z0-9_]*)' \
    "$TMP/out.s" 2>/dev/null | head -1
}

echo "compiler: $CLANG"
echo "version : $("$CLANG" --version 2>/dev/null | head -1)"
echo

UNHEALTHY=""
for a in "${ARCHS[@]}"; do
  if compile "$a" "$PREFLIGHT"; then
    echo "preflight $a: healthy"
  else
    UNHEALTHY="$UNHEALTHY $a"
    echo "preflight $a: BROKEN — a trivial kernel with no ISA features failed."
    sed 's/^/    /' "$TMP/err.txt" | head -6
  fi
done
if [ -n "$UNHEALTHY" ]; then
  echo
  echo "TOOLCHAIN UNHEALTHY for:$UNHEALTHY"
  echo "Refusing to print a capability matrix: a broken compiler is"
  echo "indistinguishable from missing hardware. Fix the install first."
  exit 3
fi
echo

printf '%-18s' "probe"
for a in "${ARCHS[@]}"; do printf '%-13s' "$a"; done
printf '\n-----------------------------------------------------------------------------------\n'

declare -A FIRSTERR FIRSTINS ALLINS CELLERR CELLPASS
for idx in "${!NAMES[@]}"; do
  name="${NAMES[$idx]}"; id="${PROBE_ID[$idx]}"
  printf '%-18s' "$name"
  for a in "${ARCHS[@]}"; do
    if compile "$a" "$id"; then
      if ! kernel_emitted; then
        printf '%-13s' "EMPTY"
        CELLERR["$a|$name"]="SILENT NO-OP: compiled with exit 0 but the code object has no kernel (amdhsa.kernels is empty). Not a capability."
        continue
      fi
      ins="$(interesting_ins)"
      if [ -n "$ins" ]; then
        printf '%-13s' "OK"
        [ -z "${ALLINS[$name]:-}" ] && ALLINS["$name"]="$ins"
        [ -z "${FIRSTINS[$a]:-}" ] && FIRSTINS["$a"]="$ins"
      else
        printf '%-13s' "OK?"
      fi
    elif grep -qE 'Stack dump|failed due to signal|submit a bug report' "$TMP/err.txt"; then
      printf '%-13s' "CRASH"
      CELLPASS["$a|$name"]="$(grep -m1 'Running pass' "$TMP/err.txt" | sed 's/.*Running pass/Running pass/')"
      CELLERR["$a|$name"]="COMPILER BUG: backend crash, not a capability verdict"
      [ -z "${FIRSTERR[$a]:-}" ] && FIRSTERR["$a"]="compiler crash in $(grep -m1 'Running pass' "$TMP/err.txt" | sed 's/.*Running pass/Running pass/')"
    else
      printf '%-13s' "REFUSED"
      e="$(grep -m1 -E 'error:' "$TMP/err.txt" | sed 's/.*error: //' | cut -c1-140)"
      CELLERR["$a|$name"]="$e"
      [ -z "${FIRSTERR[$a]:-}" ] && FIRSTERR["$a"]="$e"
    fi
  done
  printf '\n'
done

echo
echo "Instruction each OK cell actually emitted (read back out of the .s):"
for idx in "${!NAMES[@]}"; do
  [ -n "${ALLINS[${NAMES[$idx]}]:-}" ] && printf '  %-16s -> %s
' "${NAMES[$idx]}" "${ALLINS[${NAMES[$idx]}]}"
done

# "REFUSED" is a statement about ONE SPELLING under ONE FLAG SET, not about
# the silicon. Printing only the first refusal per arch hides that the cells
# fail for three unrelated reasons, which is exactly the ambiguity that let a
# broken install masquerade as a hardware limit. So: every refusal, verbatim,
# labelled with what kind of thing it is.
echo
echo "Every refusal, verbatim, and what kind of thing it is:"
for idx in "${!NAMES[@]}"; do
  name="${NAMES[$idx]}"
  for a in "${ARCHS[@]}"; do
    e="${CELLERR[$a|$name]:-}"
    [ -z "$e" ] && continue
    kind="?"
    case "$e" in
      *"needs target feature"*)
        kind="FRONTEND FEATURE GATE on this builtin spelling." ;;
      *"cannot initialize a parameter"*|*"too few arguments"*|*"no matching function"*)
        kind="BAD TEST: this build exposes no call signature we could match. Not a hardware verdict." ;;
      *"invalid output constraint"*|*"invalid operands"*)
        kind="BAD TEST: wrong asm operand types. Not a hardware verdict." ;;
      "COMPILER"*)
        kind="BACKEND CRASH." ;;
      "SILENT"*)
        kind="SILENT NO-OP: exit 0, no kernel in the code object." ;;
    esac
    printf '  %-9s %-16s %s\n' "$a" "$name" "$kind"
    printf '      %s\n' "$e"
  done
done
echo
echo "Per-arch summary:"
for a in "${ARCHS[@]}"; do
  echo "  $a: first refusal — ${FIRSTERR[$a]:-none}"
  [ -n "${FIRSTINS[$a]:-}" ] && echo "      first probe instruction emitted — ${FIRSTINS[$a]}"
done
echo
echo "Link verdict: does -c (assemble + link the device object) succeed?"
echo "  This is the stage the upstream -S matrix does NOT run. An instruction"
echo "  that assembles (-S, exit 0) but will not link (-c fails) is reported as"
echo "  L-FAIL here. docs/00-verified-facts.md section 7.2: assemble != link."
echo
printf '%-18s' "probe"
for a in "${ARCHS[@]}"; do printf '%-13s' "$a"; done
printf '\n-----------------------------------------------------------------------------------\n'
declare -A LINKERR
for idx in "${!NAMES[@]}"; do
  name="${NAMES[$idx]}"; id="${PROBE_ID[$idx]}"
  printf '%-18s' "$name"
  for a in "${ARCHS[@]}"; do
    if compile "$a" "$id" && kernel_emitted; then
      if linkstage "$a" "$id"; then
        printf '%-13s' "OK"
      else
        printf '%-13s' "L-FAIL"
        LINKERR["$a|$name"]="$(grep -m1 -E 'error:' "$TMP/linkerr.txt" | sed -E 's/.*error: //' | cut -c1-140)"
      fi
    else
      printf '%-13s' "-"   # no asm to link
    fi
  done
  printf '\n'
done
echo
echo "Every L-FAIL, verbatim:"
for idx in "${!NAMES[@]}"; do
  name="${NAMES[$idx]}"
  for a in "${ARCHS[@]}"; do
    e="${LINKERR[$a|$name]:-}"; [ -z "$e" ] && continue
    printf '  %-9s %-16s %s\n' "$a" "$name" "$e"
  done
done
exit 0
