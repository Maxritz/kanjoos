#!/usr/bin/env bash
# Kanjoos microbenchmark runner — compile census + host correctness tier.
#
#   tools/bench/run_bench.sh [compiler] [arch ...]
#   tools/bench/run_bench.sh --host-only [compiler]
#
# A bench file (tools/bench/*.hip) is compiled for each target arch and the
# emitted assembly is measured: which kernels exist, what instructions they
# contain, how many of each, and what register/scratch resources they claim.
# The SAME file is then compiled natively for the host and run, which executes
# the CPU-reference correctness checks.
#
#   TIERS
#     A  host, always runs here. Correctness of the algorithm. Exit code
#        reflects it: a failing check fails the runner.
#     B  device compile, always runs here. Instruction and resource census.
#        This is where an A/B claim becomes a measurement — but it is an
#        INSTRUCTION-ISSUE measurement, never a throughput measurement.
#     C  device run. Needs a working ROCm install and a real GPU. Reported
#        as UNMEASURED when unavailable, never quietly skipped. The driver's
#        stdout is captured and, for drivers with entries in
#        tools/bench/baselines.txt, key numbers are compared against the
#        pinned reference values with a 10% tolerance - a regression past
#        that fails the runner (exit 6). The tier C correctness gate is
#        untouched: baselines are only compared after a driver exits 0.
#
# EXIT CODES
#   0  everything ran; tier A checks passed
#   1  a tier A correctness check FAILED (algorithm bug, or a stale reference)
#   2  no device compiler found
#   3  device compiler found but UNHEALTHY (preflight kernel failed) — no
#      census is printed, because a broken compiler otherwise reports a
#      uniform wall of REFUSED indistinguishable from missing hardware
#   4  no host compiler found, so tier A could not run
#   5  a tier C device run FAILED its own correctness check
#   6  a pinned baseline in tools/bench/baselines.txt REGRESSED by more than
#      the 10% tolerance (only reachable when tiers A and C both passed)
#   7  a driver DECLARED an instruction in its KNJ_EXPECT line and the code
#      object does not contain it. A declaration is a contract, not a comment:
#      the cell is OK? and the runner fails. See the note above expect_check.
#
# A tier C driver also exits 6 when its code object is for a different arch than
# the attached GPU. That is a skip, not a failure, and not a result: such a
# binary runs without error, never dispatches a kernel, and prints confident
# nonsense. Every driver refuses explicitly rather than being filtered here.
#
# The four cell values mirror tools/isa_probe/run_isa_probe.sh, because they
# mean the same thing there:
#   OK       compiled, kernel emitted, and the expected instruction is present
#   OK?      compiled, kernel emitted, but an expected instruction is MISSING
#            -- and that FAILS the runner (exit 7). An OK? cell that still exits
#            0 is a marker nobody is obliged to read.
#   REFUSED  frontend rejected the source (bad spelling or a feature gate)
#   CRASH    backend crashed
#   EMPTY    exit 0 but the code object contains no kernel

set -uo pipefail

HOST_ONLY=0
if [ "${1:-}" = "--host-only" ]; then HOST_ONLY=1; shift; fi

# Tier B declared-instruction accounting. Declared here, next to the exit-code
# contract, so the final decision cannot read an unset variable under `set -u`.
EXPECT_MISSING=0
MISSING_LIST=""

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# The device clang is a native binary: under Git Bash it wants Windows-form
# paths, and an -I that MSYS happens to convert is not the same thing as one
# that is spelled correctly. One helper, used by every -I and --rocm-path below.
winpath() { cygpath -m "$1" 2>/dev/null || printf '%s' "$1"; }

# ------------------------------------------------------------- toolchain ---
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
HOSTCXX="${KNJ_HOSTCXX:-}"
if [ -z "$HOSTCXX" ]; then
  for cand in g++ clang++; do
    if command -v "$cand" >/dev/null 2>&1; then HOSTCXX="$cand"; break; fi
  done
fi

read -r -a ARCHS <<< "${*:-gfx1201 gfx1031}"
[ "$HOST_ONLY" = 1 ] && ARCHS=()

BENCHES=()
while IFS= read -r f; do BENCHES+=("$f"); done < <(ls "$HERE"/*.hip 2>/dev/null)
if [ "${#BENCHES[@]}" -eq 0 ]; then
  echo "no bench files in $HERE" >&2; exit 2
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

arch_flags() {
  case "$1" in
    gfx12*|gfx9[0-9]4*) echo "-Xclang -target-feature -Xclang +wavefrontsize32" ;;
    *)                   echo "" ;;
  esac
}

# A bench declares what it must contain:
#   // KNJ_EXPECT: knj_gemm_wmma_f16 = v_wmma_f32_16x16x16_f16, knj_gemm_w4a4 = v_dot8_i32_i4
expect_map() { grep -oE 'KNJ_EXPECT:.*' "$1" | head -1 | sed 's/KNJ_EXPECT://'; }

# The body of ONE kernel in a .s file: from its `name:` label to the next
# top-level identifier label. AMDGPU emits internal labels as `.LBB0_1:` (a
# leading dot) and the function end as `.Lfunc_end`, so a top-level identifier
# label is exactly a function boundary.
#
# This is per-kernel on purpose. A file-wide grep for a declared instruction
# passes whenever ANY kernel in the file happens to contain it, so a declaration
# naming kernel A is satisfied by kernel B -- and a kernel that was declared but
# never emitted at all passes on the strength of its neighbours. Both are
# exactly the OK? cells tier B exists to catch, and neither is visible to a
# whole-file search.
kernel_body() { # $1 = .s file, $2 = kernel name
  awk -v k="$2" '
    $0 ~ "^" k ":"                        { inside = 1; next }
    inside && /^[a-zA-Z_][a-zA-Z0-9_]*:/  { exit }
    inside && /\.Lfunc_end/               { exit }
    inside                                { print }
  ' "$1"
}

# Kernels this bench is supposed to define, from the extern "C" kernel decls.
kernel_list() { grep -oE 'void (knj_[a-z0-9_]+)\(' "$1" | sed 's/void //;s/($//;s/(//' | sort -u; }

# A file is a TIER C DRIVER if it pulls in the HIP runtime or allocates device
# memory, because the host compiler cannot compile either. Detected by content,
# not by the *_run suffix: expert_gemm.hip and wmma_layout.hip are drivers too
# and were previously being fed to g++, which failed them for 'hip/hip_runtime.h
# file not found' — a missing-architecture error dressed as an algorithm error.
is_device_driver() { grep -qE '#include[[:space:]]*<hip/hip_runtime\.h>|hipMalloc\(' "$1"; }

# ROCm tree that has amdhip64.lib for the link line. The device bitcode is
# LLVM's (not per-arch), and the target is selected by --offload-arch, so one
# hipcc links both arches. We only need a tree whose lib/ contains amdhip64.lib
# (or amdhip64_7.lib) for the -L link path. On this machine that is
# /g/ROCM10RT-gfx1201/lib  (and /c/ROCm72/lib as a fallback).
rocm_lib_root_for_arch() {
  case "$1" in
    gfx1201) echo "/g/ROCM10RT-gfx1201" ;;
    gfx1031) echo "/g/ROCM10RT-gfx1031" ;;
    *)       echo "" ;;
  esac
}
find_rocm_lib_path() {
  local root="${1:-}"
  for cand in "${root}/lib" "/c/ROCm72/lib"; do
    if [ -f "${cand}/amdhip64.lib" ] || [ -f "${cand}/amdhip64_7.lib" ] || [ -f "${cand}/amdhip64.dll" ]; then
      echo "$cand"; return 0
    fi
  done
  # Last resort: the value itself, in case the runner is pointed at a lib dir.
  echo "$root"
}

# rocWMMA drivers need two flags nothing else needs, and both are per-source:
# leaking them into the other drivers would change what those drivers mean.
#
#   -std=c++17   rocWMMA is not C++14-clean. Without it the headers fail with
#                "no member named 'apply' in namespace 'std'" (utility/apply.hpp)
#                followed by a cascade of constexpr errors in io_bearer_base.hpp.
#                This is a header requirement, not a preference.
#
#   --rocm-path  must point at <root>/lib/llvm -- the DEVICE library directory --
#                NOT the ROCm root. Pointing at the root answers
#                "cannot find ROCm device library". Both spellings look like
#                "the ROCm path" and only one works.
#
# A rocWMMA driver is recognised by its include, so a new one is picked up with
# no list to keep in sync.
is_rocwmma_driver() { grep -qE '#include[[:space:]]*<rocwmma/rocwmma\.hpp>' "$1"; }

# Project sources a driver links against, recognised the same way: a driver that
# writes `#include "src/<area>/<name>.h"` gets `src/<area>/<name>.cpp` added to
# its compile line when that file exists. The alternative is a hand-maintained
# list, and the failure mode of a stale list is a link error naming the COMPONENT
# (undefined `knj::ResidencyManager::*`) that reads as "the driver is broken"
# when the truth is "the runner forgot a file". Nothing to keep in sync.
# What a driver needs in order to be run by the gate, declared in its own source
# so the reason travels with the driver instead of living in a list here:
#
#   // KNJ_BENCH_ARGS: <argv the driver needs>   -> the gate passes those argv
#   // KNJ_BENCH_REQUIRES: <reason>              -> NOT RUN, reason printed
#
# WITHOUT THIS, a driver that needs a model file exits 2 on a usage error and the
# runner reports "the kernel does not compute the right answer on the real
# device" — which is a different, false claim about a driver that never
# executed a kernel. A driver that did not run is reported, never omitted, and
# never mislabelled (rule 9).
bench_declared() {   # $1 = file, $2 = tag
  grep -m1 -E "^//[[:space:]]*$2:" "$1" 2>/dev/null |
    sed -e "s|^//[[:space:]]*$2:[[:space:]]*||"
}
extra_sources_for() {
  local src="${1:-}" inc abs
  [ -f "$src" ] || return 0
  grep -oE '#include[[:space:]]*"src/[A-Za-z0-9_/.-]+\.h"' "$src" 2>/dev/null |
    sed -e 's/^#include[[:space:]]*"//' -e 's/"$//' |
    while IFS= read -r inc; do
      abs="$(cd "$HERE/../.." && pwd)/${inc%.h}.cpp"
      [ -f "$abs" ] && printf ' %s' "$abs"
    done
}
rocwmma_flags_for() {
  # Two statements, not `local a=b c=$a`: bash expands every word of a
  # declaration command before applying any of its assignments, so the second
  # one reads the OUTER (unset) variable and `set -u` aborts the whole build
  # line. That is what made every rocwmma driver look like a refuser here.
  local root="${1:-}"
  local llvm="${root}/lib/llvm"
  if [ -d "$llvm" ]; then
    printf -- '--rocm-path=%s -std=c++17' "$(cygpath -m "$llvm" 2>/dev/null || echo "$llvm")"
  else
    # No device-library tree: emit only the language flag. The build will then
    # fail loudly on the include or the device lib rather than silently get
    # built against a guessed path.
    printf -- '-std=c++17'
  fi
}

echo "device compiler : ${CLANG:-<none>}"
[ -n "$CLANG" ] && echo "                  $($CLANG --version 2>/dev/null | head -1)"
echo "host compiler   : ${HOSTCXX:-<none>}"
echo "bench files     : $(printf '%s ' "${BENCHES[@]##*/}")"
echo "target arches   : ${ARCHS[*]:-<host only>}"
echo

# ======================================================================
# TIER A — CPU reference correctness. Runs everywhere.
# ======================================================================
echo "=============================================================================="
echo "TIER A  host correctness (CPU reference vs kernel mirror)"
echo "=============================================================================="
if [ -z "$HOSTCXX" ]; then
  echo "SKIPPED: no host compiler (set KNJ_HOSTCXX). Tier A is the only tier that"
  echo "can run on a machine with no usable GPU, so this is a real gap."
  TIER_A_RC=4
else
  TIER_A_RC=0
  for src in "${BENCHES[@]}"; do
    b="$(basename "$src" .hip)"
    # Tier C drivers cannot be built by the host compiler: they include the HIP
    # runtime and allocate device memory. Skipping them is not an omission —
    # they are executed in tier C below, and if that tier cannot run, it says
    # so and the exit code reflects it.
    if is_device_driver "$src" || [[ "$b" == *_run ]]; then
      echo "-- $b: tier C driver, skipped in tier A (see below)"
      continue
    fi
    if ! "$HOSTCXX" -std=c++17 -O2 -x c++ -I"$HERE" "$src" -o "$TMP/$b.exe" 2>"$TMP/$b.cc.err"; then
      echo "-- $b: HOST BUILD FAILED"
      sed 's/^/     /' "$TMP/$b.cc.err" | head -12
      TIER_A_RC=1
      continue
    fi
    echo "-- $b"
    "$TMP/$b.exe"
    rc=$?
    [ $rc -ne 0 ] && TIER_A_RC=1
    echo
  done
fi

# ======================================================================
# TIER B — device compile census
# ======================================================================
if [ "$HOST_ONLY" = 1 ] || [ "${#ARCHS[@]}" -eq 0 ]; then
  exit $TIER_A_RC
fi

if [ -z "$CLANG" ] || ! command -v "$CLANG" >/dev/null 2>&1; then
  echo "device clang not found; set KNJ_CLANG or pass it as argv[1]" >&2
  exit 2
fi

echo "=============================================================================="
echo "TIER B  device compile census (instruction + resource, NOT throughput)"
echo "=============================================================================="
# WHY THIS TIER USED TO MEASURE NOTHING
# -------------------------------------
# The census used to pass -nogpuinc, which suppresses the GPU include paths, so
# every driver answered 'hip/hip_runtime.h' file not found and tier B reported an
# empty census while its own preflight said the toolchain was healthy. The
# census therefore never reached the link step that the sources a driver needs
# affect. The fix is the spelling AGENTS.md section 3 records as verified by
# hand: no -nogpuinc, --rocm-path at the ROCm ROOT (the <root>/lib/llvm spelling
# belongs to the hipcc rocWMMA line, a different tool with a different search
# order), and the include roots the real build uses (-I kernels, -I repo root).
# A driver that still refuses is reporting a real defect in ITSELF, which is the
# point of the census.

UNHEALTHY=""
for a in "${ARCHS[@]}"; do
  f=($(arch_flags "$a"))
  cat > "$TMP/pre.hip" <<'EOF'
#define __global__ __attribute__((global))
extern "C" __global__ void knj_pre(float *o) { o[0] = 1.0f; }
EOF
  if "$CLANG" --offload-arch="$a" -nogpuinc -nogpulib --cuda-device-only \
       -x hip -std=c++17 "${f[@]}" -S "$TMP/pre.hip" -o "$TMP/pre.s" 2>/dev/null \
     && grep -q '\.amdhsa_next_free_vgpr' "$TMP/pre.s"; then
    echo "preflight $a: healthy"
  else
    UNHEALTHY="$UNHEALTHY $a"
    echo "preflight $a: BROKEN — a kernel with no ISA features did not compile."
  fi
done
if [ -n "$UNHEALTHY" ]; then
  echo
  echo "TOOLCHAIN UNHEALTHY for:$UNHEALTHY"
  echo "Refusing to print a census: a broken compiler is indistinguishable from"
  echo "missing hardware. Run tools/doctor/kanjoos_doctor.sh for the install audit."
  exit 3
fi
echo

for arch in "${ARCHS[@]}"; do
  f=($(arch_flags "$arch"))
  echo "---- $arch ----"
  for src in "${BENCHES[@]}"; do
    b="$(basename "$src" .hip)"
    printf '  %-14s ' "$b"
    cflags=("-I$(winpath "$HERE")" "-I$(winpath "$ROOT/kernels")" "-I$(winpath "$ROOT")")
    root="$(rocm_lib_root_for_arch "$arch")"
    if [ -n "$root" ]; then
      cflags+=("--rocm-path=$(winpath "$root")")
      # Only for the drivers that include rocWMMA, matching the tier C rule: a
      # rocWMMA flag must not leak into a driver that does not use it.
      if is_rocwmma_driver "$src"; then cflags+=("-I$(winpath "$root/include")"); fi
    fi
    if ! "$CLANG" --offload-arch="$arch" -nogpulib --cuda-device-only \
         -x hip -std=c++17 -DKNJ_DEVICE_TIER=1 \
         ${cflags[@]+"${cflags[@]}"} "${f[@]}" \
         -S "$src" -o "$TMP/$b.$arch.s" 2>"$TMP/$b.$arch.err"; then
      if grep -qE 'Stack dump|failed due to signal' "$TMP/$b.$arch.err"; then
        printf 'CRASH   %s\n' "$(grep -m1 'Running pass' "$TMP/$b.$arch.err" | sed 's/.*Running pass/Running pass/')"
      else
        printf 'REFUSED %s\n' "$(grep -m1 -E 'error:' "$TMP/$b.$arch.err" | sed 's/.*error: //' | cut -c1-70)"
      fi
      continue
    fi
    if ! grep -q '\.amdhsa_next_free_vgpr' "$TMP/$b.$arch.s"; then
      printf 'EMPTY   compiled but the code object has no kernel\n'
      continue
    fi

    # Per-kernel resource census.
    awk -v arch="$arch" '
      /^[a-zA-Z_][a-zA-Z0-9_]*:/ { name=$1; sub(/:$/,"",name) }
      /\.amdhsa_next_free_vgpr/       { vgpr[name]=$2 }
      /\.amdhsa_private_segment_fixed_size/ { scratch[name]=$2 }
      /\.amdhsa_next_free_sgpr/       { sgpr[name]=$2 }
      END {
        for (k in vgpr)
          printf "  %-24s vgpr=%-4s sgpr=%-4s scratch=%-6s\n", k, vgpr[k], sgpr[k], scratch[k]
      }' "$TMP/$b.$arch.s"

    # Instruction census: what arithmetic did the compiler actually emit.
    awk '
      /^[[:space:]]*v_(mad|fma|pk_|dot|wmma|add|mul|and|lshrrev|shr|shl|sub|cvt|mac)/ {
        line=$0
        sub(/^[[:space:]]*/,"",line)
        split(line, a, /[[:space:]]+/)
        if (a[1] ~ /^v_pk_/) cnt["packed"]++
        else if (a[1] ~ /^v_wmma_/) { cnt["wmma"]++; wm[a[1]]++ }
        else if (a[1] ~ /^v_dot/)  { cnt["dot"]++;  dot[a[1]]++ }
        else if (a[1] ~ /^v_(mad|fma|mac)/) cnt["fma"]++
        else if (a[1] ~ /^v_(and|lshr|shr|shl)/) cnt["int_extract"]++
        else if (a[1] ~ /^v_(cvt|add|mul|sub)/) cnt["other_vop"]++
      }
      END {
        for (k in cnt) printf "  %-16s %d\n", k, cnt[k]
        for (k in wm) printf "    wmma   %-28s x%d\n", k, wm[k]
        for (k in dot) printf "    dot    %-28s x%d\n", k, dot[k]
      }' "$TMP/$b.$arch.s" | sort

    # Declared expectation check -- a FAILING gate, not a label.
    #
    # KNJ_EXPECT is a contract: "this driver emits this instruction". When the
    # code object does not contain it, the tier B cell is OK? and EXPECT_RC is
    # set, so the runner exits 7. The measured instruction must be inside the
    # NAMED kernel (kernel_body), not merely somewhere in the file.
    #
    # A declaration may carry an arch glob, which is how a contract that is true
    # for one target and false for another is stated honestly instead of turned
    # off:
    #   // KNJ_EXPECT: knj_a = v_dot8_i32_i4, @gfx9* knj_b = v_dot4_i32_i8
    # A scoped entry that does not name the arch under census is reported as
    # `scoped`, never silently dropped: "not applicable" and "passed" are
    # different results (AGENTS.md section 4, rule 9).
    exp="$(expect_map "$src")"
    if [ -n "$exp" ]; then
      IFS=',' read -ra pairs <<< "$exp"
      for p in "${pairs[@]}"; do
        left="${p%%=*}"
        right="${p#*=}"
        [ "$right" = "$p" ] && continue          # no '=' at all
        insn="$(printf '%s' "$right" | tr -d ' ')"
        [ -z "$insn" ] && continue
        scope=""
        left="$(printf '%s' "$left" | sed 's/^[[:space:]]*//')"
        case "$left" in
          @*)
            scope="$(printf '%s' "$left" | sed 's/^@//' | cut -d' ' -f1)"
            left="$(printf '%s' "$left" | cut -d' ' -f2-)"
            ;;
        esac
        kern="$(printf '%s' "$left" | tr -d ' ')"
        [ -z "$kern" ] && continue
        matched_scope=1
        if [ -n "$scope" ]; then
          matched_scope=0
          case "$arch" in $scope) matched_scope=1 ;; esac
        fi
        if [ "$matched_scope" = 0 ]; then
          printf '  expect %-24s %-30s scoped %s (not standing for %s)\n' \
                 "$kern" "$insn" "$scope" "$arch"
          continue
        fi
        if kernel_body "$TMP/$b.$arch.s" "$kern" | grep -q -- "$insn"; then
          printf '  expect %-24s %-30s PRESENT\n' "$kern" "$insn"
        else
          printf '  expect %-24s %-30s MISSING  <-- OK? cell, FAILS the runner\n' "$kern" "$insn"
          EXPECT_MISSING=$((EXPECT_MISSING + 1))
          MISSING_LIST="$MISSING_LIST\n    $b [$arch] $kern did not emit $insn"
        fi
      done
    fi
  done
  echo
done

echo "=============================================================================="
echo "TIER C  device run (timed, with the CPU reference as oracle)"
echo "=============================================================================="

# ------------------------------------------------------------- baselines ---
# tools/bench/baselines.txt, one entry per line, whitespace-separated:
#   arch  driver-basename  metric  dir  value  sed-expression
# The sed expression must capture the number as \1 and contain NO literal
# spaces (use [[:space:]]+). dir=min => higher is better (fail below
# value*0.9); dir=max => lower is better (fail above value*1.1).
# Entries are arch-scoped: a gfx1031 run is never judged by gfx1201 pins.
BASELINE_FILE="$HERE/baselines.txt"
BASELINE_RC=0
EXPECT_RC=0
check_baseline() { # arch driver out_file
  [ -f "$BASELINE_FILE" ] || return 0
  local arch="$1" b="$2" out="$3" e_arch drv metric dir val pat got bad
  while read -r e_arch drv metric dir val pat; do
    case "${e_arch:-}" in ""|\#*) continue ;; esac
    [ "$drv" = "$b" ] || continue
    [ "$e_arch" = "$arch" ] || continue
    got="$(sed -nE "$pat" "$out" | head -1)"
    if [ -z "$got" ]; then
      echo "  BASELINE $b/$metric: metric NOT FOUND in output - the number"
      echo "        vanished or its line changed; a pinned metric that cannot be"
      echo "        read is a regression, not a pass."
      BASELINE_RC=6
      continue
    fi
    if awk -v g="$got" -v v="$val" -v d="$dir" 'BEGIN {
          if (d == "min") exit !(g >= v * 0.90);
          else             exit !(g <= v * 1.10);
        }'; then
      bad=""
    else
      bad="  REGRESSION"
      BASELINE_RC=6
    fi
    printf '  baseline %-13s %-11s pinned %-8s got %-10s (10%% tol)%s\n' \
      "$b" "$metric" "$(printf '%s %s' "$dir" "$val")" "$got" "$bad"
  done < "$BASELINE_FILE"
}

TIER_C_RC=0
declare -A BUILT_ANY=()
HIPCC="${KNJ_HIPCC:-}"
if [ -z "$HIPCC" ]; then
  # First try a hipcc that lives inside a ROCm tree (the runner's original
  # assumption). On this machine that tree has a working clang but no hipcc,
  # so this commonly comes up empty here — fall through to the PATH + known
  # exe search below.
  for a in "${ARCHS[@]}"; do
    r="$(rocm_lib_root_for_arch "$a")"
    [ -z "$r" ] && continue
    if [ -x "$r/bin/hipcc.exe" ]; then HIPCC="$r/bin/hipcc.exe"; break; fi
    if [ -x "$r/bin/hipcc" ]; then HIPCC="$r/bin/hipcc"; break; fi
  done
fi
# Fallback: a hipcc on PATH, then a small list of known-working hipcc
# executables on this machine. The PATH hipcc here (HIP 7.16.26323) is the one
# that actually links — the ROCm tree it lives next to has clang but no hipcc,
# which is why the ROCm-tree search above comes up empty.
if [ -z "$HIPCC" ] && command -v hipcc >/dev/null 2>&1; then
  HIPCC="$(command -v hipcc)"
fi
if [ -z "$HIPCC" ]; then
  for cand in /c/ROCm72/bin/hipcc.exe /c/ROCm72/bin/hipcc; do
    if [ -x "$cand" ]; then HIPCC="$cand"; break; fi
  done
fi

DRIVERS=()
for src in "${BENCHES[@]}"; do
  if is_device_driver "$src" || [[ "$(basename "$src" .hip)" == *_run ]]; then
    DRIVERS+=("$src")
  fi
done

if [ "${#DRIVERS[@]}" -eq 0 ]; then
  echo "  no tier C drivers found in $HERE."
elif [ -z "$HIPCC" ]; then
  echo "  UNMEASURED: no working hipcc for ${ARCHS[*]}."
  echo "  ${#DRIVERS[@]} driver(s) not built: $(printf '%s ' "${DRIVERS[@]##*/}")"
  echo "  A tier that did not run is reported, never omitted."
else
  echo "  hipcc: $HIPCC"
  echo "  The link line needs -nogpulib. Two real consequences, both hit already:"
  echo "    no device-side libm (no sqrt/exp inside a kernel), and no gridDim in"
  echo "    any kernel (gfx1201 needs __ockl_get_num_groups from the runtime that"
  echo "    -nogpulib skips). See docs/00-verified-facts.md section 8.5."
  echo
  for arch in "${ARCHS[@]}"; do
    libroot="$(rocm_lib_root_for_arch "$arch")"
    [ -z "$libroot" ] && continue
    libpath="$(find_rocm_lib_path "$libroot")"
    # hipcc needs a Windows link path for -L<libdir>. cygpath is the right tool;
    # a sed whose replacement ends in a backslash would escape its own delimiter.
    winroot="$(cygpath -w "$libpath" 2>/dev/null || echo "$libpath")"
    echo "---- $arch via $HIPCC (link path ${winroot}\\lib) ----"
    for src in "${DRIVERS[@]}"; do
      b="$(basename "$src" .hip)"
      printf '  %-14s ' "$b"
      # Per-source flags. rocWMMA drivers need their own --rocm-path and C++17;
      # every other driver must not receive them. `${EXTRA[@]+...}` because the
      # runner runs under set -u and an empty array expansion is an error there.
      EXTRA=()
      if is_rocwmma_driver "$src"; then
        read -r -a EXTRA <<< "$(rocwmma_flags_for "$libroot")"
      fi
      # -I<repo root> so a driver's `#include "src/..."` resolves, and the
      # matching .cpp files on the link line. Both are per-source and inert for
      # the drivers that do not use src/ (today: only q4k_stream_ffn.hip does).
      if ! "$HIPCC" ${EXTRA[@]+"${EXTRA[@]}"} -nogpulib -O2 --offload-arch="$arch" \
           -I"$(cd "$HERE/../.." && pwd)/kernels" \
           -I"$(cd "$HERE/../.." && pwd)" \
           -L"${winroot}\\lib" -lamdhip64 $(extra_sources_for "$src") "$src" \
           -o "$TMP/$b.$arch.exe" \
           >"$TMP/$b.$arch.err" 2>&1; then
        # A refusal here is often CORRECT: gfx1031 has no WMMA, so the _gfx12
        # builtin is rightly rejected. Only a driver that built for NO arch
        # at all is a real failure, and that is judged after the loop.
        printf 'REFUSED/UNBUILT  %s\n' \
          "$(grep -m1 -E 'error:' "$TMP/$b.$arch.err" | sed 's/.*error: //' | cut -c1-60)"
        continue
      fi
      BUILT_ANY[$(basename "$src")]=""
      BENCH_ARGS="$(bench_declared "$src" KNJ_BENCH_ARGS)"
      BENCH_REQ="$(bench_declared "$src" KNJ_BENCH_REQUIRES)"
      if [ -n "$BENCH_REQ" ] && [ -z "$BENCH_ARGS" ]; then
        printf 'NOT RUN   %s\n' "$BENCH_REQ"
        continue
      fi
      # No -D for the arch guard: hipcc re-spawns clang through a command
      # STRING, so a -D carrying quotes loses them to the inner shell and
      # the compiler receives a bare identifier. The environment does not
      # pass through that layer, so the guard is fed from there.
      # stdout is captured (tee) so the baseline check can read the
      # driver's own numbers; the tier C gate below is unchanged and
      # still judged on the DRIVER's exit code, not tees.
      ARGV=()
      [ -n "$BENCH_ARGS" ] && read -r -a ARGV <<< "$BENCH_ARGS"
      KNJ_BUILD_ARCH="$arch" "$TMP/$b.$arch.exe" ${ARGV[@]+"${ARGV[@]}"} | tee "$TMP/$b.$arch.out"
      rc=${PIPESTATUS[0]}
      if [ $rc -eq 6 ]; then
        # The driver declined to run: its code object targets a different arch
        # than the attached GPU. A skip, not a failure, and not a result —
        # such a binary runs clean, dispatches nothing, and prints confident
        # nonsense. Every driver refuses explicitly rather than being filtered.
        echo
        continue
      fi
      if [ $rc -eq 2 ]; then
        # A usage error, not a wrong answer: the driver wants arguments the gate
        # does not have. Say that, and say how to fix it, rather than blaming
        # the kernel for arithmetic it never performed.
        echo "  $b: USAGE ERROR (exit 2) — this driver needs arguments the gate"
        echo "        does not pass it. Declare them with '// KNJ_BENCH_ARGS: ...',"
        echo "        or '// KNJ_BENCH_REQUIRES: <reason>' if it cannot run here."
        TIER_C_RC=5
      elif [ $rc -ne 0 ]; then
        echo "  $b: TIER C FAILED (exit $rc) — the kernel does not compute the"
        echo "        right answer on the real device. Its timing is meaningless."
        TIER_C_RC=5
      else
        # Correctness passed: its numbers may now be judged against pins.
        check_baseline "$arch" "$b" "$TMP/$b.$arch.out"
      fi
      echo
    done
  done
fi

for src in "${DRIVERS[@]}"; do
  b="$(basename "$src")"
  if [ -z "${BUILT_ANY[$b]+set}" ]; then
    echo "  $b: built for NO arch in ${ARCHS[*]} — a real build failure,"
    echo "        not an arch-specific refusal."
    TIER_C_RC=5
  fi
done

if [ $TIER_A_RC -ne 0 ]; then
  echo
  echo "TIER A FAILED — the census above describes code that does not compute the"
  echo "right answer. Do not read any of it as a performance result."
fi
if [ $TIER_C_RC -ne 0 ]; then
  echo
  echo "TIER C FAILED — a timed kernel disagreed with its oracle on a real GPU."
fi
# A missing declared instruction is a capability that was claimed and not
# delivered, so it outranks a baseline regression (which is a slowdown, not a
# falsehood) -- but both are reported before either exit code is returned.
if [ "${EXPECT_MISSING:-0}" -ne 0 ]; then
  EXPECT_RC=7
  echo
  echo "TIER B EXPECTATION FAILED -- $EXPECT_MISSING declared instruction(s) were"
  echo "not emitted. A KNJ_EXPECT line is a contract, so this is a capability that"
  echo "was claimed and not delivered: the cell is OK? and the runner fails."
  printf '%b\n' "$MISSING_LIST"
  echo "Fix the kernel, or state the truth with an arch scope: '@gfx1031 <kernel> = <insn>'."
fi
[ $TIER_A_RC -ne 0 ] && exit $TIER_A_RC
[ $TIER_C_RC -ne 0 ] && exit $TIER_C_RC
[ $EXPECT_RC -ne 0 ] && exit $EXPECT_RC
if [ $BASELINE_RC -ne 0 ]; then
  echo
  echo "BASELINE REGRESSION - a pinned number in tools/bench/baselines.txt got"
  echo "more than 10% worse than the reference run. Re-pin ONLY with a reason."
fi
exit $BASELINE_RC
