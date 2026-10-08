#!/usr/bin/env bash
# kanjoos doctor — audit the machine before any measurement is believed.
#
#   tools/doctor/kanjoos_doctor.sh [--rocm PATH] [--arch ARCH ...] [--kpack DIR]
#
# This exists because of a specific, documented failure: an ROCm install whose
# hipcc points into a deleted toolchain, and which has no amdgcn/bitcode, made
# every ISA probe report REFUSED. The matrix printed a uniform wall of
# failures that read exactly like "this GPU has no matrix units" and "this
# architecture has no dot products". It was an install fault. Nothing in the
# output said so, because every tool involved reported only an exit code.
#
# The rule this enforces: a diagnostic must distinguish
#     MEASURED     we ran it and have a number
#     NOT PRESENT  the thing is absent and that is the finding
#     UNREADABLE   we could not look, and that is a gap, not a result
# and must never print a capability verdict derived from an install fault.
#
# EXIT CODES
#   0  no blocking problem
#   1  at least one BLOCKING problem
#   2  no ROCm installation found at all
#
# Every check prints one of: OK  BLOCKING  WARN  INFO  UNREADABLE

set -uo pipefail

ROCM_OVERRIDE=""
ARCHES=()
KPACK_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rocm)  ROCM_OVERRIDE="${2:-}"; shift 2 ;;
    --arch)  ARCHES+=("${2:-}"); shift 2 ;;
    --kpack) KPACK_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ "${#ARCHES[@]}" -eq 0 ] && ARCHES=(gfx1031 gfx1201)

BLOCKING=0
WARNINGS=0
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

row() {  # status name detail
  printf '  %-10s %-26s %s\n' "$1" "$2" "$3"
}
note() { printf '\n%s\n' "$1"; }
section() { printf '\n%s\n%s\n' "$1" "$(printf '=%.0s' $(seq 1 ${#1}))"; }

printf 'kanjoos doctor\n'
printf 'platform      : %s %s\n' "$(uname -s 2>/dev/null)" "$(uname -r 2>/dev/null)"
printf 'shell         : %s\n' "$BASH_VERSION"
printf 'targets       : %s\n' "${ARCHES[*]}"

# =====================================================================
section "1. ROCm installations"
# =====================================================================
ROOTS=()
if [ -n "$ROCM_OVERRIDE" ]; then
  ROOTS+=("$ROCM_OVERRIDE")
else
  for c in "${ROCM_PATH:-}" /opt/rocm /opt/rocm-* /g/ROCM* /c/ROCm*; do
    [ -n "$c" ] && [ -d "$c" ] && ROOTS+=("$c")
  done
fi
# de-duplicate, preserving order
UNIQ=(); for r in "${ROOTS[@]}"; do
  [ -n "$r" ] && { [ -e "$r/amdgcn" ] || [ -e "$r/lib" ] || [ -e "$r/bin" ]; } && \
  case " ${UNIQ[*]} " in *" $r "*) ;; *) UNIQ+=("$r") ;; esac
done
ROOTS=("${UNIQ[@]}")

if [ "${#ROOTS[@]}" -eq 0 ]; then
  row BLOCKING rocm "no ROCm tree found (looked in /opt/rocm*, /g/ROCM*, \$ROCM_PATH)"
  echo "  Nothing below can be measured. Install ROCm, or pass --rocm PATH."
  exit 2
fi
row OK installs "${#ROOTS[@]} found: ${ROOTS[*]}"

GOOD_CLANG=""
for R in "${ROOTS[@]}"; do
  printf '\n  --- %s\n' "$R"

  # --- version stamp -------------------------------------------------
  VER="unknown"
  for vf in "$R/.info/version" "$R/VERSION" "$R/share/rocm_version"; do
    if [ -f "$vf" ]; then VER="$(tr -d '\r\n' < "$vf" | head -c 60)"; break; fi
  done
  if [ "$VER" = "unknown" ] && [ -x "$R/bin/rocminfo" ]; then
    VER="$("$R/bin/rocminfo" 2>/dev/null | grep -m1 -i 'ROCk module' | cut -c1-40)"
  fi
  row INFO version "${VER:-no version stamp}"

  # --- device compiler -----------------------------------------------
  CL=""
  for c in "$R/lib/llvm/bin/clang" "$R/lib/llvm/bin/clang.exe" \
           "$R/bin/hipcc" "$R/llvm/bin/clang"; do
    [ -x "$c" ] && { CL="$c"; break; }
  done
  if [ -z "$CL" ]; then
    row BLOCKING compiler "no clang under lib/llvm/bin or bin/"
    BLOCKING=$((BLOCKING+1)); continue
  fi
  case "$CL" in *hipcc*) row INFO compiler "using hipcc (know its risks)" ;;
                   *)      row OK compiler "$(basename "$CL")" ;; esac
  CV="$("$CL" --version 2>&1 | head -1)"
  row INFO compiler_ver "${CV:0:70}"
  if [[ "$CL" == *hipcc* ]]; then
    # hipcc prints "HIP version: ...", not a clang banner. That is what a
    # working hipcc looks like, so demanding a clang string here would report
    # a healthy install as broken.
    row INFO compiler_ver "hipcc reports a HIP version, not a clang banner"
  elif [[ "$CV" == *clang* ]]; then
    :
  else
    row BLOCKING compiler_ver "did not identify as clang: ${CV:0:50}"
    BLOCKING=$((BLOCKING+1))
  fi

  # --- does the compiler actually COMPILE AND EMIT A KERNEL? -------------
cat > "$TMP/pre.hip" <<'EOF'
#define __global__ __attribute__((global))
extern "C" __global__ void knj_pre(float *o) { o[0] = 1.0f; }
EOF
  for A in "${ARCHES[@]}"; do
    F=""; case "$A" in gfx12*) F="-Xclang -target-feature -Xclang +wavefrontsize32";; esac
    OUT="$("$CL" --offload-arch="$A" -nogpuinc -nogpulib --cuda-device-only \
            -x hip -std=c++17 $F -S "$TMP/pre.hip" -o "$TMP/pre.s" 2>&1)"
    RC=$?
    if [ $RC -eq 0 ] && grep -q '\.amdhsa_next_free_vgpr' "$TMP/pre.s"; then
      row OK "compile $A" "kernel emitted"
      case "$CL" in *clang*) [ -z "$GOOD_CLANG" ] && GOOD_CLANG="$CL" ;; esac
    elif [ $RC -eq 0 ]; then
      row BLOCKING "compile $A" "exit 0 but NO KERNEL in the code object (silent no-op)"
      BLOCKING=$((BLOCKING+1))
    elif [ -z "$OUT" ]; then
      # The failure mode this whole tool exists for: non-zero exit, empty
      # stderr. That is never a capability verdict.
      row BLOCKING "compile $A" "non-zero exit with EMPTY diagnostic — install fault, NOT a capability result"
      BLOCKING=$((BLOCKING+1))
    else
      row BLOCKING "compile $A" "$(printf '%s' "$OUT" | grep -m1 -E 'error:' | cut -c1-60)"
      BLOCKING=$((BLOCKING+1))
    fi
  done

  # --- device libraries (needed to LINK, not to compile) -------------
  # Two layouts exist and checking only one produces a false BLOCKING:
  #   ROCm 7   <rocm>/amdgcn/bitcode
  #   ROCm 10  <rocm>/lib/llvm/amdgcn/bitcode
  # The ROCm 10 runtime-only trees on this machine use the second form, and
  # hipcc still cannot find it -- which is why the working link line below
  # passes -nogpulib and names amdhip64 by hand.
  DEVBC=""
  for cand in "$R/lib/llvm/amdgcn/bitcode" "$R/amdgcn/bitcode"; do
    if [ -d "$cand" ]; then DEVBC="$cand"; break; fi
  done
  if [ -n "$DEVBC" ]; then
    N=$(find "$DEVBC" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [ "$N" -gt 0 ]; then
      row OK device_libs "${DEVBC#$R/}: $N files"
      row WARN device_libs_path "hipcc does not resolve this location on its own"
      row INFO device_libs_fix "link with: -nogpulib -L'$R/lib' -lamdhip64"
      WARNINGS=$((WARNINGS+1))
    else
      row BLOCKING device_libs "${DEVBC#$R/} is EMPTY — compile works, LINK will not"
      BLOCKING=$((BLOCKING+1))
    fi
  else
    row BLOCKING device_libs "no amdgcn/bitcode under lib/llvm/ or root — this install cannot link"
    BLOCKING=$((BLOCKING+1))
  fi

  # --- HIP runtime headers ------------------------------------------
  if [ -f "$R/include/hip/hip_runtime.h" ]; then
    row OK hip_headers "include/hip/hip_runtime.h present"
  else
    row WARN hip_headers "no include/hip/hip_runtime.h — probe/bench tier A still works, HIP host code will not"
    WARNINGS=$((WARNINGS+1))
  fi

  # --- hipcc specifically: the known trap ----------------------------
  if [ -x "$R/bin/hipcc" ] || [ -x "$R/bin/hipcc.exe" ]; then
    HCC="$R/bin/hipcc"; [ -x "$R/bin/hipcc.exe" ] && HCC="$R/bin/hipcc.exe"
    HV="$("$HCC" --version 2>&1 | head -1)"
    if [ -n "$HV" ]; then
      row OK hipcc "responds: ${HV:0:52}"
    else
      row BLOCKING hipcc "no output at all — this is the empty-diagnostic failure"
      row INFO hipcc_why "check that the clang path baked into hipcc still exists on disk"
      BLOCKING=$((BLOCKING+1))
    fi
    # A hardcoded absolute path inside hipcc is the usual culprit.
    if command -v strings >/dev/null 2>&1; then
      STALE="$(strings "$HCC" 2>/dev/null | grep -oE '[A-Za-z]:\\\\[A-Za-z0-9_\\\\.-]*clang\.exe|/opt/[A-Za-z0-9_/.-]*clang' \
              | sort -u | while read -r q; do
                  q2="${q//\\//}"; [ -e "$q2" ] || echo "$q"
                done | head -3)"
      if [ -n "$STALE" ]; then
        row BLOCKING hipcc_paths "hardcoded clang path(s) that no longer exist: $(printf '%s ' $STALE)"
        BLOCKING=$((BLOCKING+1))
      fi
    fi

    # --- CROSS-TREE hipcc: worse than a dead path, because it looks alive ---
    # A hipcc that answers --version cleanly is NOT proof it compiles for the
    # arch you think it does. /g/ROCM10RT-gfx1031/bin/hipcc.exe on this machine
    # responds perfectly and then execs
    #   G:\ROCM10RT-gfx1201\lib\llvmin\clang.exe
    # -- a DIFFERENT, live ROCm tree. Every gfx1031 build silently becomes a
    # gfx1201 build. The binary then runs on the gfx1201 card, dispatches
    # nothing, and reports confident nonsense (2327% of peak was observed).
    # "no link path" is the honest diagnosis here; "absent" would be a lie.
    if command -v strings >/dev/null 2>&1; then
      XCLANG="$(strings "$HCC" 2>/dev/null         | grep -oE '[A-Za-z]:\\[A-Za-z0-9_\\.-]*/lib/llvm/bin/clang\.exe'         | sort -u | head -1)"
      if [ -n "$XCLANG" ]; then
        MYTAG="$(echo "$R" | sed 's|.*/||')"
        XTAG="$(echo "$XCLANG" | sed 's|.*/||; s|\lib.*||')"
        if [ "$XTAG" != "$MYTAG" ]; then
          row BLOCKING hipcc_crosstree "this hipcc execs clang from a DIFFERENT ROCm tree"
          row INFO hipcc_crosstree_why "  this tree : $R"
          row INFO hipcc_crosstree_why "  execs     : $XCLANG  (tree: $XTAG)"
          row INFO hipcc_crosstree_why "  every build from here is silently $XTAG, not $MYTAG"
          BLOCKING=$((BLOCKING+1))
        else
          row OK hipcc_crosstree "hipcc execs clang from its own tree"
        fi
      else
        row INFO hipcc_crosstree "no hardcoded clang path found to cross-check"
      fi
    fi
  fi
done

if [ -n "$GOOD_CLANG" ]; then
  row OK device_clang "usable: $GOOD_CLANG"
  row INFO export "export KNJ_CLANG=$GOOD_CLANG   # then tools/isa_probe and tools/bench use it"
else
  row BLOCKING device_clang "no working device compiler found in any installation"
  exit 1
fi

# =====================================================================
section "2. Kernel packs"
# =====================================================================
KPACK_DIRS=()
if [ -n "$KPACK_OVERRIDE" ]; then
  KPACK_DIRS+=("$KPACK_OVERRIDE")
else
  for d in "${KNJ_KPACK_DIR:-}" "$HOME/kpack" /c/Users/rr/OneDrive/Desktop/kpack \
           /opt/rocm/lib/llvm/amdgcn/bitcode /usr/share/kpack; do
    [ -n "$d" ] && [ -d "$d" ] && KPACK_DIRS+=("$d")
  done
fi
FOUND_PACK=0
for D in "${KPACK_DIRS[@]}"; do
  N=$(find "$D" -maxdepth 1 -name '*.kpack' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$N" -gt 0 ]; then
    FOUND_PACK=1
    printf '  %-10s %-26s %s\n' OK "kpack dir" "$D ($N packs)"
    for a in "${ARCHES[@]}"; do
      M=$(find "$D" -maxdepth 1 -name "*${a}*.kpack" 2>/dev/null | wc -l | tr -d ' ')
      if [ "$M" -gt 0 ]; then
        row OK "pack for $a" "$(find "$D" -maxdepth 1 -name "*${a}*.kpack" -printf '%f ' 2>/dev/null)"
      else
        row WARN "pack for $a" "none — kernels for this arch must come from source, not a pack"
        WARNINGS=$((WARNINGS+1))
      fi
    done
  fi
done
[ "$FOUND_PACK" = 0 ] && {
  row WARN kernel_packs "no .kpack found in any known directory"
  row INFO kernel_packs_note "blas is the only pack the engine treats as a measured reference; absence is survivable"
  WARNINGS=$((WARNINGS+1))
}

# =====================================================================
section "3. ReBAR / large BAR"
# =====================================================================
# kanjoos.toml sets require_rebar = true, so this is not optional. But it is
# also the check most likely to be UNREADABLE over a non-interactive shell,
# and an unreadable check must never be reported as "disabled".
REQ_REBAR=$(grep -E '^require_rebar' "$(dirname "$0")/../../kanjoos.toml" 2>/dev/null | head -1 | cut -d= -f2 | tr -d ' ')
REQ_REBAR="${REQ_REBAR:-true}"

if [ -d /sys/bus/pci/devices ]; then
  # Linux: the VRAM BAR aperture is readable from sysfs when we are root or
  # the resource file is world-readable.
  GPU=""
  for d in /sys/bus/pci/devices/*/; do
    cls=$(cat "$d/class" 2>/dev/null)
    case "$cls" in 0x0300*) GPU="$d"; break ;; esac
  done
  if [ -z "$GPU" ]; then
    row UNREADABLE rebar "no display-class PCI device visible in /sys/bus/pci"
  elif [ ! -r "$GPU/resource" ]; then
    row UNREADABLE rebar "$GPU/resource not readable (need root or relaxed perms)"
    row INFO rebar_why "this is a GAP, not a finding — do not record it as 'ReBAR disabled'"
  else
    APERTURE=$(awk 'BEGIN{m=0} /^0x/ {v=strtonum($0); if (v>m) m=v} END{print m}' "$GPU/resource" 2>/dev/null)
    MB=$((APERTURE / 1048576))
    if [ "$MB" -ge 256 ]; then
      row OK rebar "largest BAR = ${MB} MiB (>= 256 MiB, ReBAR plausibly on)"
      row INFO rebar_caveat "size is NECESSARY, not sufficient; confirm with lspci -vv ReBAR=on"
    else
      if [ "$REQ_REBAR" = "true" ]; then
        row BLOCKING rebar "largest BAR = ${MB} MiB (< 256 MiB) and require_rebar = true"
        BLOCKING=$((BLOCKING+1))
      else
        row WARN rebar "largest BAR = ${MB} MiB"
        WARNINGS=$((WARNINGS+1))
      fi
    fi
  fi
else
  row UNREADABLE rebar "not a Linux host — BAR aperture cannot be read from here"
  row INFO rebar_how "read it yourself and paste it in: lspci -vv -s <gpu> | grep -i -A2 resizable-bar"
  row INFO rebar_how "Windows: DeviceIoControl(SYSTEM_INFO) via a privileged helper, or the BIOS"
fi

# =====================================================================
section "4. Tooling the harnesses need"
# =====================================================================
HC=""
for c in "${KNJ_HOSTCXX:-}" g++ clang++; do
  [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 && { HC="$c"; break; }
done
if [ -n "$HC" ]; then
  row OK host_cxx "$HC ($("$HC" --version 2>&1 | head -1 | cut -c1-46))"
  row INFO host_cxx_note "required for tools/bench tier A (CPU-reference correctness checks)"
else
  row WARN host_cxx "no host C++ compiler — bench tier A cannot run"
  WARNINGS=$((WARNINGS+1))
fi

if command -v bash >/dev/null 2>&1; then
  row OK bash "$BASH_VERSION"
fi

# =====================================================================
section "5. Verdict"
# =====================================================================
printf '  blocking : %d\n  warnings : %d\n' "$BLOCKING" "$WARNINGS"
if [ "$BLOCKING" -gt 0 ]; then
  cat <<EOF

  DO NOT TRUST ANY MEASUREMENT UNTIL THE BLOCKING ITEMS ARE FIXED.
  Every one of them is an install fault, and each of them has previously
  presented itself as a hardware limitation.
EOF
  exit 1
fi
cat <<EOF

  Install audit clean. What this does NOT establish:
    - that any kernel is CORRECT      (tools/bench/run_bench.sh tier A)
    - what the hardware can do        (tools/isa_probe/run_isa_probe.sh)
    - how fast anything is            (needs a real GPU attached; tier C)
EOF
exit 0