#!/usr/bin/env bash
# run_i7.sh -- build and run both legs of the I7 bit-identity test.
#
# Two programs, because they prove different things:
#   i7_bit_identity.py   the CONTRACT: what the canonical reduction order is,
#                        and that the wrong orders fail. Runs anywhere.
#   i7_hip.hip            the DEVICE: that a real GPU, a real PCIe hop and a
#                        real pinned-host bank reproduce the resident path in
#                        every bit. Needs gfx1201.
#
# Exit status is the test's own, not the last command's: a runner that reports
# success after a failed test is worse than no runner.
#
#   0   both legs pass
#   6   arch mismatch              7   arch guard not configured
#   10  a positive path diverged   11  a negative control did not fire
#   12  oracle or device failure
#   1   build failed

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${KNJ_BUILD_ARCH:-gfx1201}"
ROCM_ROOT="${KNJ_ROCM_ROOT:-/g/ROCM10RT-gfx1201}"
ROCM_LIB='G:\ROCM10RT-gfx1201\lib'

echo "=== I7 leg 1/3: host reference ==================================="
python "$HERE/i7_bit_identity.py"
host_rc=$?
echo "host reference exit=$host_rc"
[ "$host_rc" -ne 0 ] && exit "$host_rc"

echo
echo "=== I7 leg 2/3: build the GPU test ==============================="
HIPCC="$ROCM_ROOT/bin/hipcc.exe"
[ -x "$HIPCC" ] || HIPCC="$ROCM_ROOT/bin/hipcc"
if [ ! -x "$HIPCC" ]; then
  echo "no hipcc under $ROCM_ROOT -- set KNJ_ROCM_ROOT" >&2
  exit 1
fi
# -nogpulib: no device libm (so __expf, a hardware instruction, is mandatory)
# --offload-arch: an arch-mismatched binary still RUNS on this card and lies.
#   The arch guard in the program is the real defence; the flag is hygiene.
"$HIPCC" -nogpulib -O2 --offload-arch="$ARCH" \
  -L"$ROCM_LIB" -lamdhip64 "$HERE/i7_hip.hip" -o "$HERE/i7_hip.exe" \
  2>&1 | grep -Ev "argument unused|generated (warning|1 warning)" | grep -E "error|warning: " | head -5
build_rc=${PIPESTATUS[0]}
echo "build exit=$build_rc"
[ "$build_rc" -ne 0 ] && exit 1

echo
echo "=== I7 leg 3/3: GPU leg =========================================="
# Prove the guard works before trusting a run. An arch-mismatched binary
# executes, dispatches nothing, and leaves outputs untouched -- the exact
# failure mode that cost this repo a day in section 8.7.
KNJ_BUILD_ARCH=gfx1031 "$HERE/i7_hip.exe" >/dev/null 2>&1
guard_rc=$?
echo "arch guard: mismatched arch returned $guard_rc (expect 6)"
[ "$guard_rc" -ne 6 ] && { echo "ARCH GUARD DID NOT FIRE"; exit 6; }

KNJ_BUILD_ARCH="$ARCH" "$HERE/i7_hip.exe"
gpu_rc=$?
echo "gpu leg exit=$gpu_rc"
exit "$gpu_rc"