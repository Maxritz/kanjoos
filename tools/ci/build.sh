#!/usr/bin/env bash
# ============================================================================
# tools/ci/build.sh - CI build for the kanjoos C2 Device Abstraction Layer.
#
# Builds the HOST-ONLY C2 library (knj_device) and its unit tests
# (test_c2_device) on Windows (MinGW64/MSYS2) and Linux. The HIP device path
# is NOT built by this script - CI for the device path is a separate job that
# requires a ROCm install and a real GPU.
#
# Usage:
#   tools/ci/build.sh                  # auto-detect platform, build gfx1031 (primary)
#   tools/ci/build.sh gfx1201          # build for a specific arch
#   tools/ci/build.sh --host-only      # host-only, no ROCm check
#   tools/ci/build.sh --clean          # remove build/ before building
#
# EXIT CODES
#   0  build succeeded, all unit tests pass
#   1  build or tests failed
#   2  unsupported platform
#   3  no host C++ compiler found
#
# Platform notes:
#   Windows (MSYS2/MinGW64): host compiler is g++ from Strawberry Perl or MSYS2.
#     cmake + ninja build. KNJ_ARCH_NAME and KNJ_HAS_* are passed as -D flags.
#   Linux: host compiler is g++ or clang++. Same cmake + ninja build.
#     ROCm may or may not be present; this script builds host-only regardless.
# ============================================================================

set -uo pipefail

# --- defaults ----------------------------------------------------------------
# Primary target: gfx1031 (RDNA2) on Windows 11. gfx1201 stays supported.
ARCH=""
HOST_ONLY=0
DO_CLEAN=0

# Flags may appear in any position; the first non-flag arg is the arch.
for arg in "$@"; do
    case "$arg" in
        --host-only) HOST_ONLY=1 ;;
        --clean)     DO_CLEAN=1 ;;
        --*)         echo "Unknown arg: $arg" >&2; exit 2 ;;
        *)
            if [ -z "$ARCH" ]; then
                ARCH="$arg"
            else
                echo "Unknown arg: $arg" >&2; exit 2
            fi
            ;;
    esac
done
ARCH="${ARCH:-gfx1031}"

# --- platform detection ------------------------------------------------------
UNAME="$(uname -s 2>/dev/null || echo Unknown)"
case "$UNAME" in
    MINGW*|MSYS*) PLATFORM="windows" ;;
    Linux*)       PLATFORM="linux" ;;
    *)            echo "Unsupported platform: $UNAME" >&2; exit 2 ;;
esac

echo "=============================================================================="
echo "kanjoos C2 Device Abstraction Layer - CI build"
echo "  platform : $PLATFORM ($UNAME)"
echo "  arch     : $ARCH"
if [ "$HOST_ONLY" = "1" ]; then
    echo "  host-only: yes"
else
    echo "  host-only: no"
fi
echo "=============================================================================="
echo

# --- locate host compiler ----------------------------------------------------
HOSTCXX="${KNJ_HOSTCXX:-}"
if [ -z "$HOSTCXX" ]; then
    for cand in g++ clang++ c++/Strawberry/c/bin/g++.exe; do
        if command -v "$cand" >/dev/null 2>&1; then
            HOSTCXX="$cand"
            break
        fi
    done
fi

if [ -z "$HOSTCXX" ] || ! command -v "$HOSTCXX" >/dev/null 2>&1; then
    echo "ERROR: no host C++ compiler found."
    echo "  Set KNJ_HOSTCXX to the compiler path, or install g++/clang++."
    exit 3
fi

HOSTCXX_VERSION="$("$HOSTCXX" --version 2>/dev/null | head -1)"
echo "host compiler : $HOSTCXX ($HOSTCXX_VERSION)"

# --- locate cmake + ninja ----------------------------------------------------
CMAKE="${KNJ_CMAKE:-cmake}"
NINJA="${KNJ_NINJA:-ninja}"

for cmd in "$CMAKE" "$NINJA"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: $cmd not found. Install cmake and ninja."
        exit 3
    fi
done

echo "cmake         : $($CMAKE --version 2>/dev/null | head -1)"
echo "ninja         : $($NINJA --version 2>/dev/null | head -1)"
echo

# --- ROCm check (optional) ---------------------------------------------------
if [ "$HOST_ONLY" = "1" ]; then
    echo "ROCm: not required (host-only mode)"
    ROCM_ROOT=""
else
    ROCM_ROOT=""
    for cand in \
        "${ROCM_PATH:-}" \
        "/g/ROCM10RT-gfx1201" \
        "/g/ROCM10RT-gfx1031" \
        "/opt/rocm" \
        "/opt/rocm-6.2.0" \
        "C:/Program Files/AMD ROCm/6.2.0"; do
        if [ -n "$cand" ]; then
            if [ -x "$cand/bin/hipcc.exe" ] || [ -x "$cand/bin/hipcc" ]; then
                ROCM_ROOT="$cand"
                break
            fi
        fi
    done
    if [ -n "$ROCM_ROOT" ]; then
        echo "ROCm         : $ROCM_ROOT (present, but CI builds host-only)"
    else
        echo "ROCm         : not found (CI builds host-only - this is expected)"
    fi
fi
echo

# --- project root ------------------------------------------------------------
# $0 is tools/ci/build.sh. Go up two levels to the project root.
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT_ROOT="$HERE"
# CI uses its OWN build dir (build-ci) so a host-only CI job never clobbers a
# developer's device-enabled build/ tree (and vice versa).
BUILD_DIR="${KNJ_CI_BUILD_DIR:-$PROJECT_ROOT/build-ci}"

# --- clean (optional) -------------------------------------------------------
if [ "$DO_CLEAN" = "1" ]; then
    echo "--- cleaning build/ ---"
    rm -rf "$BUILD_DIR"
    echo
fi

# --- configure ---------------------------------------------------------------
echo "--- cmake configure ---"
CONFIGURE_CMD=("cmake" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_CXX_COMPILER="$HOSTCXX" \
    -DKNJ_ARCH="$ARCH" \
    -DKNJ_ENABLE_DEVICE=OFF)

if ! "${CONFIGURE_CMD[@]}" 2>&1 | tail -30; then
    echo
    echo "ERROR: cmake configure failed."
    exit 1
fi
echo

# --- build -------------------------------------------------------------------
echo "--- cmake build (knj_device + test_c2_device) ---"
if ! "$NINJA" -C "$BUILD_DIR" knj_device test_c2_device 2>&1 | tail -30; then
    echo
    echo "ERROR: build failed."
    exit 1
fi
echo

# --- run unit tests ---------------------------------------------------------
echo "--- running C2 unit tests ---"
TEST_BIN="$BUILD_DIR/tests/unit/test_c2_device"
if [ ! -x "$TEST_BIN" ]; then
    if [ -x "$TEST_BIN.exe" ]; then
        TEST_BIN="$TEST_BIN.exe"
    else
        echo "ERROR: test binary not found at $TEST_BIN"
        exit 1
    fi
fi

export KNJ_BUILD_ARCH="$ARCH"
TEST_OUTPUT="$("$TEST_BIN" 2>&1)"
TEST_RC=$?
echo "$TEST_OUTPUT" | tail -15
echo

if [ $TEST_RC -ne 0 ]; then
    echo "ERROR: unit tests failed (exit $TEST_RC)."
    exit 1
fi

# --- summary ----------------------------------------------------------------
echo "=============================================================================="
echo "CI build PASSED"
echo "  platform     : $PLATFORM"
echo "  arch         : $ARCH"
echo "  host compiler: $HOSTCXX"
echo "  tests run    : $(echo "$TEST_OUTPUT" | grep 'tests run:' | awk '{print $NF}')"
echo "  tests passed : $(echo "$TEST_OUTPUT" | grep 'tests passed:' | awk '{print $NF}')"
echo "  tests failed : $(echo "$TEST_OUTPUT" | grep 'tests failed:' | awk '{print $NF}')"
echo "  build dir    : $BUILD_DIR"
echo "=============================================================================="
exit 0
