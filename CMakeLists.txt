# CMakeLists.txt -- kanjoos root build.
#
# Build layout follows docs/07-build-platforms.md section 2: host-only ISO
# C++17 everywhere outside src/device/, kernels/ and src/platform/; HIP device
# code only behind the HIP language. The C2 host library and its unit tests must
# build and run on a machine with NO GPU and NO working hipcc, so the HIP path is
# an OPTION, not a hard requirement.

cmake_minimum_required(VERSION 3.20)
project(kanjoos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Per-arch flag tables and the sanitizer helper. Neither of these touches HIP.
include(cmake/arch.cmake)
include(cmake/sanitize.cmake)

# Primary target: gfx1031 (RDNA2) on Windows 11. gfx1201 (RDNA4) is the
# secondary/dev-bench arch and stays fully supported; pass -DKNJ_ARCH=gfx1201
# to build for it explicitly.
if(NOT DEFINED KNJ_ARCH)
  set(KNJ_ARCH "gfx1031" CACHE STRING "Target GPU arch: gfx1031 (primary) or gfx1201")
  set_property(CACHE KNJ_ARCH PROPERTY STRINGS gfx1031 gfx1201)
endif()

# Derived per-arch flag lists (used by the HIP and device-compile paths).
knj_arch_flags(${KNJ_ARCH} KNJ_ARCH_FLAGS)
knj_link_flags(KNJ_LINK_FLAGS)

# Set KNJ_ARCH_NAME as a CMake variable for use in subdirectory CMakeLists.txt
# This must match what cmake/arch.cmake sets via compiler flags.
set(KNJ_ARCH_NAME "${KNJ_ARCH}")

# Compiled-in isa_probe results. rocm.cmake re-derives these when HIP is
# enabled; these defaults keep a host-only build (KNJ_ENABLE_DEVICE=OFF)
# consistent with KNJ_ARCH so the C2 arch guard still has a correct baseline.
if(NOT DEFINED KNJ_HAS_WMMA)
  if(KNJ_ARCH STREQUAL "gfx1201")
    set(KNJ_HAS_WMMA 1)
  else()
    set(KNJ_HAS_WMMA 0)
  endif()
endif()
if(NOT DEFINED KNJ_HAS_FP8)
  if(KNJ_ARCH STREQUAL "gfx1201")
    set(KNJ_HAS_FP8 1)
  else()
    set(KNJ_HAS_FP8 0)
  endif()
endif()
if(NOT DEFINED KNJ_HAS_DOT_ASM)
  set(KNJ_HAS_DOT_ASM 1)   # both arches via inline asm
endif()

# ---------------------------------------------------------------------------
# HIP / device path -- optional, so a host-only machine still builds the C2
# unit tests. rocm.cmake finds hipcc and sets KNJ_HAS_DEVICE_BUILD.
# ---------------------------------------------------------------------------
option(KNJ_ENABLE_DEVICE "Build HIP device code and device integration tests" ON)
# FORCE matters: without it a device build that cached ON would keep
# KNJ_HAS_DEVICE_BUILD=ON forever, and a later -DKNJ_ENABLE_DEVICE=OFF
# configure (the CI script's host-only job) would silently keep building the
# integration subdirectory. The cache entry is rewritten every configure:
# default OFF, then rocm.cmake flips it ON (FORCE) only when hipcc was found.
set(KNJ_HAS_DEVICE_BUILD OFF CACHE INTERNAL "set by rocm.cmake on success" FORCE)
if(KNJ_ENABLE_DEVICE)
  find_program(KNJ_HIPCC_CANDIDATE NAMES hipcc hipcc.exe
    HINTS "$ENV{ROCM_PATH}/bin" "/g/ROCM10RT-gfx1201/bin")
  if(KNJ_HIPCC_CANDIDATE)
    include(cmake/rocm.cmake)
  else()
    message(STATUS "knj: no hipcc found; building host-only (KNJ_ENABLE_DEVICE=ON "
                   "but ROCm absent). Device integration test will be skipped.")
  endif()
endif()

# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
# src/device -- the C2 host library (DeviceCaps + Backend + arch guard).
# Always built; it is host C++17 and needs no GPU.
add_subdirectory(src/device)

# C2 host unit tests.
add_subdirectory(tests/unit)

# Device integration test: only when hipcc is usable. We do NOT call
# enable_language(HIP) -- cmake's HIP module is incompatible with this
# toolchain (mixing Clang HIP with MinGW host on Windows; see cmake/rocm.cmake).
# The integration test is built with an explicit hipcc add_custom_command instead.
if(KNJ_HAS_DEVICE_BUILD)
  add_subdirectory(tests/integration)
endif()
