# CMakeLists.txt -- kanjoos root build.
#
# Build layout follows docs/07-build-platforms.md section 2: host-only ISO
# C++17 everywhere outside src/device/, kernels/ and src/platform/; HIP device
# code only behind the HIP language. The C2 host library and its unit tests must
# build and run on a machine with NO GPU and NO working hipcc, so the HIP path is
# an OPTION, not a hard requirement.

cmake_minimum_required(VERSION 3.20)
project(kanjoos LANGUAGES CXX)

# Tests are registered with add_test() in the subdirectories. enable_testing()
# has to run at the TOP level too, or ctest finds no CTestTestfile.cmake in the
# build root and reports "No tests were found" while the test executables sit
# there unbuilt and unrun -- a check that silently does not happen.
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Per-arch flag tables and the sanitizer helper. Neither of these touches HIP.
include(cmake/arch.cmake)
include(cmake/sanitize.cmake)

# Target order is the one AGENTS.md section 1 states: gfx1201 (RDNA4) on
# Windows 11 first, then gfx1201 Linux, then gfx1031 (RDNA2). gfx1031 is fully
# supported and stays selected with -DKNJ_ARCH=gfx1031.
if(NOT DEFINED KNJ_ARCH)
  # gfx1201 is the first target (AGENTS.md section 1: gfx1201 on Windows 11 first,
  # then gfx1201 Linux, then gfx1031). The previous default was gfx1031, which
  # contradicted the stated target order; gfx1031 stays fully supported and is
  # selected with -DKNJ_ARCH=gfx1031.
  set(KNJ_ARCH "gfx1201" CACHE STRING "Target GPU arch: gfx1201 (first target) or gfx1031")
  set_property(CACHE KNJ_ARCH PROPERTY STRINGS gfx1201 gfx1031)
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

# ---------------------------------------------------------------------------
# Host runtime -- the engine that loads a GGUF and runs it, plus the C21
# profiler and the C24 platform probe it reports through.
#
# This target did not exist: src/cli, src/model, src/loader, src/tokenizer,
# src/util, src/platform and src/profiler were all written, and nothing in the
# build referenced any of them, so the repository could not produce a single
# executable that loads a model. Host-only ISO C++17, no HIP, no device code.
# ---------------------------------------------------------------------------
add_library(knj_runtime STATIC
    src/platform/memprobe.cpp
    src/profiler/profiler.cpp
    src/util/parallel.cpp
    src/loader/gguf.cpp
    src/loader/dequant.cpp
    src/tokenizer/tokenizer.cpp
    src/tokenizer/unicode_ranges.cpp
    src/model/model.cpp
    src/model/qwen35.cpp
)
target_include_directories(knj_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(knj_runtime PUBLIC cxx_std_17)
find_package(Threads REQUIRED)
target_link_libraries(knj_runtime PUBLIC Threads::Threads)

add_executable(kanjoos-run src/cli/main.cpp)
target_link_libraries(kanjoos-run PRIVATE knj_runtime)
target_compile_features(kanjoos-run PRIVATE cxx_std_17)
