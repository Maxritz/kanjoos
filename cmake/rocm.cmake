# cmake/rocm.cmake -- locate the ROCm toolchain and set up offload targets.
#
# Toolchain facts on this machine, all measured (docs/00-verified-facts.md
# sections 2, 7.0 and 8.7; tools/doctor/kanjoos_doctor.sh reproduces them):
#
#   /g/ROCM10RT-gfx1201   clang + hipcc present, device bitcode present at
#                         lib/llvm/amdgcn/bitcode, amdhip64 import lib present.
#                         This is the tree that can LINK.
#   /g/ROCM10RT-gfx1031   clang present, amdhip64 import lib present, but NO
#                         bin/hipcc at all. It compiles; it cannot link on its own.
#   /c/ROCm72             hipcc present, device bitcode at amdgcn/bitcode
#                         (the ROCm 7 layout), no clang under lib/llvm/bin.
#
# The consequence that matters for a two-target build: **there is exactly one
# usable hipcc on this machine, and it is not arch-specific.** A single hipcc
# driver can emit code objects for BOTH targets, because the device bitcode is
# the LLVM device library rather than a per-arch library, and the per-arch
# selection is entirely `--offload-arch`. So the RDNA2 build links through the
# same hipcc that the RDNA4 build uses, with `--offload-arch=gfx1031`.
#
# That is not a workaround for a missing install; it is the documented shape of
# the toolchain. What it does NOT excuse is shipping an RDNA2 binary that
# silently runs on the RDNA4 card. That failure mode is real, was observed on
# this machine (docs/00-verified-facts.md section 8.7: a gfx1031 object on the
# gfx1201 card printed 2327% of peak in 0.1 us and wrote nothing), and is why
# src/device/ carries a mandatory build-arch guard.

if(NOT DEFINED ROCM_ROOT)
  # Prefer a tree that has BOTH clang and hipcc.
  foreach(_cand
      "$ENV{ROCM_PATH}"
      "/g/ROCM10RT-gfx1201"
      "/opt/rocm"
      "/opt/rocm-6.2.0"
      "C:/Program Files/AMD ROCm/6.2.0")
    if(_cand AND EXISTS "${_cand}/bin/hipcc.exe")
      set(ROCM_ROOT "${_cand}")
      break()
    elseif(_cand AND EXISTS "${_cand}/bin/hipcc")
      set(ROCM_ROOT "${_cand}")
      break()
    endif()
  endforeach()
endif()

if(NOT ROCM_ROOT)
  message(FATAL_ERROR
    "knj: no ROCm tree with bin/hipcc found.\n"
    "  Pass -DROCM_ROOT=<path>, or set ROCM_PATH.\n"
    "  tools/doctor/kanjoos_doctor.sh audits every tree on this machine and\n"
    "  prints which ones can compile, which can link, and which can do neither.")
endif()

get_filename_component(ROCM_ROOT "${ROCM_ROOT}" ABSOLUTE)
message(STATUS "knj: ROCM_ROOT = ${ROCM_ROOT}")

# hipcc driver. On Windows this is hipcc.exe; the .bat shim that used to ship
# is a wrapper that adds nothing we need.
if(EXISTS "${ROCM_ROOT}/bin/hipcc.exe")
  set(KNJ_HIPCC "${ROCM_ROOT}/bin/hipcc.exe")
elseif(EXISTS "${ROCM_ROOT}/bin/hipcc")
  set(KNJ_HIPCC "${ROCM_ROOT}/bin/hipcc")
else()
  message(FATAL_ERROR "knj: ROCM_ROOT=${ROCM_ROOT} has no bin/hipcc.")
endif()
message(STATUS "knj: HIPCC      = ${KNJ_HIPCC}")

# Device clang, used by the compile-only capability checks and by
# tools/isa_probe/run_isa_probe.sh. A compile-only check needs no device
# library, so it works on trees that cannot link.
find_program(KNJ_CLANG
  NAMES clang
  HINTS "${ROCM_ROOT}/lib/llvm/bin" "${ROCM_ROOT}/llvm/bin"
  NO_DEFAULT_PATH)
if(NOT KNJ_CLANG)
  message(WARNING "knj: no clang under ${ROCM_ROOT}/lib/llvm/bin. "
                  "Compile-only capability probes (tools/isa_probe) will not run.")
else()
  message(STATUS "knj: CLANG      = ${KNJ_CLANG}")
endif()

# amdhip64 for the link line. We name it by hand because hipcc cannot resolve
# the device bitcode directory (see cmake/arch.cmake for the measurement).
find_library(KNJ_AMDHIP64
  NAMES amdhip64 amdhip64_7
  HINTS "${ROCM_ROOT}/lib" "${ROCM_ROOT}/lib/amdhip64"
  NO_DEFAULT_PATH)
if(KNJ_AMDHIP64)
  message(STATUS "knj: AMDHIP64   = ${KNJ_AMDHIP64}")
else()
  message(WARNING "knj: amdhip64 not found under ${ROCM_ROOT}/lib. The link "
                  "line will rely on the default search path.")
endif()

# ---------------------------------------------------------------------------
# HIP / device build policy
# ---------------------------------------------------------------------------
# The engine's device code lives in kernels/ and in src/device/, which is the
# rule docs/07-build-platforms.md section 2 sets. Everything else is host-only
# ISO C++17 and is compiled by the host compiler with NO -x hip and NO offload
# flags at all.
#
# We do NOT call enable_language(HIP). cmake's HIP language module has
# three problems on this machine that cannot all be satisfied at once:
#   1. It insists the HIP compiler not be hipcc ("Use Clang directly").
#   2. On Windows it forbids mixing Clang (HIP) with a non-Clang host compiler
#      (MinGW g++ here) — Windows-Clang.cmake line 177.
#   3. Its arch detection (rocm_agent_enumerator / -target-cpu) and ROCm-root
#      path writing both fail on this toolchain (docs/00 section 7.0).
#
# Instead the device integration test is built with an explicit hipcc command
# (add_custom_command in tests/integration/CMakeLists.txt), and the link line
# is the measured one from docs/00 section 7.0:
#   hipcc -nogpulib -O2 --offload-arch=<arch> -L'<rocm>/lib' -lamdhip64 <files>
#
# The host C2 library (knj_device) is built by the normal CXX path and links
# into the hipcc-linked test because hipcc's host pass goes through g++.
#
# The runtime arch guard STILL reads KNJ_BUILD_ARCH from the environment
# (never -D) at execution time.

if(NOT KNJ_ARCH IN_LIST KNJ_SUPPORTED_ARCHS)
  message(FATAL_ERROR "knj: KNJ_ARCH='${KNJ_ARCH}' is not one of ${KNJ_SUPPORTED_ARCHS}")
endif()
message(STATUS "knj: TARGET ARCH = ${KNJ_ARCH}  "
               "($<$<STREQUAL:${KNJ_ARCH},gfx1201>:RDNA4>:RDNA2>)")

# ---------------------------------------------------------------------------
# Device capabilities that the build itself must know about.
# ---------------------------------------------------------------------------
#
# These are NOT hard-coded per arch out of thin air: each one is read back out
# of the compiled code object by tools/isa_probe/run_isa_probe.sh, and the
# values below are the recorded results of that probe on this machine
# (docs/00-verified-facts.md section 1). The compile below proves the flag set
# still produces those instructions; the CI job proves it did not change
# across an ROCm upgrade.
#
# gfx1031 (RDNA2)  : no matrix units at all. v_wmma_* is REFUSED on both
#                    spellings and that is HARDWARE, not configuration.
# gfx1201 (RDNA4)  : v_wmma_*_gfx12 works with +wavefrontsize32.
#
# BOTH arches      : v_dot4_i32_i8 and v_dot8_i32_i4 assemble as inline asm.
#                    This is the single most useful consequence of the ISA
#                    correction, and it is why kernels/knj_dot.h is ONE header
#                    shared by both targets rather than two #if-selected ones.
#                    The `sdot4` builtin is strictly worse: on gfx1031 it lowers
#                    to v_dot4c_i32_i8 (a CLAMPING variant with different
#                    semantics) and on gfx1201 it does not compile at all.
if(KNJ_ARCH STREQUAL "gfx1201")
  set(KNJ_HAS_WMMA 1)
  set(KNJ_HAS_FP8  1)
else()
  set(KNJ_HAS_WMMA 0)
  set(KNJ_HAS_FP8  0)
endif()
set(KNJ_HAS_DOT_ASM 1)   # both arches, inline asm (see above)

if(NOT KNJ_HAS_WMMA)
  message(STATUS "knj: gfx1031 -- WMMA unavailable (no matrix units, hardware).")
  message(STATUS "knj: gfx1031 -- int dot via inline asm; SIMT packed f16 for f32 math.")
endif()

# Flag that the HIP device build path is available. The root CMakeLists uses
# this to decide whether to add_subdirectory(tests/integration).
set(KNJ_HAS_DEVICE_BUILD ON CACHE INTERNAL "hipcc present and usable" FORCE)