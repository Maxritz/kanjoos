# tests/integration/CMakeLists.txt -- HIP device integration test for C2.
#
# Built with an EXPLICIT hipcc command (add_custom_command), NOT cmake's HIP
# language. cmake's HIP module is incompatible with this toolchain:
#   * it forbids hipcc as CMAKE_HIP_COMPILER ("Use Clang directly");
#   * on Windows it forbids mixing Clang-HIP with a non-Clang host compiler
#     (MinGW g++ here) -- Windows-Clang.cmake line 177;
#   * its arch detection (rocm_agent_enumerator / -target-cpu) and ROCm-root
#     path writing both fail on this toolchain (docs/00 section 7.0).
#
# The link line is the MEASURED one from docs/00-verified-facts.md section 7.0:
#   hipcc -nogpulib -O2 --offload-arch=<arch> -L'<rocm>/lib' -lamdhip64 <files>
#
# hipcc's host pass goes through g++ (--driver-mode=g++), which is the same
# compiler that built knj_device, so the host C2 library links fine.
#
# Arch guard at runtime: KNJ_BUILD_ARCH from the ENVIRONMENT (never -D).
#   unset  -> exit 7 (refused)
#   mismatch -> exit 6 (declined, skip)
#   match  -> run

# The host C2 static library, built by src/device/CMakeLists.txt.
if(NOT TARGET knj_device)
  message(FATAL_ERROR
    "knj_device target not found. The root CMakeLists.txt must add_subdirectory(src/device) "
    "before tests/integration.")
endif()

# Locate hipcc (rocm.cmake already found it and set KNJ_HIPCC when
# KNJ_HAS_DEVICE_BUILD is ON).
if(NOT DEFINED KNJ_HIPCC)
  find_program(KNJ_HIPCC NAMES hipcc hipcc.exe
    HINTS "${ROCM_ROOT}/bin" "/g/ROCM10RT-gfx1201/bin")
  if(NOT KNJ_HIPCC)
    message(FATAL_ERROR "knj: hipcc not found for integration test.")
  endif()
endif()

# amdhip64 import lib for the link line.
if(NOT DEFINED KNJ_AMDHIP64)
  find_library(KNJ_AMDHIP64
    NAMES amdhip64 amdhip64_7
    HINTS "${ROCM_ROOT}/lib" "${ROCM_ROOT}/lib/amdhip64"
    NO_DEFAULT_PATH)
endif()

# --- source files -----------------------------------------------------------
# The integration test driver + the C2 device source files. We compile the C2
# sources directly in the hipcc invocation (host pass = g++) rather than linking
# a pre-built knj_device.a, because hipcc on Windows links with lld-link which
# does not resolve g++-mangled symbols from a separately-built g++ static lib.
# Compiling backend.cpp / device_caps.h through hipcc's host pass gives matching
# mangling for both the test driver and the C2 code.
#
# The C2 headers (device_caps.h) and source (backend.cpp, backend.h) are shared
# between the host-only unit test build and this HIP integration test build.
set(INTEG_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c2_device_hip.cpp
    ${CMAKE_SOURCE_DIR}/src/device/backend.cpp
)
set(INTEG_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src/device
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/tests/integration
)

# --- hipcc link line --------------------------------------------------------
# The link line is the MEASURED one from docs/00-verified-facts.md section 7.0:
#   hipcc -nogpulib -O2 --offload-arch=<arch> -L'<rocm>/lib' -lamdhip64 <files>

# Extract --offload-arch=<arch> from KNJ_ARCH_FLAGS.
set(_offload_arch "")
foreach(_f ${KNJ_ARCH_FLAGS})
  if(_f MATCHES "^--offload-arch=(.+)$")
    set(_offload_arch "${_f}")
  endif()
endforeach()
if(NOT _offload_arch)
  set(_offload_arch "--offload-arch=${KNJ_ARCH}")
endif()

# hipcc command. hipcc re-spawns clang through a command string, so quotes
# around values are eaten by the inner shell (docs/00-verified-facts.md section
# 8.7). We therefore CANNOT pass -DKNJ_ARCH_NAME="gfx1201" through hipcc.
# Instead we write the arch name and cap flags into a small config header that
# device_caps.h includes, bypassing the -D/quote problem entirely.
#
# Generate knj_arch_config.h in the build tree with the correct values.
set(INTEG_ARCH_CONFIG "${CMAKE_BINARY_DIR}/tests/integration/knj_arch_config.h")
# -include knj_arch_config.h: hipcc's host pass (g++) compiles backend.cpp
# too, and it must see the same KNJ_ARCH_NAME / KNJ_HAS_* defines as the test
# driver. backend.cpp does not #include knj_arch_config.h itself, so we use
# -include to forcibly inject it before any other header. This bypasses the
# -D quote-stripping problem (docs/00 section 8.7).
set(INTEG_HIPCC_CMD
    "${KNJ_HIPCC}"
    -O2
    -nogpulib
    "-include" "knj_arch_config.h"
    ${_offload_arch}
    "-L${ROCM_ROOT}/lib"
    "-lamdhip64"
)
# Add -I flags for each include dir.
foreach(_d ${INTEG_INCLUDE_DIRS})
  list(APPEND INTEG_HIPCC_CMD "-I${_d}")
endforeach()
list(APPEND INTEG_HIPCC_CMD
    "${INTEG_SRC}"
    "-o"
    "${CMAKE_BINARY_DIR}/tests/integration/test_c2_device_hip"
)

# --- arch config header (bypasses hipcc -D quote-stripping) ----------------
# hipcc re-spawns clang through a command string and eats quotes around -D values
# (docs/00-verified-facts.md section 8.7). We therefore generate a small config
# header with the arch name and cap flags, and include it from device_caps.h via
# a -I path. This header is in the build tree; device_caps.h reads it as a normal
# #include.
#
# The config header is generated BEFORE the hipcc compile so it is available.
set(INTEG_ARCH_CONFIG "${CMAKE_BINARY_DIR}/tests/integration/knj_arch_config.h")
add_custom_command(
    OUTPUT "${INTEG_ARCH_CONFIG}"
    COMMAND ${CMAKE_COMMAND} -E echo "// Auto-generated by CMake. Do not edit." > "${INTEG_ARCH_CONFIG}"
    COMMAND ${CMAKE_COMMAND} -E echo "#ifndef KNJ_ARCH_CONFIG_H" >> "${INTEG_ARCH_CONFIG}"
    COMMAND ${CMAKE_COMMAND} -E echo "#define KNJ_ARCH_CONFIG_H" >> "${INTEG_ARCH_CONFIG}"
    COMMAND ${CMAKE_COMMAND} -E echo "#define KNJ_ARCH_NAME \"${KNJ_ARCH_NAME}\"" >> "${INTEG_ARCH_CONFIG}"
    COMMAND ${CMAKE_COMMAND} -E echo "#define KNJ_HAS_WMMA ${KNJ_HAS_WMMA}" >> "${INTEG_ARCH_CONFIG}"
    COMMAND ${CMAKE_COMMAND} -E echo "#define KNJ_HAS_FP8 ${KNJ_HAS_FP8}" >> "${INTEG_ARCH_CONFIG}"
    COMMAND ${CMAKE_COMMAND} -E echo "#define KNJ_HAS_DOT_ASM ${KNJ_HAS_DOT_ASM}" >> "${INTEG_ARCH_CONFIG}"
    COMMAND ${CMAKE_COMMAND} -E echo "#endif // KNJ_ARCH_CONFIG_H" >> "${INTEG_ARCH_CONFIG}"
    COMMENT "Generating knj_arch_config.h for HIP integration test"
    VERBATIM
)

# --- build rule -------------------------------------------------------------
set(INTEG_BIN "${CMAKE_BINARY_DIR}/tests/integration/test_c2_device_hip")
set(INTEG_STAMP "${CMAKE_BINARY_DIR}/tests/integration/test_c2_device_hip.stamp")

set(INTEG_DEP
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c2_device_hip.cpp
    ${CMAKE_SOURCE_DIR}/src/device/backend.cpp
    ${CMAKE_SOURCE_DIR}/src/device/device_caps.h
    ${CMAKE_SOURCE_DIR}/src/device/backend.h
    "${INTEG_ARCH_CONFIG}"
)

# Use astamp file as the custom command OUTPUT so cmake does not create a
# conflicting phony rule. The stamp depends on the real binary.
add_custom_command(
    OUTPUT "${INTEG_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/tests/integration"
    COMMAND ${CMAKE_COMMAND} -E echo "=== Building C2 HIP device integration test ==="
    COMMAND ${CMAKE_COMMAND} -E echo "  hipcc ${_offload_arch} -nogpulib -I${CMAKE_SOURCE_DIR}/src/device -L${ROCM_ROOT}/lib -lamdhip64 test_c2_device_hip.cpp backend.cpp -o test_c2_device_hip"
    COMMAND ${INTEG_HIPCC_CMD}
    COMMAND ${CMAKE_COMMAND} -E touch "${INTEG_STAMP}"
    DEPENDS ${INTEG_DEP} knj_device
    COMMENT "Building C2 HIP device integration test with hipcc"
    VERBATIM
)

# Make sure the output directory exists.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tests/integration")

# Custom target to build the integration test.
add_custom_target(build_test_c2_device_hip
    DEPENDS "${INTEG_STAMP}"
)

# Note: build_test_c2_device_hip is NOT in the default 'all' target.
# Build it explicitly with `ninja build_test_c2_device_hip`, or add it as a
# dependency of your own top-level target. The host-only C2 unit tests are the
# default build; the HIP integration test is opt-in.

# Note: test_c2_device_hip is a custom target, not an installable target.
# The bench/integration runners find it by path in the build tree.
