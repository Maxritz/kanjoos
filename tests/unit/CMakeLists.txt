# tests/unit/CMakeLists.txt -- C2 host unit tests.
#
# These are HOST C++17 tests that verify the device abstraction layer
# (DeviceCaps, Backend, arch guard) without requiring a GPU or HIP.
#
# See docs/02-components.md C2, docs/ai-coder/c2-device.md,
# and docs/06-profiling.md section 8 (kanjoos doctor capability table).

# The C2 unit tests verify:
# 1. knj_probe_host_caps() returns correct DeviceCaps from compiled-in macros
# 2. knj_check_arch() enforces the arch guard correctly
# 3. knj_cap_available() returns correct values for each arch
# 4. knj_print_cap_line() produces the exact doc-expected output
# 5. Backend::create() works when KNJ_BUILD_ARCH is set correctly
# 6. Backend::create() refuses when KNJ_BUILD_ARCH is unset or mismatched
# 7. Backend::from_caps() builds a backend from injected caps
# 8. Backend::would_instantiate() gates on caps correctly

include_directories(${CMAKE_SOURCE_DIR}/src/device)
include_directories(${CMAKE_SOURCE_DIR}/tests)

# Simple test executable that runs the C2 unit tests
add_executable(test_c2_device
    test_c2_device.cpp
)

target_link_libraries(test_c2_device PRIVATE knj_device)

target_compile_features(test_c2_device PRIVATE cxx_std_17)

# Set the arch name for the test build to match the target
if(NOT DEFINED KNJ_ARCH_NAME)
    set(KNJ_ARCH_NAME "unknown")
endif()
target_compile_definitions(test_c2_device PRIVATE
    KNJ_ARCH_NAME="${KNJ_ARCH_NAME}"
    KNJ_HAS_WMMA=${KNJ_HAS_WMMA}
    KNJ_HAS_FP8=${KNJ_HAS_FP8}
    KNJ_HAS_DOT_ASM=${KNJ_HAS_DOT_ASM}
)
