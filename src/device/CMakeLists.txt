# src/device/CMakeLists.txt -- C2 Device abstraction layer.
#
# Owns: the arch split. Everything above C2 is arch-agnostic.
#
# This is HOST C++17. It compiles on the host compiler with NO HIP headers.
# The device integration test includes it only to call the table emitters,
# which run on the host before any kernel launches.
#
# See docs/02-components.md C2, docs/ai-coder/c2-device.md,
# and docs/07-build-platforms.md section 2 (host-only ISO C++17 rule).

add_library(knj_device STATIC
    backend.cpp
)

target_include_directories(knj_device PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(knj_device PUBLIC cxx_std_17)

# The C2 headers are host-only ISO C++17. They must not see any HIP
# headers or device builtins. The device integration test links this
# library and calls the table emitters from the host side.

# KNJ_ARCH_NAME is set via compiler flags in cmake/arch.cmake, but we need
# to ensure it's also available as a define for the device_caps.h header.
if(NOT DEFINED KNJ_ARCH_NAME)
    set(KNJ_ARCH_NAME "unknown")
endif()
target_compile_definitions(knj_device PRIVATE
    KNJ_ARCH_NAME="${KNJ_ARCH_NAME}"
)

# Propagate the compile-time cap flags from cmake/arch.cmake /
# cmake/rocm.cmake so device_caps.h can read them as KNJ_HAS_* macros.
target_compile_definitions(knj_device PRIVATE
    KNJ_HAS_WMMA=${KNJ_HAS_WMMA}
    KNJ_HAS_FP8=${KNJ_HAS_FP8}
    KNJ_HAS_DOT_ASM=${KNJ_HAS_DOT_ASM}
)
