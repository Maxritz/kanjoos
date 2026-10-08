# cmake/sanitize.cmake -- ASAN/UBSAN build for the eviction stress tests.
#
# Why this exists: docs/02-components.md C8 and docs/08-roadmap.md acceptance
# criteria both require "no use-after-free under an eviction stress test" and
# "cancellation releases in-flight pages without use-after-free (ASAN/UBSAN
# run)". A slot or KV page whose lifetime is bounded by wall-clock reasoning
# rather than by a HIP event is exactly the bug that only shows up as a rare
# silent read of freed VRAM, and ASAN is the only thing that turns it into a
# loud failure in CI.
#
# Off by default. It roughly doubles compile time and the device passes are not
# instrumented, so a clean run here is evidence about HOST memory discipline --
# which is where the slot tables, the warm pool and the pinned pool live, and
# which is where the bug actually is.

option(KNJ_SANITIZE "Build host code with ASAN+UBSAN" OFF)

function(knj_apply_sanitize target)
  if(NOT KNJ_SANITIZE)
    return()
  endif()

  if(MSVC)
    # /fsanitize=address. MSVC has no UBSAN equivalent, and saying so here is
    # better than silently shipping an "ASAN+UBSAN" build that has no UBSAN.
    target_compile_options(${target} PRIVATE /fsanitize=address /Zi)
    message(STATUS "knj: ${target} sanitized with MSVC /fsanitize=address (ASAN only; no MSVC UBSAN)")
  else()
    target_compile_options(${target} PRIVATE
      -fsanitize=address -fsanitize=undefined
      -fno-omit-frame-pointer -fno-optimize-sibling-calls -g)
    target_link_options(${target} PRIVATE -fsanitize=address -fsanitize=undefined)
    message(STATUS "knj: ${target} sanitized with ASAN+UBSAN")
  endif()

  # Poison allocations inside the pinned pool and the warm pool so a use-after-
  # free inside either is reported immediately rather than at the next read.
  target_compile_definitions(${target} PRIVATE KNJ_SANITIZE_BUILD=1)
endfunction()

# The stress test is the only place that is REQUIRED to run sanitized. If
# someone asks for it without the flag, say so rather than quietly running the
# uninstrumented build and calling it a pass.
function(knj_require_sanitize_for_stress)
  if(NOT KNJ_SANITIZE)
    message(WARNING
      "knj: the eviction stress test is requested without KNJ_SANITIZE=ON.\n"
      "      A clean uninstrumented run is NOT evidence of absence of "
      "use-after-free.\n"
      "      Re-run with -DKNJ_SANITIZE=ON to make that claim.")
  endif()
endfunction()