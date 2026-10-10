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

# ---------------------------------------------------------------------------
# C21 profiling unit tests.
#
# The profiler is host-only C++ by construction (src/profiler/profiler.cpp
# includes no HIP header), which is what makes the device clock domain testable
# without a GPU: the device layer feeds it offsets from one epoch event per
# stream, and offsets are numbers. These tests grade the arithmetic through the
# artifact writers, because those files are what a user reads and diffs.
#
# The sources are named explicitly rather than linked against knj_runtime:
# tests/unit is added to the build BEFORE the runtime library is declared in the
# root CMakeLists, and a target that links something declared later fails at
# generate time. Naming the two translation units keeps this test independent of
# that ordering, and keeps it building on a machine with no device toolchain.
#
# Run beside the C9 gate, which needs no GPU either:
#   ctest --test-dir build/cmake-host -R 'c21|c9' --output-on-failure
add_executable(test_c21_profiler
    test_c21_profiler.cpp
    ${CMAKE_SOURCE_DIR}/src/profiler/profiler.cpp
    ${CMAKE_SOURCE_DIR}/src/platform/memprobe.cpp
)

target_include_directories(test_c21_profiler PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_features(test_c21_profiler PRIVATE cxx_std_17)

# ---------------------------------------------------------------------------
# The tokenizer.ggml.pre dispatch gate.
#
# The engine dispatches its pre-tokeniser on the value the file declares. Both
# rules are the same code with one predicate changed (a combining mark joins the
# letter run, or is punctuation), so "did the dispatch happen" is not visible in
# any output except the token ids -- which is why the fixtures exist: ten tiny
# GGUFs with an identical vocabulary and merge table that differ only in `pre`,
# written by tools/make_tok_fixtures.py. Two of them must produce different ids
# for the same mark-bearing text, the rest must resolve to the rule their name
# selects, and an absent or unknown `pre` must be refused by name.
#
# Sources are named explicitly for the same reason as test_c21_profiler above:
# tests/unit is added before the runtime library is declared in the root
# CMakeLists; the loader/tokenizer pair is small enough to compile standalone.
#
# Regenerate the fixtures with:
#   python tools/make_tok_fixtures.py
# and verify they are what the generator would write with:
#   python tools/make_tok_fixtures.py --check
# src/tokenizer/tokenizer.cpp carries KNJ_PROFILE_OP markers (the tokenizer
# build is a measurable stage of the tokenize-only path), so the profiler's two
# host-only translation units travel with it into these standalone targets.
add_executable(test_tok_pre_dispatch
    test_tok_pre_dispatch.cpp
    ${CMAKE_SOURCE_DIR}/src/tokenizer/tokenizer.cpp
    ${CMAKE_SOURCE_DIR}/src/tokenizer/unicode_ranges.cpp
    ${CMAKE_SOURCE_DIR}/src/loader/gguf.cpp
    ${CMAKE_SOURCE_DIR}/src/profiler/profiler.cpp
    ${CMAKE_SOURCE_DIR}/src/platform/memprobe.cpp
)

target_include_directories(test_tok_pre_dispatch PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_features(test_tok_pre_dispatch PRIVATE cxx_std_17)
target_compile_definitions(test_tok_pre_dispatch PRIVATE
    KNJ_TOK_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures/tok"
)

# ---------------------------------------------------------------------------
# The pre-tokeniser whitespace-run gate.
#
# The reference's `\s+(?!\S)` gives back one CHARACTER out of a whitespace run;
# the engine gave back one BYTE, which for U+00A0 (2 bytes) and U+3000 (3 bytes)
# cut the character in half and forbade its own merge. Invisible on ASCII, and
# invisible to the 25-string cross-check -- found by tools/tok_crosscheck.py's
# fuzz pass and reduced to those two inputs. The fixtures' vocabulary carries one
# merged token per whitespace character (ids 258-261) so the split is an exactly
# assertable id list rather than an inference.
#
# Regenerate the fixtures with `python tools/make_tok_fixtures.py` after changing
# the vocabulary; `--check` verifies them.
add_executable(test_tok_whitespace_run
    test_tok_whitespace_run.cpp
    ${CMAKE_SOURCE_DIR}/src/tokenizer/tokenizer.cpp
    ${CMAKE_SOURCE_DIR}/src/tokenizer/unicode_ranges.cpp
    ${CMAKE_SOURCE_DIR}/src/loader/gguf.cpp
    ${CMAKE_SOURCE_DIR}/src/profiler/profiler.cpp
    ${CMAKE_SOURCE_DIR}/src/platform/memprobe.cpp
)

target_include_directories(test_tok_whitespace_run PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_features(test_tok_whitespace_run PRIVATE cxx_std_17)
target_compile_definitions(test_tok_whitespace_run PRIVATE
    KNJ_TOK_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures/tok"
)

# ---------------------------------------------------------------------------
# The pre-tokeniser alternative sweep.
#
# Only the whitespace-run alternative was ever swept against llama.cpp's
# behaviour; the other six were emulated from prose. The expectations in
# test_tok_pre_sweep.cpp are splits produced by the reference's ACTUAL QWEN2 /
# QWEN35 patterns (tools/tok_pre_sweep.py extracts them mechanically from
# llama.cpp's src/llama-vocab.cpp; records/tok-pre-sweep-2026-10-10.log carries
# the reviewed output). pretokenize() is static, so no fixture is needed.
add_executable(test_tok_pre_sweep
    test_tok_pre_sweep.cpp
    ${CMAKE_SOURCE_DIR}/src/tokenizer/tokenizer.cpp
    ${CMAKE_SOURCE_DIR}/src/tokenizer/unicode_ranges.cpp
    ${CMAKE_SOURCE_DIR}/src/loader/gguf.cpp
    ${CMAKE_SOURCE_DIR}/src/profiler/profiler.cpp
    ${CMAKE_SOURCE_DIR}/src/platform/memprobe.cpp
)

target_include_directories(test_tok_pre_sweep PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_features(test_tok_pre_sweep PRIVATE cxx_std_17)

enable_testing()
add_test(NAME c21_profiler COMMAND test_c21_profiler)
add_test(NAME tok_pre_dispatch COMMAND test_tok_pre_dispatch)
add_test(NAME tok_whitespace_run COMMAND test_tok_whitespace_run)
add_test(NAME tok_pre_sweep COMMAND test_tok_pre_sweep)

# ---------------------------------------------------------------------------
# The python tool gates.
#
# These used to be hand-run scripts -- one of them lived in tmp/ -- and they are
# the same kind of gate as the three C++ cases above: a measurement that fails on
# drift, covering what a C++ unit test cannot. `tok_separator_sweep` is the sweep
# that caught the byte-vs-character whitespace backtrack; `tools_smoke` is the
# functional smoke of the tokenizer tools; `tok_pre_rules` re-derives the engine's
# pre-tokeniser table from llama.cpp's ACTUAL name map and regex table and fails
# if the two disagree.
#
# Each needs machine-specific inputs: the engine binary, `llama-tokenize.exe` from
# the local llama.cpp build, a model file, and (for tok_pre_rules) the llama.cpp
# source tree. Each resolves them from the environment first (KNJ_ENGINE,
# KNJ_LLAMA_TOKENIZE, KNJ_TOK_MODEL, KNJ_LLAMA_CPP), then from the default path in
# this repo's AGENTS.md, and when one is missing it prints NOT RUN and exits 3.
# SKIP_RETURN_CODE 3 turns that into ctest's ***Skipped*** with the reason printed
# above it, so a run that could not measure anything is never reported as a pass
# and never omitted either.
find_package(Python3 COMPONENTS Interpreter QUIET)
# WHICH interpreter, explicitly. find_package picks the first install it finds,
# which on a machine with several of them is not necessarily the one carrying the
# packages the tools import (the fixture generator needs `gguf`). The tools report
# a missing package as NOT RUN rather than as a broken fixture, and this cache
# variable is how a build points them at the right interpreter:
#   cmake -S . -B build/cmake-host -DKNJ_PYTHON=<python.exe>
set(KNJ_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH
    "interpreter that runs the python tool gates")
if(KNJ_PYTHON)
  add_test(NAME tools_smoke
           COMMAND ${KNJ_PYTHON} ${CMAKE_SOURCE_DIR}/tools/smoke_tools.py)
  add_test(NAME tok_separator_sweep
           COMMAND ${KNJ_PYTHON} ${CMAKE_SOURCE_DIR}/tools/tok_crosscheck.py
                   --separator-sweep)
  add_test(NAME tok_pre_rules
           COMMAND ${KNJ_PYTHON} ${CMAKE_SOURCE_DIR}/tools/tok_pre_rules.py)
  add_test(NAME tok_fuzz_campaign
           COMMAND ${KNJ_PYTHON} ${CMAKE_SOURCE_DIR}/tools/tok_fuzz_campaign.py)
  # The fixture generator proves `tests/fixtures/tok/` is what it would write,
  # so a vocabulary change committed with stale fixtures fails here instead of
  # silently testing old bytes. Needs only the `gguf` package (no model, no
  # llama.cpp); without it the tool exits 3 and reads as SKIPPED, like the rest.
  add_test(NAME tok_fixtures
           COMMAND ${KNJ_PYTHON} ${CMAKE_SOURCE_DIR}/tools/make_tok_fixtures.py
                   --check)
  set_tests_properties(tools_smoke tok_separator_sweep tok_pre_rules tok_fuzz_campaign
                       tok_fixtures
                       PROPERTIES SKIP_RETURN_CODE 3)

else()
  message(STATUS "no python interpreter found: the python tool gates (tools_smoke, "
                 "tok_separator_sweep, tok_pre_rules, tok_fuzz_campaign, tok_fixtures) "
                 "are NOT registered; set -DKNJ_PYTHON=<python.exe> to register them")
endif()

# ---------------------------------------------------------------------------
# The MinGW runtime, for the C++ cases.
#
# The test executables are MinGW builds: they load libstdc++-6.dll,
# libgcc_s_seh-1.dll and libwinpthread-1.dll at process start. If those are not
# findable, every case dies with 0xC0000139 -- which reads as a failed assertion
# rather than as a missing DLL, and did exactly that in this repo's ctest runs
# before this block existed. The runtime ships next to the compiler that produced
# these binaries, so it is found from ${CMAKE_CXX_COMPILER} instead of being
# written down as a machine path. ENVIRONMENT_MODIFICATION (cmake 3.22+) prepends
# to the child's PATH without this file rebuilding the PATH string, which a
# Windows PATH's `;` separators would break in the plain ENVIRONMENT property.
get_filename_component(KNJ_CXX_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
if(WIN32 AND EXISTS "${KNJ_CXX_BIN}/libstdc++-6.dll")
  # Every test, not just the C++ cases: the python gates also spawn MinGW-built
  # helpers (engine, probes), which die with 0xC0000139 without this PATH entry.
  set_tests_properties(c21_profiler tok_pre_dispatch tok_whitespace_run
                       tools_smoke tok_separator_sweep tok_pre_rules tok_fuzz_campaign
                       PROPERTIES ENVIRONMENT_MODIFICATION
                       "PATH=path_list_prepend:${KNJ_CXX_BIN}")
  message(STATUS "test PATH: prepending ${KNJ_CXX_BIN} (the MinGW runtime the "
                 "test binaries link against)")
endif()
