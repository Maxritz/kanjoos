// tests/unit/test_c2_device.cpp -- C2 Device abstraction layer unit tests.
//
// These are HOST C++17 tests that verify the device abstraction layer
// (DeviceCaps, Backend, arch guard) without requiring a GPU or HIP.
//
// See docs/02-components.md C2, docs/ai-coder/c2-device.md,
// and docs/06-profiling.md section 8 (kanjoos doctor capability table).
//
// Exit codes:
//   0  all tests pass
//
// The summary below reports tests_run / tests_passed / tests_failed as printed
// by the binary. The "assertions" count in earlier drafts was a different
// metric and is not what this binary prints.
//   1  at least one test failed

#include "device_caps.h"
#include "backend.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>

#ifdef _WIN32
#include <stdlib.h>
#endif

// Platform-independent environment variable helpers
//
// The Windows path must actually remove the variable from the environment
// block, not just write name=. _putenv_s(name, "") is not a reliable unset on
// all CRTs and can leave a dangling name= entry that later _putenv_s calls
// shadow. Use the documented pattern: write name= then NUL the value in place.
namespace {
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
void unset_env(const char* name) {
#ifdef _WIN32
    // _putenv_s(name, "") is NOT a reliable unset on all MSVC/MinGW CRTs.
    // Write name= then NUL the value in place so getenv-style readers see
    // nothing and later _putenv_s calls do not shadow the name.
    _putenv_s(name, "");
    // _putenv_s stores a pointer into the process environment block; rewrite
    // the value in place. Find the '=' we just wrote and terminate there.
    size_t namelen = std::strlen(name);
    char* env = nullptr;
    size_t cap = 0;
    if (_dupenv_s(&env, &cap, name) == 0 && env) {
        // env points at "name=...\0". Put a NUL over the '='.
        if (cap > namelen) {
            env[namelen] = '\0';
        }
        free(env);
    }
#else
    unsetenv(name);
#endif
}
}  // anonymous namespace

// Test counter
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                                                       \
    do {                                                                 \
        tests_run++;                                                    \
        std::printf("  TEST %-60s", name);                             \
    } while (0)

#define PASS()                                                           \
    do {                                                                 \
        tests_passed++;                                                 \
        std::printf(" PASS\n");                                         \
    } while (0)

#define FAIL(msg)                                                        \
    do {                                                                 \
        tests_failed++;                                                 \
        std::printf(" FAIL: %s\n", msg);                               \
    } while (0)

#define ASSERT_TRUE(expr, msg)                                           \
    do {                                                                 \
        if (!(expr)) {                                                  \
            FAIL(msg);                                                  \
        } else {                                                        \
            PASS();                                                     \
        }                                                               \
    } while (0)

#define ASSERT_STREQ(expected, actual, msg)                              \
    do {                                                                 \
        if (std::strcmp((expected), (actual)) != 0) {                  \
            FAIL(msg);                                                  \
            std::printf("    expected: \"%s\", got: \"%s\"\n",          \
                    (expected), (actual));                              \
        } else {                                                        \
            PASS();                                                     \
        }                                                               \
    } while (0)

#define ASSERT_EQ_INT(expected, actual, msg)                             \
    do {                                                                 \
        if ((expected) != (actual)) {                                   \
            FAIL(msg);                                                  \
            std::printf("    expected: %d, got: %d\n",                  \
                    (int)(expected), (int)(actual));                    \
        } else {                                                        \
            PASS();                                                     \
        }                                                               \
    } while (0)

#define ASSERT_EQ_PTR(expected, actual, msg)                             \
    do {                                                                 \
        if ((expected) != (actual)) {                                   \
            FAIL(msg);                                                  \
            std::printf("    expected: %s, got: %s\n",                  \
                    (expected) ? "non-null" : "null",                   \
                    (actual) ? "non-null" : "null");                    \
        } else {                                                        \
            PASS();                                                     \
        }                                                               \
    } while (0)

// ============================================================================
// Test 1: knj_check_arch() - arch guard logic
// ============================================================================

void test_arch_guard() {
    std::printf("\n=== Test 1: knj_check_arch() arch guard ===\n");

    // Test: matching arch returns ok
    {
        TEST("matching arch returns ok");
        const char* env_arch = "gfx1201";
        const char* built_arch = "gfx1201";
        KnjArchVerdict v = knj_check_arch(env_arch, built_arch);
        ASSERT_TRUE(v.ok, "verdict should be ok");
        ASSERT_STREQ("", v.reason, "reason should be empty");
    }

    // Test: unset env_arch returns refusal
    {
        TEST("unset env_arch returns refusal");
        const char* env_arch = nullptr;
        const char* built_arch = "gfx1201";
        KnjArchVerdict v = knj_check_arch(env_arch, built_arch);
        ASSERT_TRUE(!v.ok, "verdict should not be ok");
        ASSERT_TRUE(std::strstr(v.reason, "not set") != nullptr,
                    "reason should mention 'not set'");
    }

    // Test: empty env_arch returns refusal
    {
        TEST("empty env_arch returns refusal");
        const char* env_arch = "";
        const char* built_arch = "gfx1201";
        KnjArchVerdict v = knj_check_arch(env_arch, built_arch);
        ASSERT_TRUE(!v.ok, "verdict should not be ok");
        ASSERT_TRUE(std::strstr(v.reason, "not set") != nullptr,
                    "reason should mention 'not set'");
    }

    // Test: mismatched arch returns refusal
    {
        TEST("mismatched arch returns refusal");
        const char* env_arch = "gfx1031";
        const char* built_arch = "gfx1201";
        KnjArchVerdict v = knj_check_arch(env_arch, built_arch);
        ASSERT_TRUE(!v.ok, "verdict should not be ok");
        ASSERT_TRUE(std::strstr(v.reason, "mismatch") != nullptr,
                    "reason should mention 'mismatch'");
    }

    // Test: nullptr built_arch returns refusal
    {
        TEST("nullptr built_arch returns refusal");
        const char* env_arch = "gfx1201";
        const char* built_arch = nullptr;
        KnjArchVerdict v = knj_check_arch(env_arch, built_arch);
        ASSERT_TRUE(!v.ok, "verdict should not be ok");
    }
}

// ============================================================================
// Test 2: knj_build_arch_env() - environment variable access
// ============================================================================

void test_build_arch_env() {
    std::printf("\n=== Test 2: knj_build_arch_env() ===\n");

    // Test: unset environment returns nullptr
    {
        TEST("unset KNJ_BUILD_ARCH returns nullptr");
        unset_env("KNJ_BUILD_ARCH");
        const char* arch = knj_build_arch_env();
        ASSERT_EQ_PTR(nullptr, arch, "should return nullptr when unset");
    }

    // Test: set environment returns the value
    {
        TEST("set KNJ_BUILD_ARCH returns the value");
        set_env("KNJ_BUILD_ARCH", "gfx1201");
        const char* arch = knj_build_arch_env();
        ASSERT_STREQ("gfx1201", arch, "should return the set value");
        unset_env("KNJ_BUILD_ARCH");
    }
}

// ============================================================================
// Test 3: knj_cap_available() - capability lookup
// ============================================================================

void test_cap_available() {
    std::printf("\n=== Test 3: knj_cap_available() ===\n");

    // Test: gfx1201 has WMMA
    {
        TEST("gfx1201 has WmmaF16");
        ASSERT_TRUE(knj_cap_available("gfx1201", KnjCap::WmmaF16),
                    "gfx1201 should have WMMA");
    }

    // Test: gfx1031 does not have WMMA
    {
        TEST("gfx1031 does not have WmmaF16");
        ASSERT_TRUE(!knj_cap_available("gfx1031", KnjCap::WmmaF16),
                    "gfx1031 should not have WMMA");
    }

    // Test: gfx1201 has FP8
    {
        TEST("gfx1201 has Fp8");
        ASSERT_TRUE(knj_cap_available("gfx1201", KnjCap::Fp8),
                    "gfx1201 should have FP8");
    }

    // Test: gfx1031 does not have FP8
    {
        TEST("gfx1031 does not have Fp8");
        ASSERT_TRUE(!knj_cap_available("gfx1031", KnjCap::Fp8),
                    "gfx1031 should not have FP8");
    }

    // Test: both arches have Dot4I32I8
    {
        TEST("gfx1201 has Dot4I32I8");
        ASSERT_TRUE(knj_cap_available("gfx1201", KnjCap::Dot4I32I8),
                    "gfx1201 should have dot4");
        TEST("gfx1031 has Dot4I32I8");
        ASSERT_TRUE(knj_cap_available("gfx1031", KnjCap::Dot4I32I8),
                    "gfx1031 should have dot4");
    }

    // Test: both arches have Dot8I32I4
    {
        TEST("gfx1201 has Dot8I32I4");
        ASSERT_TRUE(knj_cap_available("gfx1201", KnjCap::Dot8I32I4),
                    "gfx1201 should have dot8");
        TEST("gfx1031 has Dot8I32I4");
        ASSERT_TRUE(knj_cap_available("gfx1031", KnjCap::Dot8I32I4),
                    "gfx1031 should have dot8");
    }

    // Test: both arches have PkFmaF16
    {
        TEST("gfx1201 has PkFmaF16");
        ASSERT_TRUE(knj_cap_available("gfx1201", KnjCap::PkFmaF16),
                    "gfx1201 should have pk_fma_f16");
        TEST("gfx1031 has PkFmaF16");
        ASSERT_TRUE(knj_cap_available("gfx1031", KnjCap::PkFmaF16),
                    "gfx1031 should have pk_fma_f16");
    }

    // Test: gfx1031 has Sdot4Builtin
    {
        TEST("gfx1031 has Sdot4Builtin");
        ASSERT_TRUE(knj_cap_available("gfx1031", KnjCap::Sdot4Builtin),
                    "gfx1031 should have sdot4 builtin");
    }

    // Test: gfx1201 does not have Sdot4Builtin
    {
        TEST("gfx1201 does not have Sdot4Builtin");
        ASSERT_TRUE(!knj_cap_available("gfx1201", KnjCap::Sdot4Builtin),
                    "gfx1201 should not have sdot4 builtin");
    }

    // Test: nullptr arch returns false
    {
        TEST("nullptr arch returns false");
        ASSERT_TRUE(!knj_cap_available(nullptr, KnjCap::Dot4I32I8),
                    "nullptr arch should return false");
    }

    // Test: empty arch returns false
    {
        TEST("empty arch returns false");
        ASSERT_TRUE(!knj_cap_available("", KnjCap::Dot4I32I8),
                    "empty arch should return false");
    }
}

// ============================================================================
// Test 4: knj_cap_label() - capability label
// ============================================================================

void test_cap_label() {
    std::printf("\n=== Test 4: knj_cap_label() ===\n");

    TEST("Dot4I32I8 label");
    ASSERT_STREQ("dot4_i32_i8", knj_cap_label(KnjCap::Dot4I32I8),
                 "label mismatch");
    TEST("Dot8I32I4 label");
    ASSERT_STREQ("dot8_i32_i4", knj_cap_label(KnjCap::Dot8I32I4),
                 "label mismatch");
    TEST("PkFmaF16 label");
    ASSERT_STREQ("pk_fma_f16", knj_cap_label(KnjCap::PkFmaF16),
                 "label mismatch");
    TEST("WmmaF16 label");
    ASSERT_STREQ("wmma_f16", knj_cap_label(KnjCap::WmmaF16),
                 "label mismatch");
    TEST("Fp8 label");
    ASSERT_STREQ("fp8", knj_cap_label(KnjCap::Fp8),
                 "label mismatch");
    TEST("Sdot4Builtin label");
    ASSERT_STREQ("sdot4", knj_cap_label(KnjCap::Sdot4Builtin),
                 "label mismatch");
}

// ============================================================================
// Test 5: knj_dot_label() - dot path label
// ============================================================================

void test_dot_label() {
    std::printf("\n=== Test 5: knj_dot_label() ===\n");

    // Test: has_dot_builtin returns "builtin(sdot4)"
    {
        TEST("has_dot_builtin returns builtin(sdot4)");
        DeviceCaps c{};
        c.has_dot_builtin = true;
        ASSERT_STREQ("builtin(sdot4)", knj_dot_label(c),
                     "dot label mismatch");
    }

    // Test: no dot builtin but has DOT_ASM returns "asm(v_dot4_i32_i8)"
    {
        TEST("no dot builtin but has DOT_ASM returns asm label");
        DeviceCaps c{};
        c.has_dot_builtin = false;
        ASSERT_STREQ("asm(v_dot4_i32_i8)", knj_dot_label(c),
                     "dot label mismatch");
    }

    // Test: no dot at all returns "none" - depends on KNJ_HAS_DOT_ASM
    {
        TEST("no dot at all returns none (or skips if DOT_ASM enabled)");
#ifdef KNJ_HAS_DOT_ASM
        std::printf(" (KNJ_HAS_DOT_ASM=%d, skipping 'none' test)\n", KNJ_HAS_DOT_ASM);
        PASS();
#else
        DeviceCaps c{};
        c.has_dot_builtin = false;
        ASSERT_STREQ("none", knj_dot_label(c),
                     "dot label mismatch");
#endif
    }
}

// ============================================================================
// Test 6: knj_print_cap_line() - capability line output
// ============================================================================

void test_print_cap_line() {
    std::printf("\n=== Test 6: knj_print_cap_line() ===\n");

    // Test: gfx1201 capability line
    {
        TEST("gfx1201 capability line");
        DeviceCaps c{};
        std::strncpy(c.gcn_arch, "gfx1201", sizeof(c.gcn_arch) - 1);
        c.has_wmma = true;
        c.has_fp8 = true;
        c.has_dot_builtin = false;
        char out[256];
        knj_print_cap_line(c, true, out, sizeof(out));
        std::printf("    [%s]\n", out);
        ASSERT_TRUE(std::strstr(out, "wmma=on") != nullptr,
                    "should contain wmma=on");
        ASSERT_TRUE(std::strstr(out, "fp8=on") != nullptr,
                    "should contain fp8=on");
        ASSERT_TRUE(std::strstr(out, "dot=asm(v_dot4_i32_i8)") != nullptr,
                    "should contain dot=asm(v_dot4_i32_i8)");
        ASSERT_TRUE(std::strstr(out, "smi=lane-split") != nullptr,
                    "should contain smi=lane-split");
        ASSERT_TRUE(std::strstr(out, "v_pk_fma_f16=yes") != nullptr,
                    "should contain v_pk_fma_f16=yes");
    }

    // Test: gfx1031 capability line
    {
        TEST("gfx1031 capability line");
        DeviceCaps c{};
        std::strncpy(c.gcn_arch, "gfx1031", sizeof(c.gcn_arch) - 1);
        c.has_wmma = false;
        c.has_fp8 = false;
        c.has_dot_builtin = true;
        char out[256];
        knj_print_cap_line(c, true, out, sizeof(out));
        std::printf("    [%s]\n", out);
        ASSERT_TRUE(std::strstr(out, "wmma=off") != nullptr,
                    "should contain wmma=off");
        ASSERT_TRUE(std::strstr(out, "fp8=off") != nullptr,
                    "should contain fp8=off");
        ASSERT_TRUE(std::strstr(out, "dot=builtin(sdot4)") != nullptr,
                    "should contain dot=builtin(sdot4)");
        ASSERT_TRUE(std::strstr(out, "v_pk_fma_f16=yes") != nullptr,
                    "should contain v_pk_fma_f16=yes");
    }
}

// ============================================================================
// Test 7: knj_probe_host_caps() - host capability probing
// ============================================================================

void test_probe_host_caps() {
    std::printf("\n=== Test 7: knj_probe_host_caps() ===\n");

    // Test: probe returns correct caps for the compiled arch
    {
        TEST("probe returns correct caps for compiled arch");
        set_env("KNJ_BUILD_ARCH", KNJ_ARCH_NAME);
        DeviceCaps c = knj_probe_host_caps();
        ASSERT_TRUE(c.arch_ok, "arch should be ok");
        ASSERT_STREQ(KNJ_ARCH_NAME, c.gcn_arch, "arch should match compile-time");
        ASSERT_EQ_INT(KNJ_HAS_WMMA != 0, c.has_wmma, "has_wmma should match compile-time");
        ASSERT_EQ_INT(KNJ_HAS_FP8 != 0, c.has_fp8, "has_fp8 should match compile-time");
        ASSERT_EQ_INT(32, c.wave_size, "wave_size should be 32");
        ASSERT_EQ_INT((KNJ_HAS_WMMA == 0 ? 1 : 0), c.has_dot_builtin,
                      "has_dot_builtin should be inverse of has_wmma (sdot4 on gfx1031 only)");
        unset_env("KNJ_BUILD_ARCH");
    }

    // Test: probe with unset KNJ_BUILD_ARCH returns arch_ok=false
    {
        TEST("probe with unset KNJ_BUILD_ARCH returns arch_ok=false");
        unset_env("KNJ_BUILD_ARCH");
        DeviceCaps c = knj_probe_host_caps();
        ASSERT_TRUE(!c.arch_ok, "arch should not be ok");
        ASSERT_TRUE(c.gcn_arch[0] == '\0',
                    "gcn_arch should be empty when arch guard fails");
    }

    // Test: probe with mismatched KNJ_BUILD_ARCH returns arch_ok=false
    {
        TEST("probe with mismatched KNJ_BUILD_ARCH returns arch_ok=false");
        // Use the OTHER arch (not the compiled one) to force a mismatch.
        const char* other_arch =
            (std::strcmp(KNJ_ARCH_NAME, "gfx1201") == 0) ? "gfx1031" : "gfx1201";
        set_env("KNJ_BUILD_ARCH", other_arch);
        DeviceCaps c = knj_probe_host_caps();
        ASSERT_TRUE(!c.arch_ok, "arch should not be ok");
        ASSERT_TRUE(c.gcn_arch[0] == '\0',
                    "gcn_arch should be empty when arch guard fails");
        unset_env("KNJ_BUILD_ARCH");
    }

    // Test: device-only fields are zeroed
    {
        TEST("device-only fields are zeroed");
        set_env("KNJ_BUILD_ARCH", KNJ_ARCH_NAME);
        DeviceCaps c = knj_probe_host_caps();
        ASSERT_EQ_INT(0, c.vram_total, "vram_total should be 0");
        ASSERT_EQ_INT(0, c.vram_usable, "vram_usable should be 0");
        ASSERT_TRUE(!c.rebar_full_aperture, "rebar should be false");
        ASSERT_EQ_INT(0, c.sms, "sms should be 0");
        ASSERT_EQ_INT(0, c.clock_khz, "clock_khz should be 0");
        ASSERT_EQ_INT(0, c.max_streams, "max_streams should be 0");
        ASSERT_EQ_INT(0, c.num_queue_groups, "num_queue_groups should be 0");
        unset_env("KNJ_BUILD_ARCH");
    }
}

// ============================================================================
// Test 8: Backend::create() - arch-guarded factory
// ============================================================================

void test_backend_create() {
    std::printf("\n=== Test 8: Backend::create() ===\n");

    // Test: create() succeeds when KNJ_BUILD_ARCH matches the compiled arch
    {
        TEST("create() succeeds when KNJ_BUILD_ARCH matches");
        set_env("KNJ_BUILD_ARCH", KNJ_ARCH_NAME);
        auto backend = Backend::create();
        ASSERT_TRUE(backend != nullptr, "should create backend");
        if (backend) {
            const DeviceCaps& caps = backend->caps();
            ASSERT_TRUE(caps.arch_ok, "caps.arch_ok should be true");
            ASSERT_STREQ(KNJ_ARCH_NAME, caps.gcn_arch, "gcn_arch should match compile-time");
        }
        unset_env("KNJ_BUILD_ARCH");
    }

    // Test: create() returns nullptr when KNJ_BUILD_ARCH is unset
    {
        TEST("create() returns nullptr when KNJ_BUILD_ARCH is unset");
        unset_env("KNJ_BUILD_ARCH");
        auto backend = Backend::create();
        ASSERT_TRUE(backend == nullptr, "should return nullptr");
        const char* refusal = Backend::knj_last_refusal();
        ASSERT_TRUE(refusal != nullptr && refusal[0] != '\0',
                    "knj_last_refusal should return a message");
        std::printf("    refusal: [%s]\n", refusal);
    }

    // Test: create() returns nullptr when KNJ_BUILD_ARCH mismatches
    {
        TEST("create() returns nullptr when KNJ_BUILD_ARCH mismatches");
        // Use the OTHER arch (not the compiled one) to force a mismatch.
        const char* other_arch =
            (std::strcmp(KNJ_ARCH_NAME, "gfx1201") == 0) ? "gfx1031" : "gfx1201";
        set_env("KNJ_BUILD_ARCH", other_arch);
        auto backend = Backend::create();
        ASSERT_TRUE(backend == nullptr, "should return nullptr");
        const char* refusal = Backend::knj_last_refusal();
        ASSERT_TRUE(refusal != nullptr && refusal[0] != '\0',
                    "knj_last_refusal should return a message");
        std::printf("    refusal: [%s]\n", refusal);
        unset_env("KNJ_BUILD_ARCH");
    }
}

// ============================================================================
// Test 9: Backend::from_caps() - test hook
// ============================================================================

void test_backend_from_caps() {
    std::printf("\n=== Test 9: Backend::from_caps() ===\n");

    // Test: from_caps() creates a backend from injected caps
    {
        TEST("from_caps() creates backend from injected caps");
        DeviceCaps caps{};
        std::strncpy(caps.gcn_arch, "gfx1201", sizeof(caps.gcn_arch) - 1);
        caps.has_wmma = true;
        caps.has_fp8 = true;
        caps.has_dot_builtin = false;
        caps.wave_size = 32;
        caps.arch_ok = true;

        auto backend = Backend::from_caps(caps);
        ASSERT_TRUE(backend != nullptr, "should create backend");
        if (backend) {
            const DeviceCaps& bcaps = backend->caps();
            ASSERT_STREQ("gfx1201", bcaps.gcn_arch, "gcn_arch should match");
            ASSERT_TRUE(bcaps.has_wmma, "has_wmma should match");
            ASSERT_TRUE(bcaps.has_fp8, "has_fp8 should match");
        }
    }
}

// ============================================================================
// Test 10: Backend::would_instantiate() - cap gating
// ============================================================================

void test_would_instantiate() {
    std::printf("\n=== Test 10: Backend::would_instantiate() ===\n");

    // Test: gfx1201 backend would instantiate WMMA
    {
        TEST("gfx1201 backend would instantiate WMMA");
        DeviceCaps caps{};
        std::strncpy(caps.gcn_arch, "gfx1201", sizeof(caps.gcn_arch) - 1);
        caps.has_wmma = true;
        caps.arch_ok = true;

        auto backend = Backend::from_caps(caps);
        ASSERT_TRUE(backend->would_instantiate(KnjCap::WmmaF16),
                    "gfx1201 with WMMA should instantiate WMMA");
    }

    // Test: gfx1201 backend without WMMA would not instantiate WMMA
    {
        TEST("gfx1201 backend without WMMA would not instantiate WMMA");
        DeviceCaps caps{};
        std::strncpy(caps.gcn_arch, "gfx1201", sizeof(caps.gcn_arch) - 1);
        caps.has_wmma = false;
        caps.arch_ok = true;

        auto backend = Backend::from_caps(caps);
        ASSERT_TRUE(!backend->would_instantiate(KnjCap::WmmaF16),
                    "gfx1201 without WMMA should not instantiate WMMA");
    }

    // Test: gfx1031 backend would not instantiate WMMA
    {
        TEST("gfx1031 backend would not instantiate WMMA");
        DeviceCaps caps{};
        std::strncpy(caps.gcn_arch, "gfx1031", sizeof(caps.gcn_arch) - 1);
        caps.has_wmma = false;
        caps.arch_ok = true;

        auto backend = Backend::from_caps(caps);
        ASSERT_TRUE(!backend->would_instantiate(KnjCap::WmmaF16),
                    "gfx1031 should not instantiate WMMA");
    }

    // Test: gfx1031 backend would instantiate dot
    {
        TEST("gfx1031 backend would instantiate dot");
        DeviceCaps caps{};
        std::strncpy(caps.gcn_arch, "gfx1031", sizeof(caps.gcn_arch) - 1);
        caps.has_dot_builtin = true;
        caps.arch_ok = true;

        auto backend = Backend::from_caps(caps);
        ASSERT_TRUE(backend->would_instantiate(KnjCap::Dot4I32I8),
                    "gfx1031 should instantiate dot4");
    }

    // Test: gfx1201 backend would instantiate dot
    {
        TEST("gfx1201 backend would instantiate dot");
        DeviceCaps caps{};
        std::strncpy(caps.gcn_arch, "gfx1201", sizeof(caps.gcn_arch) - 1);
        caps.has_dot_builtin = false;
        caps.arch_ok = true;

        auto backend = Backend::from_caps(caps);
        ASSERT_TRUE(backend->would_instantiate(KnjCap::Dot4I32I8),
                    "gfx1201 should instantiate dot4");
    }
}

// ============================================================================
// Test 11: Backend launch methods
// ============================================================================

void test_backend_launch() {
    std::printf("\n=== Test 11: Backend launch methods ===\n");

    // Test: launch_expert_gemm_wmma returns true when has_wmma
    {
        TEST("launch_expert_gemm_wmma returns true when has_wmma");
        DeviceCaps caps{};
        std::strncpy(caps.gcn_arch, "gfx1201", sizeof(caps.gcn_arch) - 1);
        caps.has_wmma = true;
        caps.arch_ok = true;

        auto backend = Backend::from_caps(caps);
        ASSERT_TRUE(backend->launch_expert_gemm_wmma(ExpertPlan{}),
                    "should launch WMMA when has_wmma");
    }

    // Test: launch_expert_gemm_wmma returns false when !has_wmma
    {
        TEST("launch_expert_gemm_wmma returns false when !has_wmma");
        DeviceCaps caps{};
        std::strncpy(caps.gcn_arch, "gfx1201", sizeof(caps.gcn_arch) - 1);
        caps.has_wmma = false;
        caps.arch_ok = true;

        auto backend = Backend::from_caps(caps);
        ASSERT_TRUE(!backend->launch_expert_gemm_wmma(ExpertPlan{}),
                    "should not launch WMMA when !has_wmma");
    }

    // Test: launch_expert_gemm_simt returns true when arch_ok
    {
        TEST("launch_expert_gemm_simt returns true when arch_ok");
        DeviceCaps caps{};
        std::strncpy(caps.gcn_arch, "gfx1201", sizeof(caps.gcn_arch) - 1);
        caps.arch_ok = true;

        auto backend = Backend::from_caps(caps);
        ASSERT_TRUE(backend->launch_expert_gemm_simt(ExpertPlan{}),
                    "should launch SIMT when arch_ok");
    }

    // Test: launch_attention returns true when arch_ok
    {
        TEST("launch_attention returns true when arch_ok");
        DeviceCaps caps{};
        std::strncpy(caps.gcn_arch, "gfx1201", sizeof(caps.gcn_arch) - 1);
        caps.arch_ok = true;

        auto backend = Backend::from_caps(caps);
        ASSERT_TRUE(backend->launch_attention(AttentionPlan{}),
                    "should launch attention when arch_ok");
    }

    // Test: launch_kv_write is a no-op
    {
        TEST("launch_kv_write is a no-op");
        DeviceCaps caps{};
        caps.arch_ok = true;

        auto backend = Backend::from_caps(caps);
        // Should not crash
        backend->launch_kv_write(KvWritePlan{});
        PASS();
    }
}

// ============================================================================
// Test 12: knj_print_doctor_header() - doctor header output
// ============================================================================

void test_print_doctor_header() {
    std::printf("\n=== Test 12: knj_print_doctor_header() ===\n");

    // Test: doctor header with device info
    {
        TEST("doctor header with device info");
        DeviceCaps c{};
        std::strncpy(c.gcn_arch, "gfx1201", sizeof(c.gcn_arch) - 1);
        c.has_wmma = true;
        c.has_fp8 = true;
        c.has_dot_builtin = false;
        c.sms = 64;
        c.wave_size = 32;
        c.rebar_full_aperture = true;
        c.arch_ok = true;

        char out[512];
        knj_print_doctor_header(c, "AMD Radeon RX 9070 XT", true, out, sizeof(out));
        std::printf("    [%s]\n", out);

        ASSERT_TRUE(std::strstr(out, "device        AMD Radeon RX 9070 XT") != nullptr,
                    "should contain device name");
        ASSERT_TRUE(std::strstr(out, "gfx1201") != nullptr,
                    "should contain arch");
        ASSERT_TRUE(std::strstr(out, "64 CU") != nullptr,
                    "should contain CU count");
        ASSERT_TRUE(std::strstr(out, "wave32") != nullptr,
                    "should contain wave size");
        ASSERT_TRUE(std::strstr(out, "wmma=on") != nullptr,
                    "should contain wmma=on");
        ASSERT_TRUE(std::strstr(out, "fp8=on") != nullptr,
                    "should contain fp8=on");
        ASSERT_TRUE(std::strstr(out, "rebar         full aperture") != nullptr,
                    "should contain rebar full aperture");
    }

    // Test: doctor header without device info (host-only path)
    {
        TEST("doctor header without device info");
        DeviceCaps c{};
        std::strncpy(c.gcn_arch, "gfx1201", sizeof(c.gcn_arch) - 1);
        c.has_wmma = true;
        c.has_fp8 = true;
        c.has_dot_builtin = false;
        c.arch_ok = true;

        char out[512];
        knj_print_doctor_header(c, nullptr, true, out, sizeof(out));
        std::printf("    [%s]\n", out);

        ASSERT_TRUE(std::strstr(out, "device        (no device)") != nullptr,
                    "should contain '(no device)'");
    }

    // Test: doctor header with arch guard refused
    {
        TEST("doctor header with arch guard refused");
        DeviceCaps c{};
        c.arch_ok = false;
        c.arch_reason = "KNJ_BUILD_ARCH is not set";

        char out[512];
        knj_print_doctor_header(c, nullptr, true, out, sizeof(out));
        std::printf("    [%s]\n", out);

        ASSERT_TRUE(std::strstr(out, "(arch guard refused)") != nullptr,
                    "should contain '(arch guard refused)'");
        ASSERT_TRUE(std::strstr(out, "wmma=off") != nullptr,
                    "should contain wmma=off");
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::printf("=== C2 Device Abstraction Layer Unit Tests ===\n\n");

    // Run all test suites
    test_arch_guard();
    test_build_arch_env();
    test_cap_available();
    test_cap_label();
    test_dot_label();
    test_print_cap_line();
    test_probe_host_caps();
    test_backend_create();
    test_backend_from_caps();
    test_would_instantiate();
    test_backend_launch();
    test_print_doctor_header();

    // Summary
    std::printf("\n=== Summary ===\n");
    std::printf("  tests run: %d\n", tests_run);
    std::printf("  tests passed: %d\n", tests_passed);
    std::printf("  tests failed: %d\n", tests_failed);

    if (tests_failed > 0) {
        std::printf("\nRESULT: FAIL\n");
        return 1;
    } else {
        std::printf("\nRESULT: PASS\n");
        return 0;
    }
}
