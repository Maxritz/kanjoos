// tests/unit/test_tok_whitespace_run.cpp -- the pre-tokeniser's whitespace run.
//
// What this is defending
// ----------------------
// The reference pre-tokeniser's last alternatives include `\s+(?!\S)`, and that
// lookahead makes the regex engine give back ONE CHARACTER out of a whitespace
// run so that a leading space can still be taken by the following alternative
// ("  hi" is [" ", " hi"], not ["  ", "hi"]). This engine emulated the backtrack
// by subtracting one BYTE:
//
//     else if (j - i >= 2) len = (j - i) - 1;      // one byte, not one character
//
// For ASCII whitespace one byte is one character, so every case the hand-written
// string set covered was right. For a whitespace character wider than one byte it
// cut the character in half, and a chunk boundary is where a BPE merge is
// forbidden to cross -- so the character's own merge could not fire, and the two
// halves were emitted as separate byte tokens.
//
// This was not found by the 25-string cross-check. It was found by
// tools/tok_crosscheck.py's fuzz pass, reduced to two inputs, and reproduced on
// the real Qwen3-MOE-4x0.6B file against llama.cpp (2026-10-08):
//
//     "U+00A0 U+00B4"  engine [126, 254, 28111]   llama [4102, 28111]
//     "U+3000 U+00B4"  engine [1277, 222, 28111]  llama [22441, 28111]
//
// This test is the same case against a vocabulary whose ids are small enough to
// state exactly. The fixtures are written by tools/make_tok_fixtures.py, whose
// vocabulary carries one merged token per whitespace character (ids 258-261), so
// "the whole character formed one chunk" and "the character was split" produce
// different, assertable id lists:
//
//     NBSP + U+00B4 -> [258, 259]   (one token for the character, one for acute)
//     U+3000 + U+00B4 -> [261, 259]
//
// The controls are the ASCII cases the byte count already got right; they are
// asserted EXACTLY so that a future change to this branch cannot quietly move
// them. Exit codes: 0 every gate passed, 1 at least one failed.
//
// Run (CMake wires this as the ctest case `tok_whitespace_run`):
//   cmake --build build/cmake-host --target test_tok_whitespace_run
//   ctest --test-dir build/cmake-host -R tok_whitespace_run --output-on-failure

#include "src/loader/gguf.h"
#include "src/tokenizer/tokenizer.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failed = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failed;
    std::printf("      FAIL  %s\n", what.c_str());
  }
}

#define CHECK(x) check((x), #x)

#ifndef KNJ_TOK_FIXTURE_DIR
#error "KNJ_TOK_FIXTURE_DIR must be defined: this test needs the fixture directory"
#endif

const std::string kFixtureDir = KNJ_TOK_FIXTURE_DIR;

std::string join_ids(const std::vector<int32_t>& ids) {
  std::string out;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(ids[i]);
  }
  return out;
}

bool ids_are(const std::vector<int32_t>& ids, std::vector<int32_t> want) {
  return ids == want;
}

// Built as raw bytes so the source encoding cannot change what is measured.
// U+00A0 = C2 A0, U+3000 = E3 80 80, U+00B4 = C2 B4, U+0020 = 20.
const char kSpace[] = "\x20";
const char kNbsp[] = "\xC2\xA0";
const char kIdeo[] = "\xE3\x80\x80";
const char kAcute[] = "\xC2\xB4";

std::string cat(std::initializer_list<const char*> parts) {
  std::string s;
  for (const char* p : parts) s += p;
  return s;
}

}  // namespace

int main() {
  std::printf("== pre-tokeniser whitespace run ==\n");
  std::printf("fixtures   : %s\n", kFixtureDir.c_str());

  if (!std::filesystem::is_directory(kFixtureDir)) {
    std::printf("FAILED: the fixture directory does not exist. Run:\n"
                "  python tools/make_tok_fixtures.py\n");
    return 1;
  }

  knj::Tokenizer tk;
  try {
    tk = knj::Tokenizer::from_gguf(
        knj::GgufFile::open(kFixtureDir + "/pre_qwen2.gguf"));
  } catch (const std::exception& e) {
    std::printf("FAILED: could not build the tokenizer: %s\n", e.what());
    return 1;
  }

  // 1. A single whitespace character wider than one byte, followed by a
  //    character the vocabulary merges: the space and the acute each become one
  //    token. Both assertions fail on the byte-counted backtrack, which gives
  //    [194, 160, 259] and [260, 128, 259] instead.
  const std::string nbsp_acute = cat({kNbsp, kAcute});
  const std::vector<int32_t> got_nbsp = tk.encode(nbsp_acute);
  std::printf("  NBSP + U+00B4   -> [%s]   want [258, 259]\n", join_ids(got_nbsp).c_str());
  CHECK(ids_are(got_nbsp, {258, 259}));

  const std::string ideo_acute = cat({kIdeo, kAcute});
  const std::vector<int32_t> got_ideo = tk.encode(ideo_acute);
  std::printf("  U+3000 + U+00B4 -> [%s]   want [261, 259]\n", join_ids(got_ideo).c_str());
  CHECK(ids_are(got_ideo, {261, 259}));

  // 2. No bytes are lost: the two cases above must decode back to their own
  //    bytes. This holds even when the split bug is present (the two halves are
  //    valid byte tokens), which is exactly why it is asserted as an invariant
  //    and NOT as the discriminator -- the exact ids above are the discriminator.
  CHECK(tk.decode(got_nbsp) == nbsp_acute);
  CHECK(tk.decode(got_ideo) == ideo_acute);

  // 3. The ASCII controls. One byte is one character here, so these were right
  //    before the change and must stay right: two spaces then a word is
  //    [" ", " hi"], and a single space then the acute merges the acute.
  const std::vector<int32_t> got_two_spaces = tk.encode("  hi");
  std::printf("  \"  hi\"         -> [%s]   want [32, 32, 104, 105]\n",
              join_ids(got_two_spaces).c_str());
  CHECK(ids_are(got_two_spaces, {32, 32, 104, 105}));

  const std::vector<int32_t> got_space_acute = tk.encode(cat({kSpace, kAcute}));
  std::printf("  \" \" + U+00B4  -> [%s]   want [32, 259]\n",
              join_ids(got_space_acute).c_str());
  CHECK(ids_are(got_space_acute, {32, 259}));

  // 4. A whitespace run at end of text is branch 6 of the reference
  //    (`\s+(?!\S)` matching the whole run, no backtracking), and a CR/LF run is
  //    branch 5. Asserted so the two branches cannot be swapped silently.
  CHECK(ids_are(tk.encode(cat({kNbsp, kNbsp})), {258, 258}));
  CHECK(ids_are(tk.encode("  "), {32, 32}));

  // 5. The same whitespace rule under both `pre` names: the fixtures differ only
  //    in that field, so if this ever becomes rule-dependent, one of the two
  //    files changes its answer here.
  knj::Tokenizer tk35;
  try {
    tk35 = knj::Tokenizer::from_gguf(
        knj::GgufFile::open(kFixtureDir + "/pre_qwen35.gguf"));
  } catch (const std::exception& e) {
    std::printf("FAILED: could not build the qwen35 fixture: %s\n", e.what());
    return 1;
  }
  CHECK(tk35.encode(nbsp_acute) == got_nbsp);
  CHECK(tk35.encode(ideo_acute) == got_ideo);

  std::printf("%d check(s), %d failed\n", g_checks, g_failed);
  if (g_failed) {
    std::printf("FAILED\n");
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
