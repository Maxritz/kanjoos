// tests/unit/test_tok_pre_sweep.cpp -- the pre-tokeniser alternative sweep.
//
// What this is defending
// ----------------------
// Only the whitespace-run alternative (`\s+(?!\S)`) was ever swept against
// llama.cpp's behaviour (the separator sweep). The other six alternatives of
// the QWEN2/QWEN35 regex were emulated in `Tokenizer::pretokenize` from their
// prose. This gate asserts all seven, per rule, against splits derived from
// the reference's ACTUAL pattern strings: tools/tok_pre_sweep.py extracts the
// QWEN2/QWEN35 literals mechanically from llama.cpp's src/llama-vocab.cpp,
// runs them over crafted inputs, and the reviewed output
// (records/tok-pre-sweep-2026-10-10.log) is hardcoded here EXACTLY.
//
// The load-bearing cases, and why each is here:
//   contractions   "'s" is one chunk but "''s" is two; "'n" is not a suffix
//                  ("rock 'n' roll" -> " '", "n", "'"); case-insensitive.
//   letter runs    digits split runs ("a1b", "abc123"); a digit is never the
//                  optional prefix ("1a" -> "1","a"); "_" and "-" attach to
//                  the FOLLOWING run ("_score", "-op").
//   numbers        one codepoint at a time ("123" -> three chunks).
//   punctuation    optional leading space, trailing CR/LF attach ("!?\n");
//                  non-letters attach forward ("a!b" -> "a","!b").
//   newlines       "a \n b" -> "a"," \n"," b" (alt 5 greedy through the LF).
//   marks          the one rule difference: "a\u0301b" is ["a","\u0301b"]
//                  under qwen2 (mark is a valid alt-2 prefix) and one chunk
//                  under qwen35.
//   CJK/emoji      Han is letters (one chunk); rocket is punctuation and
//                  attaches forward like "!" does.
//   spaces         "  hi" -> " "," hi" (alt 6 backtracks one CHARACTER);
//                  "a  b" -> "a"," "," b".
//
// A test that only counted chunks would pass on shifted boundaries; every row
// asserts the exact chunk strings.
//
// Exit codes:
//   0  every row matched
//   1  at least one row diverged
//
// Run (CMake wires this as the ctest case `tok_pre_sweep`):
//   cmake --build build/cmake-host --target test_tok_pre_sweep
//   ctest --test-dir build/cmake-host -R tok_pre_sweep --output-on-failure

#include "src/tokenizer/tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Row {
  const char* label;
  knj::PreRule rule;
  const char* input;
  std::vector<std::string> want;
};

#define Q2 knj::PreRule::Qwen2
#define Q35 knj::PreRule::Qwen35

const Row kRows[] = {
    {"contr-lower", Q2, "don't", {"don", "'t"}},
    {"contr-upper", Q2, "DON'T", {"DON", "'T"}},
    {"contr-ve", Q2, "they've", {"they", "'ve"}},
    {"contr-ll", Q2, "I'll", {"I", "'ll"}},
    {"contr-re", Q2, "we're", {"we", "'re"}},
    {"contr-non", Q2, "rock 'n' roll", {"rock", " '", "n", "'", " roll"}},
    {"contr-trail", Q2, "cats'", {"cats", "'"}},
    {"contr-dbl", Q2, "''s", {"''", "s"}},
    {"contr-s", Q2, "'s", {"'s"}},
    {"letter-plain", Q2, "hello", {"hello"}},
    {"letter-space", Q2, "Hello World", {"Hello", " World"}},
    {"letter-digit-split", Q2, "a1b", {"a", "1", "b"}},
    {"letter-digit-run", Q2, "abc123", {"abc", "1", "2", "3"}},
    {"digit-letter", Q2, "123abc", {"1", "2", "3", "abc"}},
    {"digit-prefix-no", Q2, "1a", {"1", "a"}},
    {"letter-lead-space", Q2, " a", {" a"}},
    {"underscore", Q2, "under_score", {"under", "_score"}},
    {"hyphen", Q2, "co-op", {"co", "-op"}},
    {"number-run", Q2, "123", {"1", "2", "3"}},
    {"number-dot", Q2, "3.14", {"3", ".", "1", "4"}},
    {"punct-run", Q2, "!!!", {"!!!"}},
    {"punct-lead-space", Q2, " !!!", {" !!!"}},
    {"punct-crlf", Q2, "!?\n!\n", {"!?\n", "!\n"}},
    {"punct-word", Q2, "a!b", {"a", "!b"}},
    {"punct-emdash", Q2, "\u2014", {"\u2014"}},
    {"punct-cjk", Q2, "\u3002", {"\u3002"}},
    {"punct-cjk-word", Q2, "a\u3002b", {"a", "\u3002b"}},
    {"newline-mid", Q2, "a\nb", {"a", "\n", "b"}},
    {"newline-space", Q2, "a \n b", {"a", " \n", " b"}},
    {"newline-crlf", Q2, "a\r\n\r\nb", {"a", "\r\n\r\n", "b"}},
    {"cjk-letters", Q2, "\u65E5\u672C\u8A9E", {"\u65E5\u672C\u8A9E"}},
    {"emoji", Q2, "\U0001F680", {"\U0001F680"}},
    {"emoji-word", Q2, "a\U0001F680b", {"a", "\U0001F680b"}},
    {"mark-qwen2", Q2, "a\u0301b", {"a", "\u0301b"}},
    {"trail-one", Q2, "hi ", {"hi", " "}},
    {"trail-two", Q2, "hi  ", {"hi", "  "}},
    {"lead-two", Q2, "  hi", {" ", " hi"}},
    {"mid-two", Q2, "a  b", {"a", " ", " b"}},
    {"contr-lower-35", Q35, "don't", {"don", "'t"}},
    {"letter-plain-35", Q35, "hello", {"hello"}},
    {"cjk-letters-35", Q35, "\u65E5\u672C\u8A9E", {"\u65E5\u672C\u8A9E"}},
    {"mark-qwen35", Q35, "a\u0301b", {"a\u0301b"}},
};

std::string show(const std::vector<std::string>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ", ";
    s += "\"" + v[i] + "\"";
  }
  return s + "]";
}

}  // namespace

int main() {
  int fails = 0;
  for (const Row& r : kRows) {
    const std::vector<std::string> got =
        knj::Tokenizer::pretokenize(r.input, r.rule);
    const bool ok = got == r.want;
    if (!ok) ++fails;
    std::printf("  %-18s %-6s %s  got %s  want %s\n", r.label,
                r.rule == knj::PreRule::Qwen2 ? "qwen2" : "qwen35",
                ok ? "ok" : "FAIL", show(got).c_str(), show(r.want).c_str());
  }
  std::printf("%zu rows, %d failures\n", sizeof(kRows) / sizeof(kRows[0]), fails);
  return fails ? 1 : 0;
}
