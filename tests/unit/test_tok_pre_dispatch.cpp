// tests/unit/test_tok_pre_dispatch.cpp -- the tokenizer.ggml.pre dispatch gate.
//
// What this is defending
// ----------------------
// The tokenizer reads `tokenizer.ggml.pre` and applies the pre-tokeniser rule
// THAT NAME selects. Before it did, every file got the qwen2 rule, including
// files that declared `qwen35`, and nothing failed: the token ids simply changed.
// A wrong pre-tokeniser is invisible in every other way -- same vocabulary, same
// merge table, same exit code -- so the only defence is a file that measures it.
//
// These fixtures are that file, once per honoured name (see the "known pre" line
// in the output for the current count). Every fixture under tests/fixtures/tok/
// carries the SAME vocabulary and the SAME merge table and differs only in the
// value of `tokenizer.ggml.pre`, written by tools/make_tok_fixtures.py. So a
// difference between two of them is the dispatch and nothing else.
//
// Which names the table should hold is not a matter of opinion: it is DERIVED
// from the reference's actual regexes by tools/tok_pre_rules.py, which parses
// llama.cpp's name map and pre-type -> regex table and pins the result in
// tests/fixtures/tok/pre_rules.expected. Section 1b below asserts SET EQUALITY
// between that file and `knj::known_pre_names()`, in both directions, so this
// table cannot drift from the reference by a name again.
//
// The one place the two rules differ
// ----------------------------------
// qwen2 matches `[\p{L}]+`, qwen35 matches `[\p{L}\p{M}]+`: a combining mark is
// punctuation under one and part of the letter run under the other. That moves a
// chunk boundary, and a chunk boundary is where a BPE merge is forbidden to
// cross. The fixtures encode exactly that: a merge that joins "a" with the two
// bytes of U+0301 exists in both files, so under qwen35 it fires (one letter run)
// and under qwen2 it cannot (the boundary sits between the two halves).
//
//     qwen2  "a\u0301b" -> [97, 204, 129, 98]   4 tokens, no merge fired
//     qwen35 "a\u0301b" -> [257, 98]            2 tokens, both merges fired
//
// Both are asserted EXACTLY rather than by inequality, and the control case
// ("ab", no mark) is asserted to agree across the two files -- a test that only
// said "the ids differ" would also pass if the two rules differed everywhere.
//
// Exit codes:
//   0  every gate passed
//   1  at least one gate failed
//
// Run (CMake wires this as the ctest case `tok_pre_dispatch`):
//   cmake --build build/cmake-host --target test_tok_pre_dispatch
//   ctest --test-dir build/cmake-host -R tok_pre_dispatch --output-on-failure

#include "src/loader/gguf.h"
#include "src/tokenizer/tokenizer.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
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

std::string fixture(const std::string& file) { return kFixtureDir + "/" + file; }

// A fixture whose `pre` is a name this engine honours, opened and built.
knj::Tokenizer open_named(const std::string& name) {
  return knj::Tokenizer::from_gguf(knj::GgufFile::open(fixture("pre_" + name + ".gguf")));
}

// The refusal message from a fixture that must not be built. Returns "" when the
// file opened successfully, which is itself the failure.
std::string refusal_message(const std::string& file) {
  try {
    knj::GgufFile f = knj::GgufFile::open(fixture(file));
    (void)knj::Tokenizer::from_gguf(f);
    return std::string();
  } catch (const std::exception& e) {
    return std::string(e.what());
  }
}

std::string join_ids(const std::vector<int32_t>& ids) {
  std::string out;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(ids[i]);
  }
  return out;
}

// The engine's table as (name -> rule name): the same shape the expectation file
// holds, so the two can be compared as SETS rather than as lists.
std::map<std::string, std::string> engine_table() {
  std::map<std::string, std::string> out;
  for (const char* raw : knj::known_pre_names()) {
    const std::string name = raw;
    knj::PreRule r = knj::PreRule::Qwen2;
    out[name] = knj::resolve_pre_rule(name, &r) ? knj::pre_rule_name(r)
                                                : "<unresolved>";
  }
  return out;
}

// "name rule" per line, `#` comments and blanks skipped. `ok` is false only when
// the FILE cannot be read or is malformed -- an empty-but-present file is
// reported by the caller as a set difference, not swallowed here.
std::map<std::string, std::string> read_expectation(const std::string& path, bool* ok) {
  std::map<std::string, std::string> out;
  std::ifstream in(path);
  if (!in) {
    *ok = false;
    return out;
  }
  *ok = true;
  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    const size_t sp = line.find(' ');
    if (sp == std::string::npos) {
      *ok = false;
      std::printf("      FAIL  %s:%d: no rule column\n", path.c_str(), lineno);
      continue;
    }
    out[line.substr(0, sp)] = line.substr(sp + 1);
  }
  return out;
}

// "a" + U+0301 (COMBINING ACUTE ACCENT, UTF-8 CC 81) + "b", as raw bytes. Written
// as bytes rather than as a \u escape so the source encoding cannot change what
// this test measures, and SPLIT after the escape because a C++ hex escape is
// greedy: "\x81b" is one escape for 0x81b, which does not fit a char. The literal
// is built by concatenation so the 'b' cannot be absorbed into the mark.
const char kMarkText[] = "a\xCC\x81" "b";
const char kPlainText[] = "ab";

}  // namespace

int main() {
  std::printf("== tokenizer.ggml.pre dispatch ==\n");
  std::printf("fixtures   : %s\n", kFixtureDir.c_str());

  if (!std::filesystem::is_directory(kFixtureDir)) {
    std::printf("FAILED: the fixture directory does not exist. Run:\n"
                "  python tools/make_tok_fixtures.py\n");
    return 1;
  }

  // -- 1. every name the engine honours has a fixture, and each resolves to the
  //       rule that name selects. -------------------------------------------
  const std::vector<const char*> names = knj::known_pre_names();
  std::printf("known pre  : %d name(s)\n", (int)names.size());
  CHECK(names.size() > 0);

  int qwen2_count = 0, qwen35_count = 0;
  for (const char* raw : names) {
    const std::string name = raw;
    const std::string file = "pre_" + name + ".gguf";

    if (!std::filesystem::exists(fixture(file))) {
      check(false, "fixture exists for pre='" + name + "' (" + file +
                       "): run python tools/make_tok_fixtures.py");
      continue;
    }

    knj::PreRule expected = knj::PreRule::Qwen2;
    const bool resolvable = knj::resolve_pre_rule(name, &expected);
    CHECK(resolvable);

    knj::Tokenizer tk;
    bool built = true;
    try {
      tk = open_named(name);
    } catch (const std::exception& e) {
      built = false;
      std::printf("      FAIL  pre='%s' did not build: %s\n", name.c_str(), e.what());
    }
    CHECK(built);
    if (!built) continue;

    // The declared value reached the tokenizer, and it selected that name's rule.
    CHECK(tk.pre() == name);
    CHECK(tk.pre_rule() == expected);
    // The rule the tokenizer reports and the rule the name selects are the same
    // object, so the printed rule cannot disagree with the applied one.
    CHECK(std::strcmp(knj::pre_rule_name(tk.pre_rule()), knj::pre_rule_name(expected)) == 0);

    if (tk.pre_rule() == knj::PreRule::Qwen35) {
      ++qwen35_count;
    } else {
      ++qwen2_count;
    }
    std::printf("  %-20s pre=%-18s -> rule %s\n", file.c_str(), name.c_str(),
                knj::pre_rule_name(tk.pre_rule()));
  }

  // The table must actually contain the non-qwen2 rule: a table that mapped every
  // name to qwen2 would pass everything above while proving nothing.
  CHECK(qwen35_count > 0);
  CHECK(qwen2_count > 0);
  std::printf("table      : %d name(s) -> qwen2, %d -> qwen35\n", qwen2_count, qwen35_count);

  // -- 1b. the table is DERIVED, not merely self-consistent: it must equal, as a
  //        set, what tools/tok_pre_rules.py read out of the reference's own name
  //        map and regex table. A name on either side without the other is a
  //        failure -- that check is what found `megrez` and `grok-2`, which the
  //        table was missing while their regex is the qwen2 one verbatim. -----
  const std::string exp_path = kFixtureDir + "/pre_rules.expected";
  bool exp_readable = false;
  const std::map<std::string, std::string> expected = read_expectation(exp_path, &exp_readable);
  std::printf("-- derived table vs engine --\n");
  CHECK(exp_readable);
  if (!exp_readable) {
    std::printf("      FAIL  %s is missing or malformed; regenerate with:\n"
                "              python tools/tok_pre_rules.py --emit %s\n",
                exp_path.c_str(), exp_path.c_str());
  } else {
    CHECK(!expected.empty());
    const std::map<std::string, std::string> engine = engine_table();
    for (const auto& kv : expected) {
      const auto it = engine.find(kv.first);
      if (it == engine.end()) {
        check(false, "the engine's table is missing '" + kv.first +
                         "', which the reference derives as rule " + kv.second);
      } else if (it->second != kv.second) {
        check(false, "the engine maps '" + kv.first + "' to " + it->second +
                         " but the reference derives " + kv.second);
      }
    }
    for (const auto& kv : engine) {
      if (expected.find(kv.first) == expected.end()) {
        check(false, "the engine claims '" + kv.first +
                         "', which the reference's regexes do not derive");
      }
    }
    check(engine.size() == expected.size(),
          "the engine table and the derived table have the same size (" +
              std::to_string(engine.size()) + " vs " + std::to_string(expected.size()) + ")");
    std::printf("  derived=%d engine=%d\n", (int)expected.size(), (int)engine.size());
  }

  // -- 2. the rules are not interchangeable: a mark moves the boundary and the
  //       merge that would cross it fires or does not. ------------------------
  std::printf("-- the mark, on identical vocabularies --\n");
  knj::Tokenizer q2, q35;
  bool have_both = true;
  try {
    q2 = open_named("qwen2");
    q35 = open_named("qwen35");
  } catch (const std::exception& e) {
    have_both = false;
    std::printf("      FAIL  could not open the qwen2/qwen35 pair: %s\n", e.what());
  }
  CHECK(have_both);

  if (have_both) {
    // The vocabularies must be identical, or the ids below would be measuring the
    // files rather than the rules.
    CHECK(q2.vocab_size() == q35.vocab_size());
    CHECK(q2.merges() == q35.merges());

    const std::vector<int32_t> mark2 = q2.encode(kMarkText, false);
    const std::vector<int32_t> mark35 = q35.encode(kMarkText, false);
    const std::vector<int32_t> plain2 = q2.encode(kPlainText, false);
    const std::vector<int32_t> plain35 = q35.encode(kPlainText, false);

    std::printf("  qwen2  %-14s -> [%s]  (%d token(s))\n", "a+U+0301+b",
                join_ids(mark2).c_str(), (int)mark2.size());
    std::printf("  qwen35 %-14s -> [%s]  (%d token(s))\n", "a+U+0301+b",
                join_ids(mark35).c_str(), (int)mark35.size());

    // Exact, not merely different. qwen2: 'a', then the mark's two byte tokens,
    // then 'b' -- the merge cannot cross the chunk boundary. qwen35: one letter
    // run, both merges fire.
    const std::vector<int32_t> want2 = {97, 204, 129, 98};
    const std::vector<int32_t> want35 = {257, 98};
    CHECK(mark2 == want2);
    CHECK(mark35 == want35);
    check(mark2 != mark35, "the two rules produce different ids for a mark");

    // Control: without a mark the two rules must AGREE. Without this, "the ids
    // differ" would also be satisfied by two rules that differ everywhere.
    CHECK(plain2 == plain35);
    std::printf("  both   %-14s -> [%s]  (control: no mark, rules agree)\n", "ab",
                join_ids(plain2).c_str());

    // Round-trip: the ids must decode back to the exact bytes the file was
    // given, so a merge that fires cannot silently drop one.
    const std::string back = q35.decode(mark35);
    CHECK(back.size() == 4);
    CHECK(back == std::string(kMarkText));
  }

  // -- 3. a `pre` this engine cannot honour is refused, naming the value. ----
  std::printf("-- refusals --\n");
  const std::string absent = refusal_message("pre_absent.gguf");
  CHECK(!absent.empty());
  std::printf("  pre_absent.gguf  : %s\n", absent.empty() ? "<BUILT -- not refused>" : absent.c_str());
  CHECK(absent.find("declares no tokenizer.ggml.pre") != std::string::npos);

  const std::string unknown = refusal_message("pre_unknown.gguf");
  CHECK(!unknown.empty());
  std::printf("  pre_unknown.gguf : %s\n", unknown.empty() ? "<BUILT -- not refused>" : unknown.c_str());
  // The message must name the offending value, not just complain.
  CHECK(unknown.find("'llama3'") != std::string::npos);
  CHECK(unknown.find("does not") != std::string::npos);

  // -- 4. the directory has no fixtures this test does not know about: the same
  //       drift check the generator does, from the consumer's side. ----------
  int strays = 0;
  for (const std::filesystem::directory_entry& e :
       std::filesystem::directory_iterator(kFixtureDir)) {
    if (e.path().extension() != ".gguf") continue;
    const std::string stem = e.path().stem().string();
    bool known = false;
    for (const char* n : names) {
      if (stem == std::string("pre_") + n) { known = true; break; }
    }
    if (!known && (stem == "pre_absent" || stem == "pre_unknown")) known = true;
    if (!known) {
      ++strays;
      std::printf("      FAIL  unexpected fixture %s\n", e.path().filename().string().c_str());
    }
  }
  CHECK(strays == 0);

  std::printf("%d check(s), %d failed\n", g_checks, g_failed);
  if (g_failed) {
    std::printf("FAILED: the pre dispatch does not hold\n");
    return 1;
  }
  std::printf("PASS: every declared `pre` selects its own rule; an unimplemented one is refused\n");
  return 0;
}
