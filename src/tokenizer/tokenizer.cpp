// src/tokenizer/tokenizer.cpp -- byte-level BPE, built from the GGUF itself.
#include "src/tokenizer/tokenizer.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "src/tokenizer/unicode_ranges.h"

namespace knj {

const char* pre_rule_name(PreRule r) {
  return r == PreRule::Qwen35 ? "qwen35" : "qwen2";
}

namespace {

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("tokenizer: " + m); }

// The `pre` names this engine can honour, and the rule each one selects.
//
// This table is DERIVED, not transcribed. tools/tok_pre_rules.py parses the
// reference's own source -- the name map in `llama_vocab::impl::load` and the
// pre-type -> `regex_exprs` table in the `llm_tokenizer_bpe` constructor
// (`src/llama-vocab.cpp`) -- and derives which regex list every known `pre` name
// runs. A name belongs in this table exactly when that list is byte-identical to
// one of the two regexes implemented below, AND its branch sets no flag that
// changes what the tokenizer produces. (`clean_spaces` is detokenize-only and is
// not such a flag; `ignore_merges`, `escape_whitespaces`, `byte_encode`,
// `add_space_prefix`, `add_sep` and `treat_whitespace_as_suffix` are.)
//
// Measured 2026-10-08 against llama.cpp 8b4b355: exactly ten names qualify, and
// all ten are listed here. `megrez` and `grok-2` were missing from the earlier,
// prose-written version of this table although their regex is the qwen2 one
// verbatim -- the tool's completeness check is what found them. The derived set
// is pinned in tests/fixtures/tok/pre_rules.expected and checked by
// test_tok_pre_dispatch, so a name added or dropped here without re-running
// `python tools/tok_pre_rules.py --emit ...` fails the suite.
//
// Only two of these names have been run against a real file in this repo
// (`qwen2` on Qwen3-MOE-4x0.6B, `qwen35` on ThinkingCap-Qwen3.8-27B); the others
// are listed because their regex is byte-identical to one of the two, and a name
// that is not in this table is a refusal rather than a guess.
struct PreName {
  const char* name;
  PreRule rule;
};
const PreName kPreNames[] = {
    {"qwen2", PreRule::Qwen2},
    {"deepseek-r1-qwen", PreRule::Qwen2},
    {"kormo", PreRule::Qwen2},
    {"f2llmv2", PreRule::Qwen2},
    {"megrez", PreRule::Qwen2},
    {"grok-2", PreRule::Qwen2},
    {"stablelm2", PreRule::Qwen2},
    {"hunyuan", PreRule::Qwen2},
    {"solar-open", PreRule::Qwen2},
    {"qwen35", PreRule::Qwen35},
};

// GPT-2's byte <-> codepoint permutation. Bytes that are already printable
// map to themselves; the rest are pushed into the 256.. range so that every
// byte has a distinct, printable representation.
struct ByteMap {
  std::array<uint32_t, 256> to_cp{};
  std::unordered_map<uint32_t, uint8_t> to_byte;

  ByteMap() {
    std::vector<int> printable;
    for (int b = 33; b <= 126; ++b) printable.push_back(b);
    for (int b = 161; b <= 172; ++b) printable.push_back(b);
    for (int b = 174; b <= 255; ++b) printable.push_back(b);
    std::array<bool, 256> used{};
    for (int b : printable) { to_cp[b] = (uint32_t)b; used[b] = true; }
    uint32_t n = 0;
    for (int b = 0; b < 256; ++b) {
      if (!used[b]) to_cp[b] = 256 + n++;
    }
    for (int b = 0; b < 256; ++b) to_byte[to_cp[b]] = (uint8_t)b;
  }
};

const ByteMap& byte_map() {
  static const ByteMap m;
  return m;
}

void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back((char)cp);
  } else if (cp < 0x800) {
    out.push_back((char)(0xC0 | (cp >> 6)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back((char)(0xE0 | (cp >> 12)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    out.push_back((char)(0xF0 | (cp >> 18)));
    out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  }
}

// Decode one UTF-8 codepoint at `i`; on malformed input consume one byte.
uint32_t read_utf8(const std::string& s, size_t i, size_t* len) {
  const uint8_t c = (uint8_t)s[i];
  if (c < 0x80) { *len = 1; return c; }
  size_t n = 0;
  uint32_t cp = 0;
  if ((c & 0xE0) == 0xC0) { n = 2; cp = c & 0x1F; }
  else if ((c & 0xF0) == 0xE0) { n = 3; cp = c & 0x0F; }
  else if ((c & 0xF8) == 0xF0) { n = 4; cp = c & 0x07; }
  else { *len = 1; return c; }
  if (i + n > s.size()) { *len = 1; return c; }
  for (size_t k = 1; k < n; ++k) {
    const uint8_t cc = (uint8_t)s[i + k];
    if ((cc & 0xC0) != 0x80) { *len = 1; return c; }
    cp = (cp << 6) | (cc & 0x3F);
  }
  *len = n;
  return cp;
}

// ---- character classes for the pre-tokeniser ------------------------------
// The reference regex is Unicode (\p{L}, \p{M}, \p{N}, \s), so the classes are
// per CODEPOINT, from the generated table in unicode_ranges.cpp. A byte test
// (">= 0x80 is a letter") is not a simplification of that rule, it is a
// DIFFERENT rule: it joins an emoji or a CJK punctuation mark to the word before
// it, and the chunk boundary decides which merges are even possible. Measured
// before this change: "日本語のテキスト" came out as 24 byte-level ids where
// llama.cpp produces 3 tokens, and "emoji: 🚀" joined the emoji to the word.
// CR/LF stay byte tests: the regex names them literally and they are ASCII.
inline bool is_crlf(uint8_t c) { return c == '\r' || c == '\n'; }

// The class of the codepoint at `p`, and how many bytes it is. `read_utf8`
// consumes exactly one byte on malformed input, so a scan cannot stall.
struct CpAt {
  UnicodeClass cls;
  size_t bytes;
};
inline CpAt cp_at(const std::string& t, size_t p) {
  size_t len = 0;
  const uint32_t cp = read_utf8(t, p, &len);
  return CpAt{unicode_class(cp), len == 0 ? (size_t)1 : len};
}
// The rule-dependent predicates. The ONLY difference between the two rules is
// where a combining mark goes: into the letter run (qwen35) or into the
// punctuation run (qwen2). Everything else in the regex is identical.
inline bool is_letter(UnicodeClass c, PreRule r) {
  return c == UnicodeClass::Letter || (r == PreRule::Qwen35 && c == UnicodeClass::Mark);
}
inline bool is_digit(UnicodeClass c) { return c == UnicodeClass::Number; }
inline bool is_space(UnicodeClass c) { return c == UnicodeClass::Space; }
// `[^\s\p{L}\p{N}]` (qwen2) / `[^\s\p{L}\p{M}\p{N}]` (qwen35): everything that
// is not a letter *under this rule*, a number or space -- so a mark is
// punctuation under qwen2 and a letter under qwen35.
inline bool is_punct(UnicodeClass c, PreRule r) {
  return c != UnicodeClass::Number && c != UnicodeClass::Space && !is_letter(c, r);
}

}  // namespace

std::vector<const char*> known_pre_names() {
  std::vector<const char*> names;
  for (const PreName& p : kPreNames) names.push_back(p.name);
  return names;
}

// An absent key used to mean "use the qwen2 rule", which is a guess. What the
// reference does with one, measured 2026-10-08 (llama.cpp 8b4b355): it warns
// "missing pre-tokenizer type, using: 'default'" and runs the DEFAULT pre-type,
// whose BPE `regex_exprs` is a four-expression list (a punctuation run; the
// GPT-2 word rule; `\p{N}+`; `[0-9][0-9][0-9]`) -- a different rule from either
// implemented here. A name it does not know is a hard error there ("unknown
// pre-tokenizer type: '<name>'"). So both an absent key and an unknown name are
// named refusals here, and neither is tokenized with a rule the file did not
// declare.
bool resolve_pre_rule(const std::string& pre, PreRule* out) {
  if (pre.empty()) return false;
  for (const PreName& p : kPreNames) {
    if (pre == p.name) {
      *out = p.rule;
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------- pre-tokenise

// The alternatives of the Qwen2/GPT-4 pre-tokeniser, tried in order; the first
// alternative that matches at a position wins, and each is greedy. Order is
// load-bearing: a leading space is only ever attached by alternative 2 (one
// optional non-letter before letters), which is why "  hi" becomes [" ", " hi"].
std::vector<std::string> Tokenizer::pretokenize(const std::string& t, PreRule rule) {
  std::vector<std::string> out;
  const size_t n = t.size();
  size_t i = 0;
  while (i < n) {
    size_t len = 0;
    const CpAt here = cp_at(t, i);
    const UnicodeClass c = here.cls;

    // 1. contractions: 's 't 're 've 'm 'll 'd  (ASCII, and case-insensitive
    //    in the reference: the character class itself is spelled both cases)
    if (t[i] == '\'') {
      static const char* suffixes[] = {"re", "ve", "ll", "s", "t", "m", "d"};
      for (const char* suf : suffixes) {
        const size_t l = std::strlen(suf);
        if (i + 1 + l <= n) {
          bool ok = true;
          for (size_t k = 0; k < l; ++k) {
            if (std::tolower((unsigned char)t[i + 1 + k]) != suf[k]) { ok = false; break; }
          }
          if (ok) { len = 1 + l; break; }
        }
      }
    }

    // 2. [^\r\n\p{L}\p{N}]? [\p{L}\p{M}]+
    if (len == 0 && is_letter(c, rule)) {
      size_t j = i;
      while (j < n && is_letter(cp_at(t, j).cls, rule)) j += cp_at(t, j).bytes;
      len = j - i;
    }
    if (len == 0 && !is_crlf((uint8_t)t[i]) && !is_letter(c, rule) && !is_digit(c) &&
        i + here.bytes < n) {
      size_t j = i + here.bytes;
      if (is_letter(cp_at(t, j).cls, rule)) {
        while (j < n && is_letter(cp_at(t, j).cls, rule)) j += cp_at(t, j).bytes;
        len = j - i;
      }
    }

    // 3. \p{N} -- one codepoint at a time
    if (len == 0 && is_digit(c)) len = here.bytes;

    // 4. ' '? [^\s\p{L}\p{M}\p{N}]+ [\r\n]*
    if (len == 0) {
      size_t j = i;
      if (t[i] == ' ' && i + 1 < n && is_punct(cp_at(t, i + 1).cls, rule)) j = i + 1;
      if (j < n && is_punct(cp_at(t, j).cls, rule)) {
        while (j < n && is_punct(cp_at(t, j).cls, rule)) j += cp_at(t, j).bytes;
        while (j < n && is_crlf((uint8_t)t[j])) ++j;
        len = j - i;
      }
    }

    // The whitespace alternatives. 5 is tried first and needs a CR/LF run, so
    // the maximal run is measured once and each alternative reads it: 5 ends at
    // the LAST CR/LF run in it (\s* is greedy), 6 is the whole run when it
    // reaches the end of the text (\s+(?!\S)), and 7 -- \s+ -- is the whole run
    // otherwise, but the regex engine backtracks one CHARACTER out of a
    // multi-character run so that a leading space can still be taken by
    // alternative 2 ("  hi" is [" ", " hi"], which is what the reference does).
    //
    // One character is one CODEPOINT, not one byte. Counting bytes made this a
    // rule the reference does not have for every whitespace character wider than
    // ASCII: a single U+00A0 (2 bytes) was cut after its first byte, and a single
    // U+3000 (3 bytes) after its second, so the character's own merge could never
    // fire -- while the rest of the run, the ASCII case, still matched, which is
    // why the hand-written set could not see it. Measured against llama.cpp
    // 2026-10-08 (both with the same vocabulary):
    //   "U+00A0 U+00B4"  engine [126, 254, 28111]  llama [4102, 28111]
    //   "U+3000 U+00B4"  engine [1277, 222, 28111] llama [22441, 28111]
    // and after this change the engine produces 4102 / 22441 as well.
    if (len == 0 && is_space(c)) {
      size_t j = i;      // one past the run, in bytes
      size_t cps = 0;    // codepoints in the run
      size_t last = i;   // start of the run's last codepoint
      while (j < n) {
        const CpAt a = cp_at(t, j);
        if (a.cls != UnicodeClass::Space) break;
        last = j;
        j += a.bytes;
        ++cps;
      }
      size_t crlf_end = 0;
      for (size_t m = i; m < j; ++m) {
        if (!is_crlf((uint8_t)t[m])) continue;
        size_t e = m;
        while (e < j && is_crlf((uint8_t)t[e])) ++e;
        crlf_end = e;
      }
      if (crlf_end) len = crlf_end - i;              // 5. \s*[\r\n]+
      else if (j == n) len = j - i;                  // 6. \s+(?!\S) at end of text
      else if (cps >= 2) len = last - i;             // 6. \s+(?!\S): give back one char
      else len = j - i;                              // 7. \s+: one char, taken whole
    }

    if (len == 0) len = here.bytes;  // never stall: an unclassifiable codepoint is its own chunk
    out.emplace_back(t, i, len);
    i += len;
  }
  return out;
}

// ---------------------------------------------------------------- construction

Tokenizer Tokenizer::from_gguf(const GgufFile& f) {
  Tokenizer tk;
  const std::string model = f.has("tokenizer.ggml.model") ? f.meta_string("tokenizer.ggml.model") : "gpt2";
  if (model != "gpt2")
    fail("tokenizer.ggml.model is '" + model + "'; only gpt2 byte-level BPE is implemented");

  auto it = f.metadata().find("tokenizer.ggml.tokens");
  if (it == f.metadata().end() || !it->second.is_array())
    fail("tokenizer.ggml.tokens missing or not an array");
  const MetaValue& toks = it->second;
  tk.tokens_.reserve(toks.arr.size());
  for (size_t i = 0; i < toks.arr.size(); ++i) {
    tk.tokens_.push_back(toks.arr[i].s);
    tk.token_to_id_.emplace(toks.arr[i].s, (int32_t)i);
  }

  tk.special_.assign(tk.tokens_.size(), 0);
  auto tt = f.metadata().find("tokenizer.ggml.token_type");
  if (tt != f.metadata().end() && tt->second.is_array()) {
    for (size_t i = 0; i < tt->second.arr.size() && i < tk.special_.size(); ++i) {
      const int64_t ty = tt->second.arr[i].i;
      // 3 = CONTROL, 4 = USER_DEFINED: both are matched literally, never merged.
      if (ty == 3 || ty == 4) tk.special_[i] = 1;
    }
  }

  auto mg = f.metadata().find("tokenizer.ggml.merges");
  if (mg != f.metadata().end() && mg->second.is_array()) {
    int32_t rank = 0;
    for (const MetaValue& m : mg->second.arr) {
      // A merge is "left right" with one space. A file that does not use that
      // spelling would make every lookup miss -- which is silent, so it is a
      // refusal here rather than a tokenizer that produces byte ids for text.
      if (m.s.find(' ') == std::string::npos) {
        fail("tokenizer.ggml.merges entry " + std::to_string(rank) +
             " has no space in it (\"" + m.s.substr(0, 24) +
             "\": this is not the 'left right' spelling byte-level BPE uses");
      }
      tk.merge_rank_.emplace(m.s, rank++);
    }
  }

  tk.pre_ = f.has("tokenizer.ggml.pre") ? f.meta_string("tokenizer.ggml.pre") : "";
  // Dispatch on the declared pre-tokeniser. A file whose `pre` this engine does
  // not implement is refused here, by name: silently applying the qwen2 rule to a
  // file that declared something else changes the token ids without changing
  // anything else, which is the failure mode this tokenizer already had twice.
  if (!resolve_pre_rule(tk.pre_, &tk.pre_rule_)) {
    std::string known;
    for (const PreName& p : kPreNames) {
      known += (known.empty() ? "" : ", ");
      known += std::string("'") + p.name + "'";
    }
    fail(tk.pre_.empty()
             ? "the file declares no tokenizer.ggml.pre, and this engine has no "
               "default: llama.cpp warns and falls back to its DEFAULT pre-type, "
               "whose BPE regex list is a four-expression default (punctuation "
               "run, the GPT-2 word rule, \\p{N}+, [0-9][0-9][0-9]) -- a rule "
               "this engine does not implement (implemented: " + known + ")"
             : "tokenizer.ggml.pre is '" + tk.pre_ + "', which this engine does not "
               "implement; it implements " + known + " (the name map in llama.cpp's "
               "llama_vocab::impl::load, src/llama-vocab.cpp, where an unknown "
               "name is an error too). Refused rather than tokenized with a rule "
               "the file did not declare");
  }
  tk.bos_ = f.has("tokenizer.ggml.bos_token_id") ? (int32_t)f.meta_int("tokenizer.ggml.bos_token_id") : -1;
  tk.eos_ = f.has("tokenizer.ggml.eos_token_id") ? (int32_t)f.meta_int("tokenizer.ggml.eos_token_id") : -1;
  tk.add_bos_ = f.has("tokenizer.ggml.add_bos_token") && f.meta_int("tokenizer.ggml.add_bos_token") != 0;
  if (f.has("tokenizer.chat_template")) tk.chat_template_ = f.meta_string("tokenizer.chat_template");
  return tk;
}

bool Tokenizer::is_special(int32_t id) const {
  return id >= 0 && id < (int32_t)special_.size() && special_[id] != 0;
}

const std::string& Tokenizer::token_text(int32_t id) const {
  static const std::string empty;
  if (id < 0 || id >= (int32_t)tokens_.size()) return empty;
  return tokens_[id];
}

// ---------------------------------------------------------------- BPE core

namespace {

// Greedy lowest-rank pair merging, the GPT-2 algorithm. Each step picks the
// globally lowest-ranked adjacent pair, so the result is independent of scan
// direction. Returns the SYMBOLS, not their concatenation: a chunk that does
// not merge down to one symbol is ordinary (that is what a merge table does to
// a long word or a CJK phrase), and each remaining symbol is a token of its own.
//
// Returning the concatenation and looking THAT up was the original bug here: the
// lookup failed for every chunk that did not merge to a single symbol, and the
// failure was absorbed by a per-byte fallback -- so "aaaaaaaa..." (64 a's) came
// out as 64 byte ids where llama.cpp emits 8 merged tokens, and any non-ASCII
// text came out as raw bytes.
std::vector<std::string> bpe_symbols(const std::string& chunk,
                                     const std::unordered_map<std::string, int32_t>& ranks) {
  const ByteMap& bm = byte_map();
  std::vector<std::string> syms;
  for (size_t i = 0; i < chunk.size(); ++i) {
    std::string one;
    append_utf8(one, bm.to_cp[(uint8_t)chunk[i]]);
    syms.push_back(std::move(one));
  }
  if (syms.size() > 1) {
    std::string merged;
    while (syms.size() > 1) {
      int32_t best_rank = INT32_MAX;
      size_t best_i = SIZE_MAX;
      for (size_t i = 0; i + 1 < syms.size(); ++i) {
        // The file stores a merge as its two halves SEPARATED BY A SPACE
        // (tokenizer.ggml.merges holds "Ġ Ġ", "h e", ...). Looking the
        // concatenation up instead -- which is what this did -- matches nothing
        // at all, so no merge ever fired and every chunk was emitted from its
        // whole-vocab lookup or its byte fallback.
        merged = syms[i] + " " + syms[i + 1];
        auto it = ranks.find(merged);
        if (it != ranks.end() && it->second < best_rank) {
          best_rank = it->second;
          best_i = i;
        }
      }
      if (best_i == SIZE_MAX) break;
      syms[best_i] += syms[best_i + 1];
      syms.erase(syms.begin() + best_i + 1);
    }
  }
  return syms;
}

}  // namespace

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool parse_special) const {
  std::vector<int32_t> ids;

  // Split off literal special tokens first. Without this a chat template's
  // "<|im_start|>" would be BPE'd into pieces, which is a silently different
  // prompt rather than an error.
  std::vector<std::pair<size_t, int32_t>> hits;
  if (parse_special) {
    for (size_t i = 0; i < text.size();) {
      size_t best_len = 0, best_at = 0;
      int32_t best_id = -1;
      for (int32_t id = 0; id < (int32_t)tokens_.size(); ++id) {
        if (!special_[id]) continue;
        const std::string& s = tokens_[id];
        if (s.size() > best_len && i + s.size() <= text.size() &&
            text.compare(i, s.size(), s) == 0) {
          best_len = s.size();
          best_at = i;
          best_id = id;
        }
      }
      (void)best_at;
      if (best_id >= 0) {
        hits.emplace_back(i, best_id);
        i += best_len;
      } else {
        ++i;
      }
    }
  }

  size_t seg_start = 0;
  auto flush_text = [&](size_t end) {
    if (end <= seg_start) return;
    const std::string seg = text.substr(seg_start, end - seg_start);
    for (const std::string& chunk : pretokenize(seg, pre_rule_)) {
      for (const std::string& sym : bpe_symbols(chunk, merge_rank_)) {
        auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end()) {
          ids.push_back(it->second);
          continue;
        }
        // A symbol the vocabulary does not contain. This cannot happen for a
        // byte-level BPE: every byte has its own token, and a merge only exists
        // because its result is a token. So it means the file's merge table and
        // its token table contradict each other, which is a refusal -- the old
        // per-byte fallback turned exactly this into silent wrong ids.
        std::string shown;
        for (size_t k = 0; k < sym.size(); ++k) {
          char b[8];
          std::snprintf(b, sizeof(b), "%02X", (unsigned char)sym[k]);
          shown += b;
        }
        fail("merges produced a symbol the vocabulary has no token for (bytes " +
             shown + "): tokenizer.ggml.merges and tokenizer.ggml.tokens "
             "disagree in this file");
      }
    }
  };

  for (const auto& h : hits) {
    flush_text(h.first);
    ids.push_back(h.second);
    seg_start = h.first + tokens_[h.second].size();
  }
  flush_text(text.size());
  return ids;
}

std::string Tokenizer::decode_one(int32_t id) const {
  if (id < 0 || id >= (int32_t)tokens_.size()) return std::string();
  const std::string& s = tokens_[id];
  if (special_[id]) return s;  // control tokens are written as their literal text
  const ByteMap& bm = byte_map();
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    size_t len = 0;
    const uint32_t cp = read_utf8(s, i, &len);
    auto it = bm.to_byte.find(cp);
    if (it != bm.to_byte.end()) out.push_back((char)it->second);
    i += len;
  }
  return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string out;
  for (int32_t id : ids) out += decode_one(id);
  return out;
}

}  // namespace knj
