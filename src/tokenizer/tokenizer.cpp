// src/tokenizer/tokenizer.cpp -- byte-level BPE, built from the GGUF itself.
#include "src/tokenizer/tokenizer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace knj {
namespace {

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("tokenizer: " + m); }

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
// The reference regex is Unicode (\p{L}, \p{N}). This implementation is exact
// for ASCII, which is what the coherence gate exercises, and conservative
// elsewhere: every non-ASCII byte is treated as a letter so multi-byte
// sequences stay inside one "word" rather than being split per byte.
inline bool is_letter(uint8_t c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
}
inline bool is_digit(uint8_t c) { return c >= '0' && c <= '9'; }
inline bool is_space(uint8_t c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}
inline bool is_crlf(uint8_t c) { return c == '\r' || c == '\n'; }
inline bool is_punct(uint8_t c) {
  return !is_space(c) && !is_letter(c) && !is_digit(c);
}

}  // namespace

// ---------------------------------------------------------------- pre-tokenise

// The alternatives of the Qwen2/GPT-4 pre-tokeniser, tried in order; the first
// alternative that matches at a position wins, and each is greedy. Order is
// load-bearing: a leading space is only ever attached by alternative 2 (one
// optional non-letter before letters), which is why "  hi" becomes [" ", " hi"].
std::vector<std::string> Tokenizer::pretokenize(const std::string& t) {
  std::vector<std::string> out;
  const size_t n = t.size();
  const uint8_t* s = reinterpret_cast<const uint8_t*>(t.data());
  size_t i = 0;
  while (i < n) {
    size_t len = 0;

    // 1. contractions: 's 't 're 've 'm 'll 'd
    if (s[i] == '\'') {
      static const char* suffixes[] = {"re", "ve", "ll", "s", "t", "m", "d"};
      for (const char* suf : suffixes) {
        const size_t l = std::strlen(suf);
        if (i + 1 + l <= n) {
          bool ok = true;
          for (size_t k = 0; k < l; ++k) {
            if (std::tolower((unsigned char)s[i + 1 + k]) != suf[k]) { ok = false; break; }
          }
          if (ok) { len = 1 + l; break; }
        }
      }
    }

    // 2. [^\r\n\p{L}\p{N}]? \p{L}+
    if (len == 0 && is_letter(s[i])) {
      size_t j = i;
      while (j < n && is_letter(s[j])) ++j;
      len = j - i;
    }
    if (len == 0 && i + 1 < n && !is_crlf(s[i]) && !is_letter(s[i]) && !is_digit(s[i]) &&
        is_letter(s[i + 1])) {
      size_t j = i + 1;
      while (j < n && is_letter(s[j])) ++j;
      len = j - i;
    }

    // 3. \p{N} -- one digit at a time
    if (len == 0 && is_digit(s[i])) len = 1;

    // 4. ' '? [^\s\p{L}\p{N}]+ [\r\n]*
    if (len == 0) {
      size_t j = i;
      if (s[j] == ' ' && j + 1 < n && is_punct(s[j + 1])) ++j;
      if (j < n && is_punct(s[j])) {
        while (j < n && is_punct(s[j])) ++j;
        while (j < n && is_crlf(s[j])) ++j;
        len = j - i;
      }
    }

    // 5. \s* [\r\n]+
    if (len == 0 && is_space(s[i])) {
      size_t j = i;
      while (j < n && is_space(s[j])) ++j;
      if (j > i && is_crlf(s[j - 1])) len = j - i;
      else if (j - i >= 2) {
        // whitespace run that does not end in CR/LF: \s* gives back one char
        size_t k = i;
        while (k < n && is_space(s[k])) ++k;
        (void)k;
        // find the last position where a CR/LF run can end the match
        size_t best = 0;
        for (size_t m = i + 1; m < j; ++m) {
          if (is_crlf(s[m - 1])) best = m;  // \s* consumed [i,m) ending in CR/LF
        }
        // and the run must be maximal from there
        if (best) {
          size_t m = best;
          while (m < n && is_crlf(s[m])) ++m;
          len = m - i;
        }
      }
    }

    // 6. \s+ (?!\S)
    if (len == 0 && is_space(s[i])) {
      size_t j = i;
      while (j < n && is_space(s[j])) ++j;
      if (j == n) len = j - i;               // whole run, then end of text
      else if (j - i >= 2) len = (j - i) - 1;  // back off one space
    }

    // 7. \s+
    if (len == 0 && is_space(s[i])) {
      size_t j = i;
      while (j < n && is_space(s[j])) ++j;
      len = j - i;
    }

    if (len == 0) len = 1;  // never stall; a byte we cannot classify is its own chunk
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
    for (const MetaValue& m : mg->second.arr) tk.merge_rank_.emplace(m.s, rank++);
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
// direction.
std::string bpe(const std::string& chunk,
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
        merged = syms[i] + syms[i + 1];
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
  std::string out;
  for (const std::string& s : syms) out += s;
  return out;
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
    for (const std::string& chunk : pretokenize(seg)) {
      const std::string enc = bpe(chunk, merge_rank_);
      auto it = token_to_id_.find(enc);
      if (it != token_to_id_.end()) {
        ids.push_back(it->second);
      } else {
        // No single id for the merged chunk: emit its individual characters,
        // which the vocabulary always contains for byte-level BPE.
        const ByteMap& bm = byte_map();
        for (size_t k = 0; k < chunk.size(); ++k) {
          std::string one;
          append_utf8(one, bm.to_cp[(uint8_t)chunk[k]]);
          auto it1 = token_to_id_.find(one);
          if (it1 != token_to_id_.end()) ids.push_back(it1->second);
        }
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
