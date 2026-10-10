// src/tokenizer/tokenizer.h -- byte-level BPE, built from the GGUF itself.
//
// The vocabulary is read from the file, never from a compiled-in table: a
// quantised model's tokeniser is part of its fingerprint (I5), and a model that
// ships a different merges list must tokenise the way its trainer intended.
//
// The "gpt2" model here means GPT-2 byte-level BPE: text is first mapped
// byte -> printable codepoint, and merges are learned over that alphabet. The
// `pre` field ("qwen2") selects the pre-tokeniser regex which decides where a
// merge is allowed to cross. Getting that regex wrong does not crash -- it
// silently produces different token ids -- so it is implemented explicitly
// rather than approximated, and cross-checked against a reference.
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/loader/gguf.h"

namespace knj {

// The pre-tokenisers this engine implements. They are NOT interchangeable: the
// reference regexes differ in whether a combining mark joins a letter run
// (`qwen35`, `[\p{L}\p{M}]+`) or is punctuation (`qwen2`, `[\p{L}]+`), which moves
// a chunk boundary and therefore changes which merges are possible. Until this
// was dispatched on, every file got the qwen2 answer, including files that
// declare `qwen35`.
//
// Source for the mapping and for both regexes: llama.cpp `src/llama-vocab.cpp`
// -- the `tokenizer_pre == "..."` name map in `llama_vocab::impl::load`, and the
// regex table's QWEN2/QWEN35 cases in the `llm_tokenizer_bpe` constructor --
// read on this machine. tools/tok_pre_rules.py re-derives the name table from
// that source rather than from a comment, and pins it in
// tests/fixtures/tok/pre_rules.expected.
enum class PreRule {
  Qwen2,   // `\p{L}+`, marks are punctuation
  Qwen35,  // `[\p{L}\p{M}]+`, marks join the letter run
};

// The rule's name, for printing (I5: the tokeniser's identity belongs in the
// record).
const char* pre_rule_name(PreRule r);

// The `pre` names this engine can honour, in table order. Exposed so the fixture
// test derives its file names from THIS list rather than from a copy: a name
// added here without a fixture fails the test by name, instead of going silently
// unverified until a real file declares it.
std::vector<const char*> known_pre_names();

// A tiny open-addressing map: view keys, linear probing, power-of-two table,
// load factor <= 0.5, FIRST value wins on a duplicate key.
//
// This exists because of a measurement, not a preference. Building the two
// tokenizer maps (151,936 vocab entries + 151,387 merges) in
// `std::unordered_map` with `reserve` costs ~52 ms -- ~170 ns per entry, which
// is a per-entry node allocation plus a re-hash -- while the same inserts into
// this table cost ~9 ms, and FIND is ~1.9x faster too (48.9 vs 94.1 ns/lookup,
// measured on the reference model). `tmp/bench_tokmap.cpp` reproduces both
// numbers and verifies this map returns the SAME id for all 151,936 vocab
// entries; the fuzz cross-check gates it against llama.cpp.
//
// First-wins is not incidental: a file with duplicate token texts must keep the
// same id `std::unordered_map::emplace` kept (the first one), or every
// downstream id would shift.
//
// Keys are VIEWS: the bytes live in storage the Tokenizer owns (`tokens_` for
// the vocabulary, `merge_arena_` for the merge table) and both are fully built
// before any key is handed out, so no view can be invalidated by a later
// append. Both owners move by stealing their heap buffer, so the views survive
// a Tokenizer move; copying a Tokenizer is deleted for the same reason.
class TokViewMap {
 public:
  void reserve(size_t n) {
    size_t cap = 8;
    while (cap < n * 2) cap <<= 1;
    slots_.assign(cap, Slot());
    size_ = 0;
  }

  // First-wins, exactly like `unordered_map::emplace`. False when the key was
  // already present (the caller's rank/id is then dropped, as before).
  bool emplace(std::string_view k, int32_t v) {
    if (slots_.empty()) reserve(8);
    // Slot lengths are 32-bit (that is what keeps a slot 16 bytes). A longer
    // key would TRUNCATE and then match some other key's prefix -- a silent
    // wrong lookup, which is the failure this loader refuses everywhere else.
    // No real vocabulary has a 4 GiB token; a file that declares one is
    // refused instead of mis-tokenised.
    if (k.size() > 0xFFFFFFFEull) {
      throw std::runtime_error(
          "tokenizer: a token or merge key exceeds 4 GiB and cannot be indexed");
    }
    const size_t mask = slots_.size() - 1;
    size_t i = hash(k) & mask;
    while (slots_[i].ptr != nullptr) {
      if (slots_[i].len == k.size() &&
          std::memcmp(slots_[i].ptr, k.data(), k.size()) == 0) {
        return false;
      }
      i = (i + 1) & mask;
    }
    // std::string::data() is never null (C++11), and an arena substring always
    // points inside the arena, so nullptr stays a safe empty marker.
    slots_[i].ptr = k.data();
    slots_[i].len = static_cast<uint32_t>(k.size());
    slots_[i].val = v;
    ++size_;
    return true;
  }

  const int32_t* find(std::string_view k) const {
    if (slots_.empty()) return nullptr;
    const size_t mask = slots_.size() - 1;
    size_t i = hash(k) & mask;
    while (slots_[i].ptr != nullptr) {
      if (slots_[i].len == k.size() &&
          std::memcmp(slots_[i].ptr, k.data(), k.size()) == 0) {
        return &slots_[i].val;
      }
      i = (i + 1) & mask;
    }
    return nullptr;
  }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

 private:
  // 16 bytes per slot: a pointer, a 32-bit length and the value pack exactly,
  // and nullptr doubles as the empty marker -- no separate occupancy byte to
  // fetch on a probe.
  struct Slot {
    const char* ptr = nullptr;
    uint32_t len = 0;
    int32_t val = 0;
  };

  // SplitMix-style finaliser over the first and last 8 bytes. Keys here average
  // ~6 bytes, so the short path is one memcpy and one mix; long keys still mix
  // both ends, which is what catches the common "same prefix, different suffix"
  // shape of token strings.
  static uint64_t hash(std::string_view s) {
    const size_t n = s.size();
    uint64_t x = 0x9e3779b97f4a7c15ULL * (n + 1);
    if (n <= 8) {
      uint64_t v = 0;
      if (n) std::memcpy(&v, s.data(), n);
      x ^= v;
    } else {
      uint64_t a = 0, b = 0;
      std::memcpy(&a, s.data(), 8);
      std::memcpy(&b, s.data() + n - 8, 8);
      x ^= a ^ (b * 0xc2b2ae3d27d4eb4fULL);
    }
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
  }

  std::vector<Slot> slots_;
  size_t size_ = 0;
};

// Resolves a `pre` value to the rule it selects. False for an absent or empty
// key, and false for a name this engine does not implement. Both are refusals
// (`from_gguf` says which), because both would otherwise be silently tokenised
// with a rule the file did not declare.
bool resolve_pre_rule(const std::string& pre, PreRule* out);

class Tokenizer {
 public:
  Tokenizer() = default;
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  Tokenizer(Tokenizer&&) = default;
  Tokenizer& operator=(Tokenizer&&) = default;

  // Reads tokenizer.ggml.tokens / .merges / .token_type and the special ids.
  static Tokenizer from_gguf(const GgufFile& f);

  // Text -> token ids. When `parse_special` is true, control tokens appearing
  // literally in the text (e.g. "<|im_start|>") are consumed as single ids
  // instead of being tokenised as text.
  std::vector<int32_t> encode(const std::string& text, bool parse_special = false) const;

  // Token id -> raw text bytes. Ids outside the vocabulary are skipped rather
  // than substituted, so a corrupt id shows up as missing text, not as noise.
  std::string decode(const std::vector<int32_t>& ids) const;
  std::string decode_one(int32_t id) const;

  // Structural pieces the engine needs for chat formatting.
  const std::string& token_text(int32_t id) const;
  int32_t vocab_size() const { return static_cast<int32_t>(tokens_.size()); }
  int32_t bos_id() const { return bos_; }
  int32_t eos_id() const { return eos_; }
  bool add_bos() const { return add_bos_; }
  const std::string& chat_template() const { return chat_template_; }

  // How many merges the file carries, and its declared pre-tokeniser rule. Both
  // are part of the tokeniser's identity (I5) and both are printed by the probe:
  // a file with an empty merge table, or a `pre` this engine does not implement,
  // tokenises differently without failing.
  int32_t merges() const { return static_cast<int32_t>(merge_rank_.size()); }
  const std::string& pre() const { return pre_; }
  PreRule pre_rule() const { return pre_rule_; }

  // True when the id is a CONTROL / USER_DEFINED token (never produced by BPE).
  bool is_special(int32_t id) const;

 private:
  // Pre-tokeniser: split `text` into the chunks the merge table may operate on,
  // under the rule the file declared.
  static std::vector<std::string> pretokenize(const std::string& text, PreRule rule);

  // The two maps hold VIEWS, not copies, in an open-addressing table (see
  // TokViewMap for the measurement that chose it). Every key's bytes live in
  // storage this object owns -- `tokens_` for the vocabulary, `merge_arena_`
  // for the merge table -- so building a 152K-vocab tokenizer does not allocate
  // and copy 304K separate key strings on top of the copies it already has
  // (measured as 55 of 70 ms of the tokenize-only path before this). Both
  // owners are declared BEFORE the map that views them and are never mutated
  // after `from_gguf` returns, so the views survive every legal Tokenizer
  // operation, including the move that returns one.
  //
  // Copying is deleted because a copy would duplicate the storage while the
  // new maps' views kept pointing at the OLD storage: dangling keys that fail
  // lookups silently. Move-only makes that bug unrepresentable.
  std::vector<std::string> tokens_;                        // id -> encoded text
  TokViewMap token_to_id_;
  std::string merge_arena_;                                // all "left right" keys
  TokViewMap merge_rank_;                                  // key -> rank
  std::string pre_;                                        // tokenizer.ggml.pre
  PreRule pre_rule_ = PreRule::Qwen2;                      // resolved from pre_
  std::vector<uint8_t> special_;                           // id -> 0/1
  int32_t bos_ = -1;
  int32_t eos_ = -1;
  bool add_bos_ = false;
  std::string chat_template_;
};

}  // namespace knj
