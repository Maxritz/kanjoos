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
#include <string>
#include <unordered_map>
#include <vector>

#include "src/loader/gguf.h"

namespace knj {

class Tokenizer {
 public:
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

  // True when the id is a CONTROL / USER_DEFINED token (never produced by BPE).
  bool is_special(int32_t id) const;

 private:
  // Pre-tokeniser: split `text` into the chunks the merge table may operate on.
  static std::vector<std::string> pretokenize(const std::string& text);

  std::vector<std::string> tokens_;                        // id -> encoded text
  std::unordered_map<std::string, int32_t> token_to_id_;
  std::unordered_map<std::string, int32_t> merge_rank_;    // "a b" -> rank
  std::vector<uint8_t> special_;                           // id -> 0/1
  int32_t bos_ = -1;
  int32_t eos_ = -1;
  bool add_bos_ = false;
  std::string chat_template_;
};

}  // namespace knj
