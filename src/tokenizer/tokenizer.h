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

// Resolves a `pre` value to the rule it selects. False for an absent or empty
// key, and false for a name this engine does not implement. Both are refusals
// (`from_gguf` says which), because both would otherwise be silently tokenised
// with a rule the file did not declare.
bool resolve_pre_rule(const std::string& pre, PreRule* out);

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

  std::vector<std::string> tokens_;                        // id -> encoded text
  std::unordered_map<std::string, int32_t> token_to_id_;
  std::unordered_map<std::string, int32_t> merge_rank_;    // "left right" -> rank
  std::string pre_;                                        // tokenizer.ggml.pre
  PreRule pre_rule_ = PreRule::Qwen2;                      // resolved from pre_
  std::vector<uint8_t> special_;                           // id -> 0/1
  int32_t bos_ = -1;
  int32_t eos_ = -1;
  bool add_bos_ = false;
  std::string chat_template_;
};

}  // namespace knj
