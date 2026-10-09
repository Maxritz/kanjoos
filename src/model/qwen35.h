// src/model/qwen35.h -- the `qwen35` trunk, front end only.
//
// `qwen35` is the architecture Qwen's own Qwen3.8 checkpoints carry: a hybrid
// trunk of gated-delta-net (SSM) blocks and gated full-attention blocks, hidden
// 5120, ff 17408, vocab 248320. It is *not* `qwen3moe` with different numbers,
// and every file that carries it is refused by the existing forward pass -- see
// docs/10-dflash-draft-models.md section 5 and docs/MISSING-ITEMS.md section 6.
//
// This file implements the part of the trunk that can be *verified* before the
// recurrence exists, and refuses the rest by name:
//
//   verified   geometry read from the file's own metadata
//              every tensor the trunk needs, bound and shape-checked per layer
//              the tokenizer, built from the file, round-tripped
//              embedding rows, decoded through the engine's own dequantizers
//              one layer's input projections (norm -> qkv/gate, or norm -> q/k/v)
//   refused    the SSM recurrence, the gated attention, the dense FFN, the head,
//              and the merged nextn MTP head
//
// The point of the split is that "loads" is not a result: a wrong column stride,
// a transposed weight or a mis-decoded block all load happily. Everything in the
// `verified` list is compared, element by element, against an independent oracle
// (tools/ggufmeta/qwen35_oracle.py) whose reference vectors are written to disk
// with a manifest. See docs/06-profiling.md for the reporting contract.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/loader/dequant.h"
#include "src/loader/gguf.h"
#include "src/tokenizer/tokenizer.h"

namespace knj {

struct Qwen35Geometry {
  int n_layer = 0;
  int n_embd = 0;
  int n_head = 0;
  int n_head_kv = 0;
  int head_dim = 0;
  int n_ff = 0;
  int n_vocab = 0;
  int n_ctx_train = 0;
  int nextn_layers = 0;
  int full_attn_interval = 0;
  int rope_dims = 0;
  // `rope.dimension_sections`: how many cos/sin PAIRS of the rotated dims read
  // each position axis (t, h, w, extra). qwen35 uses ggml's interleaved M-RoPE,
  // so the pairs are not laid out in blocks -- see `imrope_axis()` in the .cpp.
  int rope_sections[4] = {0, 0, 0, 0};
  int ssm_conv_kernel = 0;
  int ssm_state = 0;
  int ssm_groups = 0;
  int ssm_dt_rank = 0;
  int ssm_inner = 0;
  float rms_eps = 1e-6f;
  float rope_base = 1e6f;
  // Per layer: true = recurrent (SSM), false = full attention. Read from
  // `attention.recurrent_layers` when the file carries it, otherwise derived
  // from `full_attention_interval`. Both are the file's own statement, but they
  // are not the same statement: ThinkingCap's file carries the array (48/17) and
  // the GSQ-RCO file does not, so the interval gives it 49/16. Which source was
  // used is reported, because a wrong layer kind binds the wrong tensors.
  std::vector<bool> recurrent;
  bool kinds_from_array = false;
  int ssm_layers() const;
  int attn_layers() const;
  int conv_dim() const { return 2 * ssm_groups * ssm_state + ssm_inner; }
  int q_dim() const { return n_head * head_dim; }
  int kv_dim() const { return n_head_kv * head_dim; }
  std::string describe() const;
};

// One tensor the trunk needs, and whether the file has it with the shape the
// metadata implies. A binding problem is reported, never thrown from the
// reporter, so the operator sees the whole gap rather than the first entry.
struct Qwen35Binding {
  std::string name;
  bool found = false;
  bool shape_ok = false;
  std::string wanted;
  std::string got;
  bool dequantizable = true;
  std::string type_name;
  bool ok() const { return found && shape_ok && dequantizable; }
};

struct Qwen35Report {
  std::vector<Qwen35Binding> bindings;
  int layers_bound = 0;
  int recurrent_layers = 0;
  int attention_layers = 0;
  int nextn_tensors = 0;
  int nextn_block = -1;
  bool all_ok() const;
  std::string summary() const;
};

class Qwen35 {
 public:
  // Throws std::runtime_error with a named reason on a malformed or
  // non-`qwen35` file. Missing *metadata* is fatal here; missing *tensors* are
  // reported through the binding table instead.
  static std::unique_ptr<Qwen35> open(const std::string& path);

  const Qwen35Geometry& geom() const { return g_; }
  const GgufFile& file() const { return file_; }
  const Tokenizer& tokenizer() const { return tok_; }
  bool is_recurrent(int layer) const {
    return layer >= 0 && layer < (int)g_.recurrent.size() && g_.recurrent[layer];
  }

  const std::string& base(int layer) const;  // cached "blk.N."

  // Every tensor the trunk needs, resolved and shape-checked.
  const Qwen35Report& report() const { return rep_; }

  // out[n_embd] = row `token` of token_embd, decoded from the quantization.
  void embed_row(int32_t token, float* out) const;

  // out[width] = x * rsqrt(mean(x^2)+eps) * w, w decoded from any supported type.
  void rmsnorm(const float* x, const TensorInfo& w, float* out, int width) const;

  // y[n_out] = W x for a tensor stored [n_in, n_out] in ggml order.
  void project(const TensorInfo& w, const float* x, float* y, int n_out) const;

  // The pieces of the trunk that are not implemented, each naming the layer
  // count it affects. Empty means the trunk could run -- it cannot, today.
  std::vector<std::string> missing() const;

  // ---- the recurrent (GatedDeltaNet) layer -------------------------------
  //
  // Order and arithmetic are taken from the two reference implementations read
  // on this machine, which agree: transformers' Qwen3NextGatedDeltaNet
  // (torch_recurrent_gated_delta_rule) and llama.cpp's src/models/qwen35.cpp with
  // ggml-cpu's gated_delta_net kernel. Named so the comparison can say WHICH
  // step diverges.
  struct RecurrentOut {
    std::vector<float> x;        // [T, n_embd]      input after attn_norm
    std::vector<float> qkv;      // [T, conv_dim]
    std::vector<float> conv_raw; // [T, conv_dim]
    std::vector<float> conv;     // [T, conv_dim]    after SiLU
    std::vector<float> q;        // [T, num_k_heads * head_dim]  l2-normed, scaled
    std::vector<float> k;        // [T, num_k_heads * head_dim]  l2-normed
    std::vector<float> beta;     // [T, num_v_heads]
    std::vector<float> g;        // [T, num_v_heads]  negative per-token decay
    std::vector<float> o;        // [T, num_v_heads * head_dim]  delta-rule output
    std::vector<float> gated;    // [T, value_dim]    after RMSNormGated
    std::vector<float> out;      // [T, n_embd]       after ssm_out
    std::vector<float> state;    // [num_v_heads, head_dim, head_dim]
    int steps = 0;
  };

  // Runs `n` tokens through layer `layer` (which must be recurrent). The conv
  // history and the recurrent state persist between calls, so a caller can
  // prefill in one call and continue in another -- which is the property the
  // probe tests rather than asserts.
  void recurrent_layer(int layer, const int32_t* tokens, int n, RecurrentOut& out);
  void reset_recurrent_state(int layer);

  // ---- the gated full-attention layer ------------------------------------
  //
  // Order from the two references read on this machine, which agree:
  // llama.cpp's `build_layer_attn` (src/models/qwen35.cpp) and transformers'
  // Qwen3NextAttention. Joint QG projection -> QG split, where the layout is
  // [q_head(head_dim) | gate_head(head_dim)] interleaved per head -> Q RMS norm
  // over head_dim -> K/V projections -> K RMS norm -> partial RoPE (NEOX pairs,
  // rope_dims of head_dim) -> causal attention with GQA -> multiply the
  // attention output elementwise by sigmoid(gate) -> output projection.
  struct AttentionOut {
    std::vector<float> x;        // [T, n_embd]      after attn_norm
    std::vector<float> qg;       // [T, 2 * q_dim]    raw attn_q projection
    std::vector<float> q_norm;   // [T, q_dim]        after the Q RMS norm, pre-rope
    std::vector<float> q_rope;   // [T, q_dim]        after RoPE
    std::vector<float> k_norm;   // [T, kv_dim]       after the K RMS norm, pre-rope
    std::vector<float> k_rope;   // [T, kv_dim]       after RoPE
    std::vector<float> v;        // [T, kv_dim]
    std::vector<float> scores;   // [T, n_head, kv_len]  causal softmax weights
    std::vector<float> attn;     // [T, q_dim]        attention output, pre-gate
    std::vector<float> gated;    // [T, q_dim]        after sigmoid(gate)
    std::vector<float> out;      // [T, n_embd]
    int steps = 0;
    int kv_len = 0;              // keys in the cache after the call
    int rope_pairs[4] = {0, 0, 0, 0};  // rotated pairs per IMROPE axis
  };

  // Runs `n` tokens through layer `layer` (which must be a full-attention
  // layer). The K/V cache persists between calls, so this is prefill followed by
  // decode -- the same property the recurrent path tests.
  void attention_layer(int layer, const int32_t* tokens, int n, AttentionOut& out);
  void reset_attention_state(int layer);

  // What `exp(g)` was over the last run: the recurrence is stable iff every
  // value is in (0, 1]. Printed by the probe, because a wrong reading of the
  // stored A would show up here before it shows up in a token.
  double last_decay_min() const { return decay_min_; }
  double last_decay_max() const { return decay_max_; }

  // ---- the trunk ----------------------------------------------------------
  //
  // The causal stack, composed exactly the way llama.cpp's qwen35 graph composes
  // it (`src/models/qwen35.cpp`, read on this machine):
  //
  //   for il in [0, n_layer):
  //     inp = cur
  //     cur = rmsnorm(cur, attn_norm[il])        // the block norms its own input
  //     cur = is_recr(il) ? linear_attention(cur) : full_attention(cur)
  //     cur = cur + inp                          // "attn_residual"
  //     res = cur
  //     cur = ffn(rmsnorm(cur, post_attention_norm[il]))   // dense SwiGLU, no MoE
  //     cur = cur + res                          // "post_ffn"
  //   logits = output.weight * rmsnorm(cur, output_norm)
  //
  // `n_layer` is `block_count - nextn_predict_layers` = 64 here: llama.cpp notes
  // that MTP/NextN blocks are "loaded as extra decoder blocks but not executed in
  // the main pass", so blk.64 is the MTP block and the trunk is blk.0..blk.63.
  // There is no logit scaling and no softcap on this head.
  struct TrunkOut {
    std::vector<float> hidden;     // [T, n_embd]  after the last trunk layer
    std::vector<float> normed;     // [T, n_embd]  after output_norm (the head's input)
    std::vector<float> logits;     // [T, n_vocab] one row per token
    // [trunk_layers * T * n_embd], the residual stream after each layer, when the
    // caller asks for it. The layer-by-layer comparison is what localises a defect
    // to one block instead of reporting "the logits are wrong".
    std::vector<float> per_layer;
    // The same shape, holding the stream after each layer's MIXER residual and
    // before its FFN. Two points per layer, because a wrong FFN and a wrong block
    // are different defects and this is what tells them apart.
    std::vector<float> per_layer_mix;
    int trunk_layers = 0;
    int steps = 0;
  };
  int trunk_layers() const { return g_.n_layer - g_.nextn_layers; }
  void trunk_forward(const int32_t* tokens, int n, TrunkOut& out, bool keep_per_layer);
  // Drops every layer's carried state (conv history, delta-rule state, K/V).
  void reset_trunk_state();

 private:
  // Per recurrent layer: the conv's last (kernel-1) inputs and the delta-rule
  // state, both sized at open() so no allocation happens per token.
  struct RecurrentState {
    std::vector<float> conv_hist;  // [(ker-1), conv_dim]
    std::vector<float> state;      // [num_v_heads, head_dim, head_dim]
  };

  // Per attention layer: the roped keys and the values of every position seen so
  // far, in position order, plus the count. Sized lazily; the probe's scale.
  struct AttentionState {
    std::vector<float> k;  // [len, n_head_kv * head_dim]
    std::vector<float> v;  // [len, n_head_kv * head_dim]
    int len = 0;
  };

 public:

 private:
  Qwen35() = default;
  void read_geometry();
  void bind_tensors();

  // The blocks themselves, taking a [n, n_embd] hidden state: the probe embeds
  // tokens and calls these, and so does every trunk step. `recurrent_layer` and
  // `attention_layer` are the embedding wrappers the per-layer probes use.
  void recurrent_block(int layer, const float* hidden, int n, RecurrentOut& out);
  void attention_block(int layer, const float* hidden, int n, AttentionOut& out);
  // Dense SwiGLU FFN: down(silu(gate(x)) * up(x)). No router -- llama.cpp's
  // qwen35 asserts `ffn_gate_inp == nullptr` ("Qwen3.5 does not use MoE FFN").
  void dense_ffn(int layer, const float* x, int n, float* out);

  GgufFile file_;
  Tokenizer tok_;
  Qwen35Geometry g_;
  Qwen35Report rep_;
  std::vector<std::string> base_;
  std::vector<float> scratch_;
  std::map<int, RecurrentState> rst_;   // one per recurrent layer touched
  std::map<int, AttentionState> ast_;   // one per attention layer touched
  double decay_min_ = 0.0;
  double decay_max_ = 0.0;
};

// The comparison of one engine vector against one oracle vector.
struct VecCheck {
  std::string name;
  size_t n = 0;
  double max_abs = 0.0;
  double rel_rmse = 0.0;
  // Exactly-zero engine values. Informational: a Q4_K-decoded embedding row
  // legitimately contains exact zeros (d * 0 - dmin * 0), so a zero is only a
  // defect when the *reference* is not zero there -- that is `unwritten`.
  uint64_t zeros = 0;
  uint64_t unwritten = 0;   // engine 0 where the reference is not 0
  uint64_t mismatches = 0;  // elements outside the tolerance
  int64_t first_bad = -1;
  double ref_max_abs = 0.0;
  // True when the reference file is absent or shorter than the engine vector.
  // That is NOT a failed comparison -- nothing was compared -- so it is reported
  // as NO REF rather than as a FAIL with perfect-looking numbers.
  bool ref_missing = false;
  bool verdict = false;
};

// Compares engine values against `dir/<name>.f32`, element by element.
// rtol is relative to the reference vector's RMS.
VecCheck compare_vector(const std::string& dir, const std::string& name,
                        const std::vector<float>& got, double rtol);

// Runs the whole probe and prints it. Returns the process exit code: 3 for
// "front end verified, trunk refused" -- never 0, because nothing here produces
// logits yet.
struct Qwen35ProbeOptions {
  std::string model;
  std::string ref_dir;       // empty = no oracle comparison
  std::string text = "The capital of France is";
  // Explicit token ids, comma separated. The oracle is invoked with `--tokens`,
  // so a comparison needs the ENGINE to run the same ids; without this the probe
  // tokenises `text` and the two sides silently disagree about what was computed.
  // Empty means "tokenise text", which is what the per-layer probes have always
  // done.
  std::string tokens;
  int layer = 0;
  double rtol = 2e-3;
  // Run one recurrent layer end to end (conv1d carry, the per-group delta rule,
  // the alpha/beta/ssm_a gating) and compare the 12 `layer_*` vectors. Without
  // it the probe verifies only the layer's input projections.
  bool recurrent = false;
  // Run one full-attention layer end to end (QG split, QK-norm, partial RoPE,
  // causal attention, the gate) and compare the 11 `layer_*` vectors.
  bool attention = false;
  // Run the whole trunk: every causal layer in order with its residuals, the dense
  // FFN, output_norm and the untied head, and compare the per-layer residual stream,
  // the head's input and the logits. This path produces logits, so a verified run
  // exits 0 rather than 3.
  bool forward = false;
  bool quiet = false;
};
int run_qwen35_probe(const Qwen35ProbeOptions& opt);

}  // namespace knj
