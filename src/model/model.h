// src/model/model.h -- Qwen3-MoE forward pass over the GGUF's own bytes.
//
// Two design rules drive this file, and both come from the engine contract
// rather than from convenience:
//
//  1. Geometry is read from the file's metadata, never hardcoded. A model whose
//     block count or expert count disagrees with what the engine assumed must
//     fail at load, not compute quietly with the wrong strides.
//  2. Weights stay quantized in the mapping and are decoded *inside* the inner
//     product. A dequantized 1.5 B model is ~6 GB; the streaming target is far
//     larger, so materialising f32 weights is not a shortcut this code can
//     take even once and then be trusted later.
//
// The forward pass is batched from the start: `forward(tokens, n)` runs n
// tokens through one set of weight reads. A prefill of 512 tokens therefore
// touches the model bytes once rather than 512 times, which is the difference
// between a prefill that is worth measuring and one that is not. Decode is the
// same code with n == 1.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/loader/gguf.h"
#include "src/tokenizer/tokenizer.h"
#include "src/util/parallel.h"

namespace knj {

struct ModelGeometry {
  int n_layer = 0;
  int n_embd = 0;
  int n_head = 0;
  int n_head_kv = 0;
  int head_dim = 0;
  int n_ff_exp = 0;          // per-expert intermediate width
  int n_expert = 0;
  int n_expert_used = 0;
  int n_vocab = 0;
  int n_ctx_train = 0;
  float rms_eps = 1e-6f;
  float rope_base = 1e6f;
  // ggml's two rotary conventions. NEOX rotates the pair (x[i], x[i + d/2]);
  // NORMAL rotates (x[2i], x[2i+1]). They are not interchangeable: picking the
  // wrong one leaves the model fluent-looking and wrong. See model.cpp.
  bool rope_split_halves = true;

  int q_dim() const { return n_head * head_dim; }
  int kv_dim() const { return n_head_kv * head_dim; }
  std::string describe() const;
};

class Model {
 public:
  struct Options {
    int n_ctx = 2048;
    int threads = 0;  // 0 = one per hardware thread
  };

  static std::unique_ptr<Model> open(const std::string& path, const Options& opt);

  const ModelGeometry& geom() const { return g_; }
  const Tokenizer& tokenizer() const { return tok_; }
  const GgufFile& file() const { return file_; }
  int n_ctx() const { return n_ctx_; }
  int pos() const { return pos_; }
  // Threads a parallel region actually uses, including the calling thread.
  // 0 in Options means "one per hardware thread", so printing the option
  // would report 0 for the common case -- not the width that ran.
  int threads() const { return pool_ ? pool_->total() : 1; }

  // Drops the KV cache and returns to position 0.
  void reset();

  // Runs `n` tokens starting at the current position. Returns a pointer to
  // n_vocab logits for the *last* token in the batch. The pointer stays valid
  // until the next call.
  const float* forward(const int32_t* tokens, int n);

 private:
  Model() = default;

  struct Layer {
    const TensorInfo* attn_norm = nullptr;
    const TensorInfo* ffn_norm = nullptr;
    const TensorInfo* attn_q = nullptr;
    const TensorInfo* attn_k = nullptr;
    const TensorInfo* attn_v = nullptr;
    const TensorInfo* attn_o = nullptr;
    const TensorInfo* q_norm = nullptr;
    const TensorInfo* k_norm = nullptr;
    const TensorInfo* gate_inp = nullptr;
    const TensorInfo* gate_exps = nullptr;
    const TensorInfo* up_exps = nullptr;
    const TensorInfo* down_exps = nullptr;
  };

  void read_geometry();
  void bind_tensors();
  void check_shape(const TensorInfo& t, std::initializer_list<int64_t> want) const;

  // y[0..m) = W . x, where W has shape [n_in, n_out] in ggml order and x holds
  // m rows of n_in floats. `ldy` is n_out for a fresh destination. Threads
  // split the output columns, so every weight byte is read exactly once.
  void matmul(const TensorInfo& w, const float* x, int m, int ldx, float* y, int ldy);
  // Same, but on expert `e` of a 3-D [n_in, n_out, n_expert] tensor.
  void matmul_expert(const TensorInfo& w, int e, const float* x, int m, int ldx,
                     float* y, int ldy);
  void matmul_plain(const TensorInfo& w, const float* x, int m, int ldx, float* y,
                    int ldy);

  // out[i] = x[i] * w[i] * rsqrt(mean(x^2) + eps), one row per call.
  void rmsnorm(const float* x, const TensorInfo& w, float* out, int width);
  void rmsnorm_rows(const float* x, int m, int width, const TensorInfo& w, float* out);
  // RoPE in place over `n_head` consecutive heads of `d` elements each.
  void rope(float* v, int n_head, int pos);
  void softmax(float* v, int n);

  void moe(const Layer& w, const float* hx, int m, float* out);

  GgufFile file_;
  Tokenizer tok_;
  std::unique_ptr<ThreadPool> pool_;
  ModelGeometry g_;
  int n_ctx_ = 0;
  int pos_ = 0;

  std::vector<Layer> layers_;
  const TensorInfo* tok_embd_ = nullptr;
  const TensorInfo* out_norm_ = nullptr;
  const TensorInfo* out_w_ = nullptr;

  // Reused scratch, sized once at open. Capacity is n_ctx_ rows wherever the
  // first dimension is a token count.
  std::vector<float> x_, xn_, q_, k_, v_, ctx_, ao_, hx_;
  std::vector<float> rlog_, y_gate_, y_up_, y_down_, gat_;
  // The expert branch's own output. It is deliberately *not* x_: moe() starts
  // by zeroing its destination, so writing the result into x_ would wipe the
  // residual the branch has to be added back to.
  std::vector<float> ffn_out_;
  std::vector<float> scores_, logits_, inv_freq_;
  std::vector<float> kv_k_, kv_v_;
  std::vector<std::vector<int32_t>> ex_tokens_;
  std::vector<std::vector<float>> ex_weight_;

  // Scratch that used to be allocated per call. Each of these was reached many
  // times per token -- head_out_ once per head per layer, probs_/chosen_ once
  // per layer, norm_stage_/norm_w_/col_tmp_ once per layer -- so the allocator
  // was being handed a few hundred tiny requests per token for no reason.
  // Sizes are fixed at open() and no two of them are live at once, so reuse is
  // safe: rmsnorm() is only ever called from the forward pass, never from
  // inside a parallel matmul body.
  std::vector<float> head_out_;    // per-head norm output, head_dim floats
  std::vector<float> norm_stage_;  // rmsnorm staging for a non-F32 weight, n_embd
  std::vector<float> norm_w_;      // rmsnorm_rows' per-layer weight decode, n_embd
  std::vector<float> col_tmp_;     // matmul_plain's per-column decode, n_embd
  std::vector<float> probs_;       // router probabilities, n_expert
  std::vector<int> chosen_;        // router top-k picks, n_expert_used
};

}  // namespace knj
