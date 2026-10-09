// src/model/model.cpp -- Qwen3-MoE forward pass.
//
// The architecture is transcribed from llama.cpp's src/models/qwen3moe.cpp, and
// the intent is to match it exactly, including the parts that look arbitrary:
//
//   * rope_type for QWEN3MOE is NEOX, i.e. ggml rotates the pair
//     (x[i], x[i + head_dim/2]) -- HF's rotate_half. The other ggml convention
//     (NORMAL, adjacent pairs) produces a model that still emits plausible
//     tokens and is wrong everywhere, so it is not a detail to guess at.
//   * q_norm / k_norm are RMSNorm over one head's 128 values *before* RoPE.
//   * attention scale is 1/sqrt(head_dim), there is no softcap.
//   * the router softmaxes over all experts, then the selected top-k weights
//     are renormalised to sum 1 (norm_w = true). That equals softmaxing over
//     only the selected logits, which is what the code below does.
//   * the expert branch is down(silu(gate(x)) * up(x)); Qwen3-MoE has no shared
//     expert.
#include "src/model/model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "src/loader/dequant.h"
#include "src/profiler/profiler.h"

namespace knj {
namespace {

// Blocks are at most 256 weights (the K-quants), and dequant_dot_f32 relies on
// that. Fail loudly rather than overflow a stack buffer if ggml ever adds one.
constexpr int kMaxBlockWeights = 256;

// C21 scope names are "class" or "class:scope". The profiler groups by class
// for --profile-detail=class and keeps the whole string for =layer, so one
// instrumentation pass serves both. Copying the pattern here keeps the model
// free of any knowledge of the profiler's grouping rules.
inline void profile_name(char* buf, size_t n, const char* cls, int layer) {
  if (layer < 0) std::snprintf(buf, n, "%s", cls);
  else std::snprintf(buf, n, "%s:%d", cls, layer);
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// KNJ_DUMP_HIDDEN=<dir> writes named f32 intermediates of the forward pass.
//
// This exists so the forward pass can be *bisected* against an independent
// implementation (tools/ref_qwen3moe.py, which is itself checked against
// llama.cpp) rather than argued about by reading. A disagreement then names the
// tensor where the two implementations part company, which is a fact; an
// opinion about which line looks wrong is not.
void dump_f32(const char* name, const float* p, size_t n) {
  const char* dir = std::getenv("KNJ_DUMP_HIDDEN");
  if (!dir || !*dir) return;
  const std::string path = std::string(dir) + "/" + name + ".f32";
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return;
  std::fwrite(p, sizeof(float), n, f);
  std::fclose(f);
}

}  // namespace

std::string ModelGeometry::describe() const {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "layers=%d hidden=%d heads=%d/%d head_dim=%d q_dim=%d kv_dim=%d\n"
                "  experts=%d top_k=%d expert_ff=%d vocab=%d ctx_train=%d\n"
                "  rms_eps=%.3g rope_base=%.6g rope=%s",
                n_layer, n_embd, n_head, n_head_kv, head_dim, q_dim(), kv_dim(),
                n_expert, n_expert_used, n_ff_exp, n_vocab, n_ctx_train, rms_eps,
                rope_base, rope_split_halves ? "split-halves (ggml NEOX)" : "adjacent pairs (ggml NORMAL)");
  return std::string(buf);
}

void Model::check_shape(const TensorInfo& t, std::initializer_list<int64_t> want) const {
  if (t.dims.size() != want.size()) {
    throw std::runtime_error("model: tensor '" + t.name + "' has " +
                             std::to_string(t.dims.size()) + " dims, expected " +
                             std::to_string(want.size()));
  }
  size_t i = 0;
  for (int64_t d : want) {
    if (t.dims[i] != (uint64_t)d) {
      throw std::runtime_error("model: tensor '" + t.name + "' dim " + std::to_string(i) +
                               " is " + std::to_string(t.dims[i]) + ", expected " +
                               std::to_string(d));
    }
    ++i;
  }
}

void Model::read_geometry() {
  const std::string arch = file_.meta_string("general.architecture");
  if (arch != "qwen3moe") {
    throw std::runtime_error("model: architecture is '" + arch +
                             "', this forward pass only implements 'qwen3moe'");
  }
  const std::string p = arch + ".";
  g_.n_layer = (int)file_.meta_int(p + "block_count");
  g_.n_embd = (int)file_.meta_int(p + "embedding_length");
  g_.n_head = (int)file_.meta_int(p + "attention.head_count");
  g_.n_head_kv = (int)file_.meta_int_or(p + "attention.head_count_kv", g_.n_head);
  g_.head_dim = (int)file_.meta_int(p + "attention.key_length");
  g_.n_ff_exp = (int)file_.meta_int(p + "expert_feed_forward_length");
  g_.n_expert = (int)file_.meta_int(p + "expert_count");
  g_.n_expert_used = (int)file_.meta_int(p + "expert_used_count");
  g_.n_ctx_train = (int)file_.meta_int(p + "context_length");
  g_.rms_eps = (float)file_.meta_float(p + "attention.layer_norm_rms_epsilon");
  g_.rope_base = (float)file_.meta_float(p + "rope.freq_base");
  g_.n_vocab = (int)file_.meta_int_or("vocab_size",
                                      (int64_t)file_.require("token_embd.weight").dims[1]);

  if (g_.head_dim <= 0) {
    throw std::runtime_error("model: attention.key_length is missing");
  }
  if (g_.n_head % g_.n_head_kv) {
    throw std::runtime_error("model: head_count is not a multiple of head_count_kv");
  }
  if (g_.n_expert_used > g_.n_expert) {
    throw std::runtime_error("model: expert_used_count exceeds expert_count");
  }
}

void Model::bind_tensors() {
  tok_embd_ = &file_.require("token_embd.weight");
  out_norm_ = &file_.require("output_norm.weight");
  out_w_ = &file_.require("output.weight");

  check_shape(*tok_embd_, {g_.n_embd, g_.n_vocab});
  check_shape(*out_norm_, {g_.n_embd});
  check_shape(*out_w_, {g_.n_embd, g_.n_vocab});
  if (!ggml_type_is_dequantizable(tok_embd_->type) ||
      !ggml_type_is_dequantizable(out_w_->type)) {
    throw std::runtime_error("model: embedding or output weights use an unsupported type");
  }

  layers_.assign(g_.n_layer, Layer{});
  for (int i = 0; i < g_.n_layer; ++i) {
    const std::string b = "blk." + std::to_string(i) + ".";
    Layer& L = layers_[i];
    L.attn_norm = &file_.require(b + "attn_norm.weight");
    L.ffn_norm = &file_.require(b + "ffn_norm.weight");
    L.attn_q = &file_.require(b + "attn_q.weight");
    L.attn_k = &file_.require(b + "attn_k.weight");
    L.attn_v = &file_.require(b + "attn_v.weight");
    L.attn_o = &file_.require(b + "attn_output.weight");
    L.q_norm = &file_.require(b + "attn_q_norm.weight");
    L.k_norm = &file_.require(b + "attn_k_norm.weight");
    L.gate_inp = &file_.require(b + "ffn_gate_inp.weight");
    L.gate_exps = &file_.require(b + "ffn_gate_exps.weight");
    L.up_exps = &file_.require(b + "ffn_up_exps.weight");
    L.down_exps = &file_.require(b + "ffn_down_exps.weight");

    check_shape(*L.attn_norm, {g_.n_embd});
    check_shape(*L.ffn_norm, {g_.n_embd});
    check_shape(*L.attn_q, {g_.n_embd, g_.q_dim()});
    check_shape(*L.attn_k, {g_.n_embd, g_.kv_dim()});
    check_shape(*L.attn_v, {g_.n_embd, g_.kv_dim()});
    check_shape(*L.attn_o, {g_.q_dim(), g_.n_embd});
    check_shape(*L.q_norm, {g_.head_dim});
    check_shape(*L.k_norm, {g_.head_dim});
    check_shape(*L.gate_inp, {g_.n_embd, g_.n_expert});
    check_shape(*L.gate_exps, {g_.n_embd, g_.n_ff_exp, g_.n_expert});
    check_shape(*L.up_exps, {g_.n_embd, g_.n_ff_exp, g_.n_expert});
    check_shape(*L.down_exps, {g_.n_ff_exp, g_.n_embd, g_.n_expert});
  }
}

std::unique_ptr<Model> Model::open(const std::string& path, const Options& opt) {
  std::unique_ptr<Model> m(new Model());
  m->file_ = GgufFile::open(path);
  m->read_geometry();
  m->bind_tensors();

  if (opt.n_ctx <= 0) throw std::runtime_error("model: n_ctx must be positive");
  if (opt.n_ctx > m->g_.n_ctx_train) {
    // Beyond the trained window the rope frequencies are extrapolated, which
    // produces coherent-looking noise. Refuse instead of pretending.
    throw std::runtime_error("model: n_ctx " + std::to_string(opt.n_ctx) +
                             " exceeds the trained context " +
                             std::to_string(m->g_.n_ctx_train));
  }
  m->n_ctx_ = opt.n_ctx;
  m->pool_.reset(new ThreadPool(opt.threads));
  m->tok_ = Tokenizer::from_gguf(m->file_);
  if (m->tok_.vocab_size() > m->g_.n_vocab) {
    throw std::runtime_error("model: tokenizer has more tokens than the output layer");
  }

  const int C = m->n_ctx_;
  const int H = m->g_.n_embd, QD = m->g_.q_dim(), KVD = m->g_.kv_dim();
  const int F = m->g_.n_ff_exp, V = m->g_.n_vocab;
  m->x_.assign((size_t)C * H, 0.0f);
  m->xn_.assign((size_t)C * H, 0.0f);
  m->q_.assign((size_t)C * QD, 0.0f);
  m->k_.assign((size_t)C * KVD, 0.0f);
  m->v_.assign((size_t)C * KVD, 0.0f);
  m->ctx_.assign((size_t)C * QD, 0.0f);
  m->ao_.assign((size_t)C * H, 0.0f);
  m->hx_.assign((size_t)C * H, 0.0f);
  m->gat_.assign((size_t)C * H, 0.0f);
  m->rlog_.assign((size_t)C * m->g_.n_expert, 0.0f);
  m->y_gate_.assign((size_t)C * F, 0.0f);
  m->y_up_.assign((size_t)C * F, 0.0f);
  m->y_down_.assign((size_t)C * H, 0.0f);
  m->ffn_out_.assign((size_t)C * H, 0.0f);
  m->scores_.assign(C, 0.0f);
  m->logits_.assign(V, 0.0f);
  m->kv_k_.assign((size_t)m->g_.n_layer * C * KVD, 0.0f);
  m->kv_v_.assign((size_t)m->g_.n_layer * C * KVD, 0.0f);
  m->ex_tokens_.resize(m->g_.n_expert);
  m->ex_weight_.resize(m->g_.n_expert);
  m->head_out_.assign((size_t)m->g_.head_dim, 0.0f);
  m->norm_stage_.assign((size_t)H, 0.0f);
  m->norm_w_.assign((size_t)H, 0.0f);
  m->col_tmp_.assign((size_t)H, 0.0f);
  m->probs_.assign((size_t)m->g_.n_expert, 0.0f);
  m->chosen_.assign((size_t)m->g_.n_expert_used, 0);

  // inv_freq[i] = rope_base ^ (-2i / head_dim), the only frequency table the
  // forward pass needs. Computing it once keeps powf out of the inner loops.
  m->inv_freq_.resize(m->g_.head_dim / 2);
  for (int i = 0; i < m->g_.head_dim / 2; ++i) {
    m->inv_freq_[i] = std::pow(m->g_.rope_base, -2.0f * (float)i / (float)m->g_.head_dim);
  }
  return m;
}

void Model::reset() {
  pos_ = 0;
  std::fill(kv_k_.begin(), kv_k_.end(), 0.0f);
  std::fill(kv_v_.begin(), kv_v_.end(), 0.0f);
}

void Model::rmsnorm(const float* x, const TensorInfo& w, float* out, int width) {
  float ss = 0.0f;
  for (int i = 0; i < width; ++i) ss += x[i] * x[i];
  const float scale = 1.0f / std::sqrt(ss / (float)width + g_.rms_eps);
  if (w.type == GgmlType::F32) {
    const float* wp = reinterpret_cast<const float*>(file_.data_of(w));
    for (int i = 0; i < width; ++i) out[i] = x[i] * scale * wp[i];
  } else {
    // Staged in the reusable buffer when this call's width fits (it is n_embd
    // or head_dim, both <= n_embd); a wider call -- there is none today -- still
    // works rather than trashing memory.
    if ((size_t)width > norm_stage_.size()) norm_stage_.resize((size_t)width);
    float* tmp = norm_stage_.data();
    dequant_row_f32(w.type, file_.data_of(w), tmp, width);
    for (int i = 0; i < width; ++i) out[i] = x[i] * scale * tmp[i];
  }
}

void Model::rmsnorm_rows(const float* x, int m, int width, const TensorInfo& w, float* out) {
  // The weight is loop-invariant across rows, so decode it once.
  if ((size_t)width > norm_w_.size()) norm_w_.resize((size_t)width);
  float* wp = norm_w_.data();
  dequant_row_f32(w.type, file_.data_of(w), wp, width);
  for (int r = 0; r < m; ++r) {
    const float* xr = x + (size_t)r * width;
    float* orow = out + (size_t)r * width;
    float ss = 0.0f;
    for (int i = 0; i < width; ++i) ss += xr[i] * xr[i];
    const float scale = 1.0f / std::sqrt(ss / (float)width + g_.rms_eps);
    for (int i = 0; i < width; ++i) orow[i] = xr[i] * scale * wp[i];
  }
}

void Model::rope(float* v, int n_head, int pos) {
  const int d = g_.head_dim, half = d / 2;
  for (int h = 0; h < n_head; ++h) {
    float* p = v + (size_t)h * d;
    for (int i = 0; i < half; ++i) {
      const float theta = (float)pos * inv_freq_[i];
      const float c = std::cos(theta), s = std::sin(theta);
      // ggml NEOX: rotate the pair (p[i], p[i + d/2]).
      const float x0 = g_.rope_split_halves ? p[i] : p[2 * i];
      const float x1 = g_.rope_split_halves ? p[i + half] : p[2 * i + 1];
      const float y0 = x0 * c - x1 * s;
      const float y1 = x0 * s + x1 * c;
      if (g_.rope_split_halves) {
        p[i] = y0;
        p[i + half] = y1;
      } else {
        p[2 * i] = y0;
        p[2 * i + 1] = y1;
      }
    }
  }
}

void Model::softmax(float* v, int n) {
  float mx = v[0];
  for (int i = 1; i < n; ++i) mx = std::max(mx, v[i]);
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    v[i] = std::exp(v[i] - mx);
    sum += v[i];
  }
  const float inv = 1.0f / sum;
  for (int i = 0; i < n; ++i) v[i] *= inv;
}

void Model::matmul_plain(const TensorInfo& w, const float* x, int m, int ldx, float* y,
                         int ldy) {
  const int n_in = (int)w.dims[0], n_out = (int)w.dims[1];
  const uint64_t stride = column_stride_bytes(w.type, n_in);
  const uint8_t* base = file_.data_of(w);
  if ((size_t)n_in > col_tmp_.size()) col_tmp_.resize((size_t)n_in);
  float* col = col_tmp_.data();
  for (int j = 0; j < n_out; ++j) {
    dequant_row_f32(w.type, base + (uint64_t)j * stride, col, n_in);
    for (int r = 0; r < m; ++r) {
      const float* xr = x + (size_t)r * ldx;
      float acc = 0.0f;
      for (int i = 0; i < n_in; ++i) acc += col[i] * xr[i];
      y[(size_t)r * ldy + j] = acc;
    }
  }
}

void Model::matmul(const TensorInfo& w, const float* x, int m, int ldx, float* y,
                   int ldy) {
  const int n_in = (int)w.dims[0], n_out = (int)w.dims[1];
  const GgmlType type = w.type;
  const uint64_t bw = ggml_type_block_weights(type);
  if (bw == 1) {
    matmul_plain(w, x, m, ldx, y, ldy);
    return;
  }
  if (bw > (uint64_t)kMaxBlockWeights) {
    throw std::runtime_error("model: weight block is wider than 256 weights");
  }

  const uint64_t bb = ggml_type_block_bytes(type);
  const uint64_t stride = column_stride_bytes(type, n_in);
  const uint8_t* base = file_.data_of(w);

  auto body = [&](int j0, int j1) {
    // Stack, not heap: one of these is built per work-partition, so a vector
    // here allocated and freed thousands of times per token.
    float buf[kMaxBlockWeights];
    for (int j = j0; j < j1; ++j) {
      const uint8_t* src = base + (uint64_t)j * stride;
      if (m == 1) {
        // Fused decode-and-dot: the column is never materialised.
        y[j] = dequant_dot_f32(type, src, x, (uint64_t)n_in);
        continue;
      }
      for (int r = 0; r < m; ++r) y[(size_t)r * ldy + j] = 0.0f;
      const uint64_t nblk = (uint64_t)n_in / bw;
      for (uint64_t b = 0; b < nblk; ++b) {
        dequant_row_f32(type, src + b * bb, buf, bw);
        for (int r = 0; r < m; ++r) {
          const float* xr = x + (size_t)r * ldx + b * bw;
          float acc = 0.0f;
          for (uint64_t l = 0; l < bw; ++l) acc += buf[l] * xr[l];
          y[(size_t)r * ldy + j] += acc;
        }
      }
    }
  };

  // `total()`, not `size()`: size() counts only the spawned workers and
  // excludes the calling thread, which also pulls from the queue. With
  // size() the 2-thread pool reported 1 and took the serial path, so
  // --threads 2 was exactly as slow as --threads 1.
  const uint64_t work = (uint64_t)n_in * (uint64_t)n_out * (uint64_t)m;
  if (work < (1ull << 16) || pool_->total() <= 1) {
    body(0, n_out);
    return;
  }
  const int nt = std::min<int>(n_out, pool_->total() * 4);
  const int per = (n_out + nt - 1) / nt;
  pool_->for_each(nt, [&](int t) { body(t * per, std::min(n_out, (t + 1) * per)); });
}

void Model::matmul_expert(const TensorInfo& w, int e, const float* x, int m, int ldx,
                          float* y, int ldy) {
  const int n_in = (int)w.dims[0], n_out = (int)w.dims[1];
  const uint64_t one = column_stride_bytes(w.type, n_in) * (uint64_t)n_out;
  // Expert e is a contiguous slab of the same column layout; reusing matmul on
  // a view would need a second code path, so instead the slab base is moved by
  // hand and the body is shared through a tiny shim tensor-free call.
  const uint64_t total = one * (uint64_t)w.dims[2];
  if ((uint64_t)e * one + one > total) throw std::runtime_error("model: expert out of range");

  // Temporarily reinterpret as a 2-D tensor over expert `e`'s slab.
  TensorInfo sub = w;
  sub.dims = {w.dims[0], w.dims[1]};
  sub.offset = w.offset + (uint64_t)e * one;
  matmul(sub, x, m, ldx, y, ldy);
}

void Model::moe(const Layer& w, const float* hx, int m, float* out, int layer) {
  const int H = g_.n_embd, E = g_.n_expert, K = g_.n_expert_used, F = g_.n_ff_exp;

  // C21: the expert branch is the dominant component (57% of device time on the
  // reference model, docs/CODING-LOG.PENDING.md Phase 42), and a single `moe` row
  // says nothing about WHERE it goes. Each stage below carries its own scope, so
  // `--profiling` attributes the branch instead of restating that it is
  // expensive. `:layer` names them `moe-gate:12` for --profile-detail=layer; at
  // the default =class detail the suffix is stripped and the rows read
  // `moe-router`, `moe-gate`, `moe-up`, `moe-act`, `moe-down`, `moe-gather`,
  // `moe-scatter` beside a much smaller `moe` (loop and fill overhead).
  //
  // Scope names deliberately do NOT use a bare `router`/`gate`: those would
  // collide with nothing today but would read as top-level components rather
  // than as parts of `moe`.
  char nm[48];

  // Router. Softmax over every expert, then keep the top K and renormalise:
  // because softmax is monotonic, top-K of the probabilities is top-K of the
  // logits, and renormalising the kept weights is identical to softmaxing over
  // only the kept logits.
  {
    profile_name(nm, sizeof(nm), "moe-router", layer);
    KNJ_PROFILE_OP(nm);
    matmul(*w.gate_inp, hx, m, H, rlog_.data(), E);
    for (auto& v : ex_tokens_) v.clear();
    for (auto& v : ex_weight_) v.clear();

    if ((size_t)E > probs_.size()) probs_.resize((size_t)E);
    if ((size_t)K > chosen_.size()) chosen_.resize((size_t)K);
    float* probs = probs_.data();
    int* chosen = chosen_.data();
    for (int t = 0; t < m; ++t) {
      const float* lg = rlog_.data() + (size_t)t * E;
      float mx = lg[0];
      for (int e = 1; e < E; ++e) mx = std::max(mx, lg[e]);
      float sum = 0.0f;
      for (int e = 0; e < E; ++e) {
        probs[e] = std::exp(lg[e] - mx);
        sum += probs[e];
      }
      for (int e = 0; e < E; ++e) probs[e] /= sum;

      for (int k = 0; k < K; ++k) {
        int best = -1;
        for (int e = 0; e < E; ++e) {
          bool taken = false;
          for (int q = 0; q < k; ++q) taken = taken || chosen[q] == e;
          if (taken) continue;
          if (best < 0 || probs[e] > probs[best]) best = e;
        }
        chosen[k] = best;
      }
      float keep = 0.0f;
      for (int k = 0; k < K; ++k) keep += probs[chosen[k]];
      const float inv = keep > 0.0f ? 1.0f / keep : 0.0f;
      for (int k = 0; k < K; ++k) {
        ex_tokens_[chosen[k]].push_back(t);
        ex_weight_[chosen[k]].push_back(probs[chosen[k]] * inv);
      }
    }
  }

  std::fill(out, out + (size_t)m * H, 0.0f);

  for (int e = 0; e < E; ++e) {
    const int cnt = (int)ex_tokens_[e].size();
    if (cnt == 0) continue;
    const std::vector<int32_t>& rows = ex_tokens_[e];
    // Gather the rows this expert actually sees, so one expert's weights are
    // read once for all of them -- the whole point of batching a sparse layer.
    {
      profile_name(nm, sizeof(nm), "moe-gather", layer);
      KNJ_PROFILE_OP(nm);
      for (int r = 0; r < cnt; ++r) {
        std::memcpy(gat_.data() + (size_t)r * H, hx + (size_t)rows[r] * H, sizeof(float) * H);
      }
    }
    {
      profile_name(nm, sizeof(nm), "moe-gate", layer);
      KNJ_PROFILE_OP(nm);
      matmul_expert(*w.gate_exps, e, gat_.data(), cnt, H, y_gate_.data(), F);
    }
    {
      profile_name(nm, sizeof(nm), "moe-up", layer);
      KNJ_PROFILE_OP(nm);
      matmul_expert(*w.up_exps, e, gat_.data(), cnt, H, y_up_.data(), F);
    }
    {
      profile_name(nm, sizeof(nm), "moe-act", layer);
      KNJ_PROFILE_OP(nm);
      for (size_t i = 0, n = (size_t)cnt * F; i < n; ++i) {
        y_gate_[i] = silu(y_gate_[i]) * y_up_[i];
      }
    }
    {
      profile_name(nm, sizeof(nm), "moe-down", layer);
      KNJ_PROFILE_OP(nm);
      matmul_expert(*w.down_exps, e, y_gate_.data(), cnt, F, y_down_.data(), H);
    }
    {
      profile_name(nm, sizeof(nm), "moe-scatter", layer);
      KNJ_PROFILE_OP(nm);
      for (int r = 0; r < cnt; ++r) {
        const float wt = ex_weight_[e][r];
        float* dst = out + (size_t)rows[r] * H;
        const float* src = y_down_.data() + (size_t)r * H;
        for (int i = 0; i < H; ++i) dst[i] += wt * src[i];
      }
    }
  }
}

const float* Model::forward(const int32_t* tokens, int n) {
  if (n <= 0) throw std::runtime_error("model: forward needs at least one token");
  if (pos_ + n > n_ctx_) {
    throw std::runtime_error("model: context exhausted (" + std::to_string(pos_) + " + " +
                             std::to_string(n) + " > " + std::to_string(n_ctx_) + ")");
  }
  const int H = g_.n_embd, QD = g_.q_dim(), KVD = g_.kv_dim(), D = g_.head_dim;
  const int V = g_.n_vocab;
  // Intermediates are dumped for the *first* forward only. A decode step is
  // another call with n == 1, and it would otherwise overwrite the prefill's
  // dumps with single-token tensors -- quietly comparing the wrong thing.
  const bool dbg = (pos_ == 0);

  // Embeddings: the token index selects a column of token_embd, which is one
  // contiguous run of H weights.
  {
    KNJ_PROFILE_OP("embed");
    const uint64_t stride = column_stride_bytes(tok_embd_->type, H);
    const uint8_t* base = file_.data_of(*tok_embd_);
    for (int t = 0; t < n; ++t) {
      const int32_t id = tokens[t];
      if (id < 0 || id >= V) throw std::runtime_error("model: token id out of range");
      dequant_row_f32(tok_embd_->type, base + (uint64_t)id * stride,
                      x_.data() + (size_t)t * H, H);
    }
  }
  if (dbg) dump_f32("00_embed", x_.data(), (size_t)n * H);

  const float attn_scale = 1.0f / std::sqrt((float)D);

  for (int il = 0; il < g_.n_layer; ++il) {
    const Layer& w = layers_[il];
    char nm[48];

    {
      profile_name(nm, sizeof(nm), "norm", il);
      KNJ_PROFILE_OP(nm);
      rmsnorm_rows(x_.data(), n, H, *w.attn_norm, xn_.data());
    }
    if (dbg && il == 0) dump_f32("01_attn_norm", xn_.data(), (size_t)n * H);
    {
      profile_name(nm, sizeof(nm), "qkv", il);
      KNJ_PROFILE_OP(nm);
      matmul(*w.attn_q, xn_.data(), n, H, q_.data(), QD);
      matmul(*w.attn_k, xn_.data(), n, H, k_.data(), KVD);
      matmul(*w.attn_v, xn_.data(), n, H, v_.data(), KVD);
    }

    profile_name(nm, sizeof(nm), "attention", il);
    {
    KNJ_PROFILE_OP(nm);
    for (int t = 0; t < n; ++t) {
      const int p = pos_ + t;
      float* q = q_.data() + (size_t)t * QD;
      float* k = k_.data() + (size_t)t * KVD;
      // Per-head RMSNorm, then RoPE, then the cache append.
      float* tmp = head_out_.data();
      for (int h = 0; h < g_.n_head; ++h) {
        rmsnorm(q + (size_t)h * D, *w.q_norm, tmp, D);
        std::memcpy(q + (size_t)h * D, tmp, sizeof(float) * D);
      }
      for (int h = 0; h < g_.n_head_kv; ++h) {
        rmsnorm(k + (size_t)h * D, *w.k_norm, tmp, D);
        std::memcpy(k + (size_t)h * D, tmp, sizeof(float) * D);
      }
      rope(q, g_.n_head, p);
      rope(k, g_.n_head_kv, p);

      const size_t kv_base = ((size_t)il * n_ctx_ + p) * KVD;
      std::memcpy(kv_k_.data() + kv_base, k, sizeof(float) * KVD);
      std::memcpy(kv_v_.data() + kv_base, v_.data() + (size_t)t * KVD,
                  sizeof(float) * KVD);
    }
    if (dbg && il == 0) {
      dump_f32("02_q_rope", q_.data(), (size_t)n * QD);
      dump_f32("03_k_rope", k_.data(), (size_t)n * KVD);
      dump_f32("04_v", v_.data(), (size_t)n * KVD);
    }

    const int group = g_.n_head / g_.n_head_kv;
    for (int t = 0; t < n; ++t) {
      const int p = pos_ + t;
      const float* q = q_.data() + (size_t)t * QD;
      float* c = ctx_.data() + (size_t)t * QD;
      for (int h = 0; h < g_.n_head; ++h) {
        const int kvh = h / group;
        const float* qh = q + (size_t)h * D;
        for (int s = 0; s <= p; ++s) {
          const float* kh = kv_k_.data() + (((size_t)il * n_ctx_ + s) * KVD) +
                            (size_t)kvh * D;
          float acc = 0.0f;
          for (int d = 0; d < D; ++d) acc += qh[d] * kh[d];
          scores_[s] = acc * attn_scale;
        }
        softmax(scores_.data(), p + 1);
        float* ch = c + (size_t)h * D;
        std::memset(ch, 0, sizeof(float) * D);
        for (int s = 0; s <= p; ++s) {
          const float* vh = kv_v_.data() + (((size_t)il * n_ctx_ + s) * KVD) +
                            (size_t)kvh * D;
          const float a = scores_[s];
          for (int d = 0; d < D; ++d) ch[d] += a * vh[d];
        }
      }
    }
    }  // attention

    if (dbg && il == 0) dump_f32("05_ctx", ctx_.data(), (size_t)n * QD);
    {
      profile_name(nm, sizeof(nm), "attn-o", il);
      KNJ_PROFILE_OP(nm);
      matmul(*w.attn_o, ctx_.data(), n, QD, ao_.data(), H);
      for (size_t i = 0, cnt = (size_t)n * H; i < cnt; ++i) x_[i] += ao_[i];
    }
    if (dbg && il == 0) dump_f32("06_after_attn", x_.data(), (size_t)n * H);

    {
      profile_name(nm, sizeof(nm), "norm-ffn", il);
      KNJ_PROFILE_OP(nm);
      rmsnorm_rows(x_.data(), n, H, *w.ffn_norm, hx_.data());
    }
    if (dbg && il == 0) dump_f32("07_ffn_norm", hx_.data(), (size_t)n * H);

    // The expert branch is a *residual* addition: llama.cpp computes
    // `cur = moe_out + ffn_inp`, where ffn_inp is the tensor ffn_norm was
    // taken of. moe() must therefore write beside x_, never into it -- an
    // earlier version passed x_ as the output, and the `fill` that starts the
    // accumulation silently erased every layer's FFN residual.
    {
      profile_name(nm, sizeof(nm), "moe", il);
      KNJ_PROFILE_OP(nm);
      moe(w, hx_.data(), n, ffn_out_.data(), il);
    }
    if (dbg && il == 0) {
      dump_f32("08_router_logits", rlog_.data(), (size_t)n * g_.n_expert);
      dump_f32("09b_moe_out", ffn_out_.data(), (size_t)n * H);
    }
    {
      profile_name(nm, sizeof(nm), "residual", il);
      KNJ_PROFILE_OP(nm);
      for (size_t i = 0, cnt = (size_t)n * H; i < cnt; ++i) x_[i] += ffn_out_[i];
    }
    if (dbg) {
      char nm[32];
      std::snprintf(nm, sizeof(nm), "10_after_layer%02d", il);
      dump_f32(nm, x_.data(), (size_t)n * H);
    }
  }

  // Only the last position's logits are needed by any caller here.
  {
    KNJ_PROFILE_OP("head");
    rmsnorm(x_.data() + (size_t)(n - 1) * H, *out_norm_, xn_.data(), H);
    if (dbg) dump_f32("11_final_norm", xn_.data(), H);
    matmul(*out_w_, xn_.data(), 1, H, logits_.data(), V);
  }
  if (dbg) dump_f32("12_logits", logits_.data(), (size_t)V);

  pos_ += n;
  return logits_.data();
}

}  // namespace knj
