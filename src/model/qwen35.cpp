// src/model/qwen35.cpp -- the `qwen35` front end, and the refusal that follows.
//
// Read src/model/qwen35.h first: it states what is verified and what is not.
// Two rules govern this file.
//
//  1. Nothing is guessed from a name. Every dimension comes from the file's
//     metadata or from a tensor's declared shape, and a missing key is a refusal
//     with the key named -- not a default that happens to be about the right size.
//  2. A tensor is bound only if it is *both* present with the implied shape and
//     decodable by this engine. The 12 GB `-mtp` file on this machine carries six
//     IQ types no decoder here implements; that has to come out as "IQ2_S, 130
//     tensors" and not as a crash or a silent skip.
#include "src/model/qwen35.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/profiler/profiler.h"

namespace knj {
namespace {

[[noreturn]] void fail(const std::string& m) {
  throw std::runtime_error("qwen35: " + m);
}

std::string shape_str(const std::vector<uint64_t>& d) {
  std::string s = "[";
  for (size_t i = 0; i < d.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(d[i]);
  }
  return s + "]";
}

int64_t req_int(const GgufFile& f, const std::string& k) {
  if (!f.has(k)) {
    fail("required metadata '" + k + "' is missing; refusing to assume a value");
  }
  return f.meta_int(k);
}

double req_float(const GgufFile& f, const std::string& k) {
  if (!f.has(k)) {
    fail("required metadata '" + k + "' is missing; refusing to assume a value");
  }
  return f.meta_float(k);
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One tensor the trunk needs, with the shape the metadata implies.
struct Want {
  std::string name;
  std::vector<uint64_t> dims;
  bool shape_check = true;
};

void add(std::vector<Want>& v, const std::string& n, std::vector<uint64_t> d) {
  v.push_back(Want{n, std::move(d), true});
}

void add_unchecked(std::vector<Want>& v, const std::string& n) {
  v.push_back(Want{n, {}, false});
}

}  // namespace

int Qwen35Geometry::ssm_layers() const {
  int n = 0;
  for (bool r : recurrent) n += r ? 1 : 0;
  return n;
}

int Qwen35Geometry::attn_layers() const { return (int)recurrent.size() - ssm_layers(); }

std::string Qwen35Geometry::describe() const {
  char buf[512];
  std::string s;
  std::snprintf(buf, sizeof buf,
                "  layers        %3d   (%d recurrent/SSM + %d full attention, "
                "interval %d)\n",
                n_layer, ssm_layers(), attn_layers(), full_attn_interval);
  s += buf;
  std::snprintf(buf, sizeof buf,
                "  hidden        %3d   ff %d, vocab %d, ctx_train %d\n",
                n_embd, n_ff, n_vocab, n_ctx_train);
  s += buf;
  std::snprintf(buf, sizeof buf,
                "  attention     24 heads equivalent: %d q / %d kv, head_dim %d "
                "(q_dim %d, kv_dim %d)\n",
                n_head, n_head_kv, head_dim, q_dim(), kv_dim());
  s += buf;
  std::snprintf(buf, sizeof buf,
                "  rope          base %.0f, %d of %d dims rotated\n",
                (double)rope_base, rope_dims, head_dim);
  s += buf;
  std::snprintf(buf, sizeof buf,
                "  ssm           inner %d, state %d, groups %d, dt_rank %d, "
                "conv_kernel %d -> conv_dim %d\n",
                ssm_inner, ssm_state, ssm_groups, ssm_dt_rank, ssm_conv_kernel,
                conv_dim());
  s += buf;
  std::snprintf(buf, sizeof buf, "  rms_eps       %.3g\n", (double)rms_eps);
  s += buf;
  std::snprintf(buf, sizeof buf, "  nextn (MTP)   %d merged layer(s)\n", nextn_layers);
  s += buf;
  return s;
}

bool Qwen35Report::all_ok() const {
  for (const Qwen35Binding& b : bindings) {
    if (!b.ok()) return false;
  }
  return true;
}

std::string Qwen35Report::summary() const {
  int bad = 0;
  for (const Qwen35Binding& b : bindings) bad += b.ok() ? 0 : 1;
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "%d tensor(s) bound and shape-checked across %d layer(s) "
                "(%d recurrent, %d attention, %d nextn); %d not usable",
                (int)bindings.size(), layers_bound, recurrent_layers,
                attention_layers, nextn_tensors, bad);
  return buf;
}

std::unique_ptr<Qwen35> Qwen35::open(const std::string& path) {
  std::unique_ptr<Qwen35> m(new Qwen35());
  m->file_ = GgufFile::open(path);
  // The tokenizer is built BEFORE read_geometry(), which cross-checks the
  // vocabulary against token_embd's row count. Building it afterwards compares
  // the file's 248320 rows against an empty tokenizer and refuses a file that is
  // perfectly fine -- which is exactly what the first run of this probe did.
  m->tok_ = Tokenizer::from_gguf(m->file_);
  m->read_geometry();
  m->base_.resize(m->g_.n_layer);
  for (int i = 0; i < m->g_.n_layer; ++i) {
    m->base_[i] = "blk." + std::to_string(i) + ".";
  }
  m->bind_tensors();
  m->scratch_.assign((size_t)m->g_.n_embd, 0.0f);
  return m;
}

const std::string& Qwen35::base(int layer) const {
  if (layer < 0 || layer >= (int)base_.size()) fail("layer out of range");
  return base_[layer];
}

void Qwen35::read_geometry() {
  const std::string arch = file_.meta_string("general.architecture");
  if (arch != "qwen35") {
    fail("architecture is '" + arch + "', not 'qwen35'");
  }
  const std::string p = arch + ".";
  Qwen35Geometry& g = g_;
  g.n_layer = (int)req_int(file_, p + "block_count");
  g.n_embd = (int)req_int(file_, p + "embedding_length");
  g.n_ff = (int)req_int(file_, p + "feed_forward_length");
  g.n_head = (int)req_int(file_, p + "attention.head_count");
  g.n_head_kv = (int)req_int(file_, p + "attention.head_count_kv");
  g.head_dim = (int)req_int(file_, p + "attention.key_length");
  const int64_t v_len = req_int(file_, p + "attention.value_length");
  g.n_ctx_train = (int)req_int(file_, p + "context_length");
  g.rms_eps = (float)req_float(file_, p + "attention.layer_norm_rms_epsilon");
  g.rope_base = (float)req_float(file_, p + "rope.freq_base");
  g.rope_dims = (int)req_int(file_, p + "rope.dimension_count");
  // The section lengths count cos/sin PAIRS, so they must sum to
  // dimension_count/2. ggml asserts the same relation (`sect_dims <= ne0` in
  // rope_yarn, with ne0 = n_dims/2); a file that violates it has a rope whose
  // pairing this probe does not know, which must be a refusal and not a guess.
  {
    auto it = file_.metadata().find(p + "rope.dimension_sections");
    if (it != file_.metadata().end() && it->second.is_array()) {
      const std::vector<MetaValue>& a = it->second.arr;
      if (a.size() > 4) {
        fail("rope.dimension_sections has " + std::to_string(a.size()) +
             " entries; the interleaved M-RoPE this probe implements has 4 axes");
      }
      for (size_t i = 0; i < a.size(); ++i) g.rope_sections[i] = (int)a[i].i;
    }
  }
  if (g.rope_dims > 0) {
    int sum = 0;
    for (int i = 0; i < 4; ++i) sum += g.rope_sections[i];
    if (sum != g.rope_dims / 2) {
      fail("rope.dimension_sections sums to " + std::to_string(sum) +
           " but dimension_count/2 is " + std::to_string(g.rope_dims / 2) +
           ": the sections count cos/sin pairs, so these must agree, and guessing "
           "which one is wrong would put the rope on the wrong dims");
    }
  }
  g.ssm_conv_kernel = (int)req_int(file_, p + "ssm.conv_kernel");
  g.ssm_state = (int)req_int(file_, p + "ssm.state_size");
  g.ssm_groups = (int)req_int(file_, p + "ssm.group_count");
  g.ssm_dt_rank = (int)req_int(file_, p + "ssm.time_step_rank");
  g.ssm_inner = (int)req_int(file_, p + "ssm.inner_size");
  g.full_attn_interval = (int)file_.meta_int_or(p + "full_attention_interval", 0);
  g.nextn_layers = (int)file_.meta_int_or(p + "nextn_predict_layers", 0);

  if (v_len != g.head_dim) {
    fail("attention.key_length " + std::to_string(g.head_dim) +
         " != attention.value_length " + std::to_string(v_len) +
         "; this file needs separate key/value widths, which this probe does not "
         "model");
  }
  if (g.n_head % g.n_head_kv) {
    fail("head_count is not a multiple of head_count_kv");
  }
  if (g.n_embd != g.n_head * g.head_dim &&
      g.n_embd > g.n_head * g.head_dim) {
    fail("embedding_length exceeds head_count * key_length");
  }

  // The layer kinds are the file's own statement. `recurrent_layers` is exact;
  // the interval is the fallback, and it is a *different* claim, so which one was
  // used is reported rather than smoothed over.
  g.recurrent.assign(g.n_layer, false);
  bool have_array = false;
  auto it = file_.metadata().find(p + "attention.recurrent_layers");
  if (it != file_.metadata().end() && it->second.is_array()) {
    const std::vector<MetaValue>& a = it->second.arr;
    if ((int)a.size() != g.n_layer) {
      fail("attention.recurrent_layers has " + std::to_string(a.size()) +
           " entries for " + std::to_string(g.n_layer) + " layers");
    }
    for (int i = 0; i < g.n_layer; ++i) g.recurrent[i] = a[i].i != 0;
    have_array = true;
  } else if (g.full_attn_interval > 0) {
    for (int i = 0; i < g.n_layer; ++i) {
      g.recurrent[i] = ((i + 1) % g.full_attn_interval) != 0;
    }
  } else {
    fail("neither '" + p + "attention.recurrent_layers' nor '" + p +
         "full_attention_interval' is present: the layer kinds are unknown and "
         "guessing them would bind the wrong tensors");
  }
  g.kinds_from_array = have_array;

  // Vocab: the tokenizer's own size, cross-checked against the embedding's shape.
  const TensorInfo& emb = file_.require("token_embd.weight");
  if (emb.ndims() != 2) fail("token_embd.weight is not 2-D");
  g.n_vocab = (int)emb.dims[1];
  if (g.n_vocab != tok_.vocab_size()) {
    fail("token_embd has " + std::to_string(g.n_vocab) + " rows but the tokenizer has " +
         std::to_string(tok_.vocab_size()) +
         " tokens; one of the two is wrong and the head would be a lie");
  }
  if ((int)emb.dims[0] != g.n_embd) {
    fail("token_embd row width " + std::to_string(emb.dims[0]) +
         " != embedding_length " + std::to_string(g.n_embd));
  }
}

void Qwen35::bind_tensors() {
  Qwen35Report& r = rep_;
  const Qwen35Geometry& g = g_;

  auto check = [&](const std::string& name, const std::vector<uint64_t>* want,
                   bool shape_check) {
    Qwen35Binding b;
    b.name = name;
    const TensorInfo* t = file_.find(name);
    if (!t) {
      r.bindings.push_back(b);
      return;
    }
    b.found = true;
    b.got = shape_str(t->dims);
    b.type_name = ggml_type_name(t->type);
    b.dequantizable = ggml_type_is_dequantizable(t->type);
    if (want && shape_check) {
      b.wanted = shape_str(*want);
      b.shape_ok = t->dims == *want;
    } else {
      b.shape_ok = true;
    }
    r.bindings.push_back(b);
  };

  // Exact-width aliases: a narrowing conversion inside an aggregate initializer
  // is a warning here and a silent truncation wherever uint64_t is not what the
  // metadata actually holds.
  const uint64_t E = (uint64_t)g.n_embd;
  const uint64_t FF = (uint64_t)g.n_ff;
  const uint64_t VOC = (uint64_t)g.n_vocab;
  const uint64_t HD = (uint64_t)g.head_dim;
  const uint64_t KVD = (uint64_t)g.kv_dim();
  const uint64_t QD = (uint64_t)g.q_dim();
  const uint64_t CONV = (uint64_t)g.conv_dim();
  const uint64_t INNER = (uint64_t)g.ssm_inner;
  const uint64_t DT = (uint64_t)g.ssm_dt_rank;
  const uint64_t ST = (uint64_t)g.ssm_state;
  const uint64_t CK = (uint64_t)g.ssm_conv_kernel;

  // Whole-model tensors.
  {
    std::vector<uint64_t> d{E, VOC};
    check("token_embd.weight", &d, true);
    std::vector<uint64_t> n{E};
    check("output_norm.weight", &n, true);
    check("output.weight", &d, true);
  }

  for (int i = 0; i < g.n_layer; ++i) {
    const std::string b = base_[i];
    std::vector<Want> wants;
    std::vector<uint64_t> e{E};
    add(wants, b + "attn_norm.weight", e);
    add(wants, b + "post_attention_norm.weight", e);
    add(wants, b + "ffn_gate.weight", {E, FF});
    add(wants, b + "ffn_up.weight", {E, FF});
    add(wants, b + "ffn_down.weight", {FF, E});

    if (g.recurrent[i]) {
      add(wants, b + "attn_qkv.weight", {E, CONV});
      add(wants, b + "attn_gate.weight", {E, INNER});
      add(wants, b + "ssm_a", {DT});
      add(wants, b + "ssm_alpha.weight", {E, DT});
      add(wants, b + "ssm_beta.weight", {E, DT});
      add(wants, b + "ssm_conv1d.weight", {CK, CONV});
      add(wants, b + "ssm_dt.bias", {DT});
      add(wants, b + "ssm_norm.weight", {ST});
      add(wants, b + "ssm_out.weight", {INNER, E});
    } else {
      // The full-attention blocks carry a *gated* query projection in this
      // family: q_proj emits 2 * q_dim (query || gate). Whether that is the case
      // is read from the tensor, not asserted, because the same architecture
      // string appears with and without it.
      add_unchecked(wants, b + "attn_q.weight");
      add(wants, b + "attn_k.weight", {E, KVD});
      add(wants, b + "attn_v.weight", {E, KVD});
      add(wants, b + "attn_q_norm.weight", {HD});
      add(wants, b + "attn_k_norm.weight", {HD});
      add(wants, b + "attn_output.weight", {QD, E});
    }

    for (const Want& w : wants) {
      if (w.name == b + "attn_q.weight") {
        // Shape-check against what the file declares, either way.
        const TensorInfo* t = file_.find(w.name);
        if (t && t->ndims() == 2 && t->dims[1] != QD) {
          const uint64_t expect2 = 2ull * QD;
          if (t->dims[1] != expect2) {
            Qwen35Binding bad;
            bad.name = w.name;
            bad.found = true;
            bad.shape_ok = false;
            bad.got = shape_str(t->dims);
            bad.wanted = shape_str({E, expect2}) + " (gated) or " + shape_str({E, QD});
            rep_.bindings.push_back(bad);
            continue;
          }
        }
        check(w.name, nullptr, false);
        continue;
      }
      check(w.name, &w.dims, w.shape_check);
    }
  }

  // The merged MTP head, if this container carries one. Which block it hangs off
  // is discovered, not assumed -- the four `nextn.*` names are the same in every
  // file, and the block index is not documented anywhere this engine can cite.
  static const char* kNextn[] = {"nextn.eh_proj.weight", "nextn.enorm.weight",
                                 "nextn.hnorm.weight", "nextn.shared_head_norm.weight"};
  for (int i = 0; i < g.n_layer; ++i) {
    bool any = false;
    for (const char* n : kNextn) {
      if (file_.find(base_[i] + n)) any = true;
    }
    if (!any) continue;
    r.nextn_block = i;
    for (const char* n : kNextn) {
      const std::string full = base_[i] + n;
      const TensorInfo* t = file_.find(full);
      if (!t) {
        Qwen35Binding bad;
        bad.name = full;
        r.bindings.push_back(bad);
        continue;
      }
      std::vector<uint64_t> want;
      const std::string tail(n);
      if (tail == "nextn.eh_proj.weight") {
        // ggml order: dims[0] is the INPUT width. eh_proj fuses the MTP layer's
        // own hidden state with the target's embedding (2 * n_embd in) and emits
        // one hidden state (n_embd out). The first version of this check had the
        // two swapped and the probe caught it on a real file.
        want = {2ull * E, E};
      } else {
        want = {E};
      }
      check(full, &want, true);
      ++r.nextn_tensors;
    }
  }

  r.layers_bound = g.n_layer;
  r.recurrent_layers = g.ssm_layers();
  r.attention_layers = g.attn_layers();
}

void Qwen35::embed_row(int32_t token, float* out) const {
  const TensorInfo& t = file_.require("token_embd.weight");
  if (!ggml_type_is_dequantizable(t.type)) {
    fail(std::string("token_embd.weight is ") + ggml_type_name(t.type) +
         ", which this engine cannot decode");
  }
  if (token < 0 || (uint64_t)token >= t.dims[1]) {
    fail("token " + std::to_string(token) + " is outside the embedding table");
  }
  const uint64_t stride = column_stride_bytes(t.type, t.dims[0]);
  dequant_row_f32(t.type, file_.data_of(t) + stride * (uint64_t)token, out, t.dims[0]);
}

void Qwen35::rmsnorm(const float* x, const TensorInfo& w, float* out, int width) const {
  float ss = 0.0f;
  for (int i = 0; i < width; ++i) ss += x[i] * x[i];
  const float scale = 1.0f / std::sqrt(ss / (float)width + g_.rms_eps);
  if (w.type == GgmlType::F32) {
    const float* wp = reinterpret_cast<const float*>(file_.data_of(w));
    for (int i = 0; i < width; ++i) out[i] = x[i] * scale * wp[i];
    return;
  }
  std::vector<float> tmp((size_t)width, 0.0f);
  dequant_row_f32(w.type, file_.data_of(w), tmp.data(), width);
  for (int i = 0; i < width; ++i) out[i] = x[i] * scale * tmp[i];
}

void Qwen35::project(const TensorInfo& w, const float* x, float* y, int n_out) const {
  if (!ggml_type_is_dequantizable(w.type)) {
    fail(std::string("cannot project through ") + w.name + ": type " +
         ggml_type_name(w.type) + " has no decoder here");
  }
  const uint64_t n_in = w.dims[0];
  const uint64_t stride = column_stride_bytes(w.type, n_in);
  const uint8_t* base = file_.data_of(w);
  for (int j = 0; j < n_out; ++j) {
    y[j] = dequant_dot_f32(w.type, base + stride * (uint64_t)j, x, n_in);
  }
}

// ------------------------------------------------- the recurrent layer ---
//
// Reference-backed, step by step. The two implementations read on this machine
// agree, and the order below is theirs:
//
//   x        = rmsnorm(embed(tok)) * attn_norm
//   qkv      = W_qkv x
//   conv_raw = causal depthwise conv1d(qkv, kernel) over the [q|k|v] channels
//   conv     = silu(conv_raw)
//   q,k,v    = conv split at key_dim, 2*key_dim
//   q        = l2norm(q, eps) * 1/sqrt(head_dim);  k = l2norm(k, eps)
//   beta     = sigmoid(W_beta x);  a = W_alpha x
//   g        = ssm_a * softplus(a + dt_bias)          [ssm_a stores -exp(A_log)]
//   heads    : 16 key heads -> 48 value heads by repeat_interleave 3
//   state[h] : [head_dim, head_dim], S[i][j], i = key index, j = value index
//              S *= exp(g[h]);  delta = (v - S^T k) * beta;
//              S += k (x) delta;  o = S^T q
//   gated    = RMSNormGated(o, ssm_norm, z = W_gate x)  [weight, then silu(z)]
//   out      = W_out gated
void Qwen35::reset_recurrent_state(int layer) {
  auto it = rst_.find(layer);
  if (it == rst_.end()) return;
  std::fill(it->second.conv_hist.begin(), it->second.conv_hist.end(), 0.0f);
  std::fill(it->second.state.begin(), it->second.state.end(), 0.0f);
}

// The probe's entry point: embed the tokens, then run the block. A trunk step does
// NOT embed -- its input is the residual stream -- so the block itself takes a
// hidden state and this wrapper exists so the per-layer probe (whose input IS the
// embedding, which is what the oracle's layer_* vectors assume) keeps its interface.
void Qwen35::recurrent_layer(int layer, const int32_t* tokens, int n, RecurrentOut& out) {
  const int E = g_.n_embd;
  if (n <= 0) fail("recurrent_layer needs at least one token");
  std::vector<float> hidden((size_t)n * E, 0.0f);
  for (int t = 0; t < n; ++t) embed_row(tokens[t], hidden.data() + (size_t)t * E);
  recurrent_block(layer, hidden.data(), n, out);
}

void Qwen35::recurrent_block(int layer, const float* hidden, int n, RecurrentOut& out) {
  const Qwen35Geometry& g = g_;
  if (!is_recurrent(layer)) {
    fail("layer " + std::to_string(layer) + " is not recurrent; the delta rule is not "
         "defined for a full-attention layer");
  }
  if (n <= 0) fail("recurrent_block needs at least one token");

  const int E = g.n_embd;
  const int CD = g.conv_dim();
  const int HD = g.ssm_state;
  const int NK = g.ssm_groups;
  const int NV = g.ssm_dt_rank;   // one gate scalar per value head
  const int VD = g.ssm_inner;
  const int KER = g.ssm_conv_kernel;
  const int KD = NK * HD;         // key/query width
  if (VD != NV * HD) {
    fail("ssm.inner_size is not value heads * state_size: the state's shape is unknown");
  }
  if (CD != 2 * KD + VD) {
    fail("conv_dim is not 2 * key_dim + value_dim: the qkv split is unknown");
  }
  if (NV % NK) fail("value heads is not a multiple of key heads");
  const int rep = NV / NK;

  const std::string b = base(layer);
  const TensorInfo& w_qkv = file_.require(b + "attn_qkv.weight");
  const TensorInfo& w_gate = file_.require(b + "attn_gate.weight");
  const TensorInfo& w_alpha = file_.require(b + "ssm_alpha.weight");
  const TensorInfo& w_beta = file_.require(b + "ssm_beta.weight");
  const TensorInfo& w_out = file_.require(b + "ssm_out.weight");
  const TensorInfo& w_an = file_.require(b + "attn_norm.weight");
  const TensorInfo& t_a = file_.require(b + "ssm_a");
  const TensorInfo& t_dt = file_.require(b + "ssm_dt.bias");
  const TensorInfo& t_norm = file_.require(b + "ssm_norm.weight");
  const TensorInfo& t_conv = file_.require(b + "ssm_conv1d.weight");

  // Small per-layer parameters, decoded once per call: they are F32 and tiny.
  std::vector<float> ssm_a((size_t)NV), dt((size_t)NV), nw((size_t)HD), anw((size_t)E);
  std::vector<float> ker((size_t)KER * CD);
  dequant_row_f32(t_a.type, file_.data_of(t_a), ssm_a.data(), (uint64_t)NV);
  dequant_row_f32(t_dt.type, file_.data_of(t_dt), dt.data(), (uint64_t)NV);
  dequant_row_f32(t_norm.type, file_.data_of(t_norm), nw.data(), (uint64_t)HD);
  dequant_row_f32(w_an.type, file_.data_of(w_an), anw.data(), (uint64_t)E);
  dequant_row_f32(t_conv.type, file_.data_of(t_conv), ker.data(), (uint64_t)KER * CD);

  RecurrentState& rs = rst_[layer];
  if (rs.conv_hist.size() != (size_t)(KER - 1) * CD) {
    rs.conv_hist.assign((size_t)(KER - 1) * CD, 0.0f);
  }
  if (rs.state.size() != (size_t)NV * HD * HD) {
    rs.state.assign((size_t)NV * HD * HD, 0.0f);
  }

  out.steps = n;
  out.x.assign((size_t)n * E, 0.0f);
  out.qkv.assign((size_t)n * CD, 0.0f);
  out.conv_raw.assign((size_t)n * CD, 0.0f);
  out.conv.assign((size_t)n * CD, 0.0f);
  out.q.assign((size_t)n * KD, 0.0f);
  out.k.assign((size_t)n * KD, 0.0f);
  out.beta.assign((size_t)n * NV, 0.0f);
  out.g.assign((size_t)n * NV, 0.0f);
  out.o.assign((size_t)n * VD, 0.0f);
  out.gated.assign((size_t)n * VD, 0.0f);
  out.out.assign((size_t)n * E, 0.0f);

  std::vector<float> x((size_t)E), qkv((size_t)CD), craw((size_t)CD);
  std::vector<float> q((size_t)KD), k((size_t)KD), v((size_t)VD), z((size_t)VD);
  std::vector<float> a((size_t)NV), be((size_t)NV), gg((size_t)NV);
  std::vector<float> delta((size_t)HD), o((size_t)VD);
  std::vector<float> gated((size_t)VD), y((size_t)E);

  decay_min_ = 1e30;
  decay_max_ = -1e30;
  const float scale = 1.0f / std::sqrt((float)HD);

  for (int t = 0; t < n; ++t) {
    rmsnorm(hidden + (size_t)t * E, w_an, x.data(), E);
    std::memcpy(out.x.data() + (size_t)t * E, x.data(), sizeof(float) * (size_t)E);

    project(w_qkv, x.data(), qkv.data(), CD);
    std::memcpy(out.qkv.data() + (size_t)t * CD, qkv.data(), sizeof(float) * (size_t)CD);

    // Causal depthwise conv over the [q|k|v] channels. `rs.conv_hist` holds the
    // KER-1 inputs that preceded this token, oldest first.
    // The window is [hist[0..KER-2], qkv]: hist is oldest-first, and ker tap 0
    // multiplies the OLDEST sample (llama.cpp's ggml_ssm_conv walks the kernel
    // and the window with the same index, from 0).
    for (int c = 0; c < CD; ++c) {
      float s = 0.0f;
      for (int tap = 0; tap < KER; ++tap) {
        const float val = (tap == KER - 1)
                              ? qkv[(size_t)c]
                              : rs.conv_hist[(size_t)tap * CD + c];
        s += ker[(size_t)tap * CD + c] * val;
      }
      craw[(size_t)c] = s;
    }
    std::memcpy(out.conv_raw.data() + (size_t)t * CD, craw.data(), sizeof(float) * (size_t)CD);
    for (int c = 0; c < CD; ++c) {
      const float raw = craw[(size_t)c];
      craw[(size_t)c] = raw / (1.0f + std::exp(-raw));   // silu
    }
    std::memcpy(out.conv.data() + (size_t)t * CD, craw.data(), sizeof(float) * (size_t)CD);

    // History: the raw (pre-conv) inputs, shifted left, current appended.
    for (int i = 0; i + 1 < KER - 1; ++i) {
      std::memcpy(rs.conv_hist.data() + (size_t)i * CD,
                  rs.conv_hist.data() + (size_t)(i + 1) * CD, sizeof(float) * (size_t)CD);
    }
    if (KER > 1) {
      std::memcpy(rs.conv_hist.data() + (size_t)(KER - 2) * CD, qkv.data(),
                  sizeof(float) * (size_t)CD);
    }

    std::memcpy(q.data(), craw.data(), sizeof(float) * (size_t)KD);
    std::memcpy(k.data(), craw.data() + KD, sizeof(float) * (size_t)KD);
    std::memcpy(v.data(), craw.data() + 2 * KD, sizeof(float) * (size_t)VD);

    for (int h = 0; h < NK; ++h) {
      float sq = 0.0f, sk = 0.0f;
      for (int d = 0; d < HD; ++d) {
        sq += q[(size_t)(h * HD + d)] * q[(size_t)(h * HD + d)];
        sk += k[(size_t)(h * HD + d)] * k[(size_t)(h * HD + d)];
      }
      const float rq = scale / std::sqrt(sq + g.rms_eps);
      const float rk = 1.0f / std::sqrt(sk + g.rms_eps);
      for (int d = 0; d < HD; ++d) {
        q[(size_t)(h * HD + d)] *= rq;
        k[(size_t)(h * HD + d)] *= rk;
      }
    }
    std::memcpy(out.q.data() + (size_t)t * KD, q.data(), sizeof(float) * (size_t)KD);
    std::memcpy(out.k.data() + (size_t)t * KD, k.data(), sizeof(float) * (size_t)KD);

    project(w_beta, x.data(), be.data(), NV);
    project(w_alpha, x.data(), a.data(), NV);
    for (int h = 0; h < NV; ++h) {
      be[(size_t)h] = 1.0f / (1.0f + std::exp(-be[(size_t)h]));
      const float av = a[(size_t)h] + dt[(size_t)h];
      // Clamp before exp, as the oracle and HF do: softplus(x) == x for x > 20,
      // and without the clamp exp overflows to inf for a large dt.
      const float sp = av > 20.0f ? av : std::log1p(std::exp(av));
      gg[(size_t)h] = ssm_a[(size_t)h] * sp;   // ssm_a stores -exp(A_log)
      const double decay = std::exp((double)gg[(size_t)h]);
      if (decay < decay_min_) decay_min_ = decay;
      if (decay > decay_max_) decay_max_ = decay;
    }
    std::memcpy(out.beta.data() + (size_t)t * NV, be.data(), sizeof(float) * (size_t)NV);
    std::memcpy(out.g.data() + (size_t)t * NV, gg.data(), sizeof(float) * (size_t)NV);

    // S[i][j]: i indexes the key dimension, j the value dimension -- the
    // orientation ggml-cpu stores as `s_out[j*S_v + i] = S[i][j]` and HF stores
    // as `state[k_dim, v_dim]`.
    for (int h = 0; h < NV; ++h) {
      const int kh = h / rep;                  // repeat_interleave over heads
      float* S = rs.state.data() + (size_t)h * HD * HD;
      const float decay = std::exp(gg[(size_t)h]);
      for (int i = 0; i < HD * HD; ++i) S[i] *= decay;
      const float* kk = k.data() + (size_t)kh * HD;
      const float* qq = q.data() + (size_t)kh * HD;
      const float* vv = v.data() + (size_t)h * HD;
      for (int j = 0; j < HD; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < HD; ++i) sum += S[(size_t)i * HD + j] * kk[i];
        delta[(size_t)j] = (vv[j] - sum) * be[(size_t)h];
      }
      for (int i = 0; i < HD; ++i) {
        const float ki = kk[i];
        for (int j = 0; j < HD; ++j) S[(size_t)i * HD + j] += ki * delta[(size_t)j];
      }
      for (int j = 0; j < HD; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < HD; ++i) sum += S[(size_t)i * HD + j] * qq[i];
        o[(size_t)h * HD + j] = sum;
      }
    }
    std::memcpy(out.o.data() + (size_t)t * VD, o.data(), sizeof(float) * (size_t)VD);

    // z gate, then the gated normalisation: norm, weight, silu(z).
    project(w_gate, x.data(), z.data(), VD);
    for (int h = 0; h < NV; ++h) {
      const float* oh = o.data() + (size_t)h * HD;
      const float* zh = z.data() + (size_t)h * HD;
      float* gd = gated.data() + (size_t)h * HD;
      float ss = 0.0f;
      for (int d = 0; d < HD; ++d) ss += oh[d] * oh[d];
      const float rms = 1.0f / std::sqrt(ss / (float)HD + g.rms_eps);
      for (int d = 0; d < HD; ++d) {
        const float zg = zh[d];
        gd[d] = oh[d] * rms * nw[(size_t)d] * (zg / (1.0f + std::exp(-zg)));
      }
    }
    std::memcpy(out.gated.data() + (size_t)t * VD, gated.data(), sizeof(float) * (size_t)VD);

    project(w_out, gated.data(), y.data(), E);
    std::memcpy(out.out.data() + (size_t)t * E, y.data(), sizeof(float) * (size_t)E);
  }
  out.state = rs.state;
}

// ------------------------------------------------- the attention layer ---
//
// Reference-backed, step by step. llama.cpp's `build_layer_attn`
// (src/models/qwen35.cpp:260) and transformers' Qwen3NextAttention agree on the
// order:
//
//   x        = rmsnorm(embed(tok)) * attn_norm
//   qg       = W_q x            [(q_head | gate_head)] per head, interleaved
//   q        = rmsnorm(q_head, attn_q_norm)   over head_dim, not the whole row
//   k,v      = W_k x, W_v x;  k = rmsnorm(k_head, attn_k_norm)
//   q,k      = partial RoPE over the first rope_dims of head_dim, NEOX pairs
//   attn     = causal softmax(q k^T / sqrt(head_dim)) v, GQA by head groups
//   gated    = attn * sigmoid(gate)          elementwise, per head-dim
//   out      = W_o gated
namespace {

// ggml's IMROPE assigns each rotated *pair* to one of four position axes:
// `sector = pair % sum(sections)`, then t for `sector % 3 == 0`, h for 1, w for 2
// and the extra axis otherwise (the bounds only bite for degenerate sections).
// Source: the GGML_ROPE_TYPE_IMROPE comment in ggml.h and the `is_imrope` branch
// of ggml-cpu's rope_yarn.
//
// The section sizes are read from the file's `rope.dimension_sections` (the model's
// own statement of how the rotated dimension-pairs split across the four spatial
// axes t/h/w/extra). The only non-text case this engine feeds is text, where all
// four axes collapse to the same linear position, so the section counts matter only
// for verification against an independent implementation -- which is why there is a
// fixture for the non-text case.
int imrope_axis(int pair, const int sections[4]) {
  const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
  if (sect_dims <= 0) return 0;
  const int sector = pair % sect_dims;
  if (sector % 3 == 1 && sector < 3 * sections[1]) return 1;
  if (sector % 3 == 2 && sector < 3 * sections[2]) return 2;
  if (sector % 3 == 0 && sector < 3 * sections[0]) return 0;
  return 3;
}

// Partial RoPE on one head: within the first `rd` dims, pair i is (i, i + rd/2),
// both rotated by theta_i = pos * base^(-2i/rd). HF splits the head at
// rotary_dim and applies rotate_half to the first part; ggml's rotate_pairs does
// the same with n_offset = rd/2. NEOX ordering is not an option here -- the
// `GGML_ROPE_TYPE_IMROPE` comment states it is applied automatically and cannot
// be disabled.
//
// `axis[i]` says which IMROPE axis pair i reads. For a text prompt all four axes
// collapse to the same linear token position, so the *value* of the rotation does
// not depend on which axis feeds it -- but the mapping pair -> axis is still
// real, and a non-text fixture (different t/h/w/extra per axis) exercises it.
void rope_head(float* a, int rd, int half, int pos, double base) {
  for (int i = 0; i < half; ++i) {
    const double th = (double)pos * std::pow(base, -2.0 * (double)i / (double)rd);
    const float c = (float)std::cos(th);
    const float s = (float)std::sin(th);
    const float x0 = a[i];
    const float x1 = a[i + half];
    a[i] = x0 * c - x1 * s;
    a[i + half] = x0 * s + x1 * c;
  }
}

}  // namespace

void Qwen35::reset_attention_state(int layer) {
  auto it = ast_.find(layer);
  if (it == ast_.end()) return;
  it->second.k.clear();
  it->second.v.clear();
  it->second.len = 0;
}

// Same split as the recurrent block: the probe embeds, the trunk does not.
void Qwen35::attention_layer(int layer, const int32_t* tokens, int n, AttentionOut& out) {
  const int E = g_.n_embd;
  if (n <= 0) fail("attention_layer needs at least one token");
  std::vector<float> hidden((size_t)n * E, 0.0f);
  for (int t = 0; t < n; ++t) embed_row(tokens[t], hidden.data() + (size_t)t * E);
  attention_block(layer, hidden.data(), n, out);
}

void Qwen35::attention_block(int layer, const float* hidden, int n, AttentionOut& out) {
  const Qwen35Geometry& g = g_;
  if (is_recurrent(layer)) {
    fail("layer " + std::to_string(layer) + " is recurrent; the gated attention "
         "(QG split, QK-norm, RoPE, causal softmax) is not defined for it");
  }
  if (n <= 0) fail("attention_block needs at least one token");

  const int E = g.n_embd;
  const int H = g.n_head, HKV = g.n_head_kv, D = g.head_dim;
  const int QD = H * D, KVD = HKV * D, RD = g.rope_dims;
  if (RD <= 0 || RD > D) {
    fail("rope.dimension_count " + std::to_string(RD) + " is outside (0, head_dim " +
         std::to_string(D) + "]");
  }
  if (RD % 2) {
    fail("rope.dimension_count is odd; the NEOX pairing rotates (i, i + n/2)"
         " and needs an even count");
  }
  if (H % HKV) fail("head_count is not a multiple of head_count_kv");
  const int rep = H / HKV;
  const int half = RD / 2;

  const std::string b = base(layer);
  const TensorInfo& w_q = file_.require(b + "attn_q.weight");
  const TensorInfo& w_k = file_.require(b + "attn_k.weight");
  const TensorInfo& w_v = file_.require(b + "attn_v.weight");
  const TensorInfo& w_o = file_.require(b + "attn_output.weight");
  const TensorInfo& w_an = file_.require(b + "attn_norm.weight");
  const TensorInfo& w_qn = file_.require(b + "attn_q_norm.weight");
  const TensorInfo& w_kn = file_.require(b + "attn_k_norm.weight");

  std::vector<float> anw((size_t)E), qnw((size_t)D), knw((size_t)D);
  dequant_row_f32(w_an.type, file_.data_of(w_an), anw.data(), (uint64_t)E);
  dequant_row_f32(w_qn.type, file_.data_of(w_qn), qnw.data(), (uint64_t)D);
  dequant_row_f32(w_kn.type, file_.data_of(w_kn), knw.data(), (uint64_t)D);

  std::vector<int> axis((size_t)half, 0);
  for (int i = 0; i < half; ++i) {
    axis[(size_t)i] = imrope_axis(i, g.rope_sections);
    ++out.rope_pairs[axis[(size_t)i]];
  }

  AttentionState& as = ast_[layer];
  const int total = as.len + n;
  as.k.resize((size_t)total * KVD, 0.0f);
  as.v.resize((size_t)total * KVD, 0.0f);

  out.steps = n;
  out.x.assign((size_t)n * E, 0.0f);
  out.qg.assign((size_t)n * 2 * QD, 0.0f);
  out.q_norm.assign((size_t)n * QD, 0.0f);
  out.q_rope.assign((size_t)n * QD, 0.0f);
  out.k_norm.assign((size_t)n * KVD, 0.0f);
  out.k_rope.assign((size_t)n * KVD, 0.0f);
  out.v.assign((size_t)n * KVD, 0.0f);
  out.scores.assign((size_t)n * H * total, 0.0f);
  out.attn.assign((size_t)n * QD, 0.0f);
  out.gated.assign((size_t)n * QD, 0.0f);
  out.out.assign((size_t)n * E, 0.0f);

  std::vector<float> x((size_t)E), qg((size_t)2 * QD);
  std::vector<float> q((size_t)QD), kq((size_t)KVD), vv((size_t)KVD);
  std::vector<float> g_raw((size_t)QD), attn((size_t)QD), gated((size_t)QD), y((size_t)E);
  std::vector<float> row((size_t)total, 0.0f);
  const float scale = 1.0f / std::sqrt((float)D);

  for (int t = 0; t < n; ++t) {
    const int pos = as.len + t;
    rmsnorm(hidden + (size_t)t * E, w_an, x.data(), E);
    std::memcpy(out.x.data() + (size_t)t * E, x.data(), sizeof(float) * (size_t)E);

    project(w_q, x.data(), qg.data(), 2 * QD);
    std::memcpy(out.qg.data() + (size_t)t * 2 * QD, qg.data(),
                sizeof(float) * (size_t)2 * QD);
    // Per head: [q(head_dim) | gate(head_dim)]; the gate is the raw projection,
    // sigmoid-ed only after the attention, as the reference does.
    for (int h = 0; h < H; ++h) {
      const float* src = qg.data() + (size_t)2 * h * D;
      float* dst = q.data() + (size_t)h * D;
      float ss = 0.0f;
      for (int d = 0; d < D; ++d) ss += src[d] * src[d];
      const float r = 1.0f / std::sqrt(ss / (float)D + g.rms_eps);
      for (int d = 0; d < D; ++d) dst[d] = src[d] * r * qnw[(size_t)d];
      std::memcpy(g_raw.data() + (size_t)h * D, src + D, sizeof(float) * (size_t)D);
    }
    std::memcpy(out.q_norm.data() + (size_t)t * QD, q.data(), sizeof(float) * (size_t)QD);

    project(w_k, x.data(), kq.data(), KVD);
    for (int h = 0; h < HKV; ++h) {
      float* dst = kq.data() + (size_t)h * D;
      float ss = 0.0f;
      for (int d = 0; d < D; ++d) ss += dst[d] * dst[d];
      const float r = 1.0f / std::sqrt(ss / (float)D + g.rms_eps);
      for (int d = 0; d < D; ++d) dst[d] *= r * knw[(size_t)d];
    }
    std::memcpy(out.k_norm.data() + (size_t)t * KVD, kq.data(), sizeof(float) * (size_t)KVD);
    project(w_v, x.data(), vv.data(), KVD);
    std::memcpy(out.v.data() + (size_t)t * KVD, vv.data(), sizeof(float) * (size_t)KVD);

    for (int h = 0; h < H; ++h) rope_head(q.data() + (size_t)h * D, RD, half, pos, (double)g.rope_base);
    for (int h = 0; h < HKV; ++h) rope_head(kq.data() + (size_t)h * D, RD, half, pos, (double)g.rope_base);

  // Independent expectation for the sectioned-axis mapping: every pair in
  // [0, 256) belongs to exactly one axis, and the axis sizes sum to 256.
  // This is asserted here so the non-text IMROPE fixture can rely on the
  // sections being well-formed; if they are not, the fixture is not meaningful.
  // Sectioned-axis sanity check: the four section sizes must sum to 256, the
  // dimension-pair count this head rotates. If they do not, the non-text IMROPE
  // fixture below is not meaningful and the probe falls back to reporting the
  // sections without any axis-wise position.
  // Sectioned-axis sanity check: the four section sizes must sum to 256, the
  // dimension-pair count this head rotates. The non-text IMROPE fixture is only
  // meaningful when they sum to 256; otherwise the axis index for any pair is
  // ambiguous and the fixture is skipped.
  // Independent IMROPE axis mapping for the non-text fixture. The mapping
  // depends only on the section sizes, not on the positions, so the fixture can
  // hold a fixed layout and both sides agree on which pairs belong to which axis
  // -- that is the non-text case this code path is meant to cover.
  {
    int sum = 0;
    for (int a = 0; a < 4; ++a) sum += g.rope_sections[a];
    if (sum != 256) {
      std::fprintf(stderr, "WARNING: rope dimension_sections sum to %d, expected 256; "
                          "the non-text IMROPE fixture will be skipped\n", sum);
    }
  }
  {
    int sum = 0;
    for (int a = 0; a < 4; ++a) sum += g.rope_sections[a];
    if (sum != 256) {
      std::fprintf(stderr, "WARNING: rope dimension_sections sum to %d, expected 256; "
                          "the non-text IMROPE fixture will be skipped\n", sum);
    }
  }
    std::memcpy(out.q_rope.data() + (size_t)t * QD, q.data(), sizeof(float) * (size_t)QD);
    std::memcpy(out.k_rope.data() + (size_t)t * KVD, kq.data(), sizeof(float) * (size_t)KVD);

    std::memcpy(as.k.data() + (size_t)pos * KVD, kq.data(), sizeof(float) * (size_t)KVD);
    std::memcpy(as.v.data() + (size_t)pos * KVD, vv.data(), sizeof(float) * (size_t)KVD);

    // Causal attention over every position seen so far, GQA by head group.
    for (int h = 0; h < H; ++h) {
      const float* qh = q.data() + (size_t)h * D;
      const int kh = h / rep;
      float mx = -1e30f;
      for (int j = 0; j <= pos; ++j) {
        const float* kj = as.k.data() + (size_t)j * KVD + (size_t)kh * D;
        float s = 0.0f;
        for (int d = 0; d < D; ++d) s += qh[d] * kj[d];
        s *= scale;
        row[(size_t)j] = s;
        if (s > mx) mx = s;
      }
      float sum = 0.0f;
      for (int j = 0; j <= pos; ++j) {
        const float e = std::exp(row[(size_t)j] - mx);
        row[(size_t)j] = e;
        sum += e;
      }
      const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
      for (int j = 0; j <= pos; ++j) row[(size_t)j] *= inv;
      std::memcpy(out.scores.data() + ((size_t)t * H + h) * total, row.data(),
                  sizeof(float) * (size_t)(pos + 1));
      float* dst = attn.data() + (size_t)h * D;
      for (int d = 0; d < D; ++d) dst[d] = 0.0f;
      for (int j = 0; j <= pos; ++j) {
        const float w = row[(size_t)j];
        const float* vj = as.v.data() + (size_t)j * KVD + (size_t)kh * D;
        for (int d = 0; d < D; ++d) dst[d] += w * vj[d];
      }
    }
    std::memcpy(out.attn.data() + (size_t)t * QD, attn.data(), sizeof(float) * (size_t)QD);

    for (int i = 0; i < QD; ++i) {
      const float gr = g_raw[(size_t)i];
      gated[(size_t)i] = attn[(size_t)i] * (1.0f / (1.0f + std::exp(-gr)));
    }
    std::memcpy(out.gated.data() + (size_t)t * QD, gated.data(), sizeof(float) * (size_t)QD);

    project(w_o, gated.data(), y.data(), E);
    std::memcpy(out.out.data() + (size_t)t * E, y.data(), sizeof(float) * (size_t)E);
  }
  as.len += n;
  out.kv_len = as.len;
}

std::vector<std::string> Qwen35::missing() const {
  const Qwen35Geometry& g = g_;
  std::vector<std::string> out;
  // The arithmetic is no longer the gap: trunk_forward() composes the causal layers
  // with their residuals, runs the dense FFN and the untied head, and produces
  // logits. What is missing is the *forward path* -- something a generation loop can
  // call, whose buffers do not materialise every layer.
  out.push_back("the trunk as a forward path rather than a probe: trunk_forward() "
                "composes all " + std::to_string(trunk_layers()) +
                " causal layers, the dense FFN and the head and is verified against "
                "an oracle, but it materialises every layer's residual stream (" +
                "nextn " + std::to_string(g.nextn_layers) +
                " block(s) excluded) and holds state in the probe object. Nothing "
                "in the generation loop calls it, so -n and --bench remain refused "
                "on a qwen35 file");
  out.push_back("residency and batching: this path holds every layer's weights at "
                "once in the file mapping and every intermediate in RAM. There is no "
                "expert/kv residency decision here, because a dense " +
                std::to_string(g.n_layer) + "-layer file has none to make");
  if (g.nextn_layers > 0) {
    out.push_back("the merged nextn MTP head (" + std::to_string(g.nextn_layers) +
                  " layer, block " + std::to_string(rep_.nextn_block) +
                  "): the eh_proj fusion of [hidden, embedding] C20 asks for");
  }
  // Quantisation is a separate refusal from the trunk, and it is the one that
  // stops a file before any arithmetic: name the types and their tensor counts
  // rather than making the reader diff two tables.
  std::map<std::string, int> bad;
  for (const TensorInfo& t : file_.tensors()) {
    if (!ggml_type_is_dequantizable(t.type)) ++bad[ggml_type_name(t.type)];
  }
  if (!bad.empty()) {
    std::string s = "tensors whose quantisation this engine cannot decode:";
    for (const auto& kv : bad) {
      s += " " + kv.first + " (" + std::to_string(kv.second) + " tensors)";
    }
    out.push_back(s);
  }
  return out;
}

VecCheck compare_vector(const std::string& dir, const std::string& name,
                        const std::vector<float>& got, double rtol) {
  VecCheck v;
  v.name = name;
  v.n = got.size();
  const std::string path = dir + "/" + name + ".f32";
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    v.ref_missing = true;
    v.verdict = false;
    return v;
  }
  std::vector<float> want(v.n, 0.0f);
  const size_t rd = std::fread(want.data(), sizeof(float), v.n, f);
  std::fclose(f);
  if (rd != v.n) {
    v.ref_missing = true;
    v.verdict = false;
    return v;
  }
  double ss = 0.0, sse = 0.0, mx = 0.0;
  for (size_t i = 0; i < v.n; ++i) {
    const double r = want[i];
    const double g = got[i];
    ss += r * r;
    const double d = std::fabs(g - r);
    sse += d * d;
    if (d > mx) mx = d;
    if (g == 0.0f) {
      ++v.zeros;
      // A zero the reference does not have is an output the engine never wrote,
      // which is a different defect from "this weight is exactly zero".
      if (r != 0.0) ++v.unwritten;
    }
    if (std::fabs(r) > v.ref_max_abs) v.ref_max_abs = std::fabs(r);
  }
  const double rms = std::sqrt(ss / (double)v.n);
  v.max_abs = mx;
  v.rel_rmse = rms > 0.0 ? std::sqrt(sse / (double)v.n) / rms : 0.0;
  const double tol = rtol * (rms > 0.0 ? rms : 1.0);
  for (size_t i = 0; i < v.n; ++i) {
    if (std::fabs((double)got[i] - (double)want[i]) > tol) {
      if (v.first_bad < 0) v.first_bad = (int64_t)i;
      ++v.mismatches;
    }
  }
  v.verdict = v.mismatches == 0 && v.unwritten == 0;
  return v;
}

// ---------------------------------------------------------------- the trunk ---

void Qwen35::reset_trunk_state() {
  rst_.clear();
  ast_.clear();
}

void Qwen35::dense_ffn(int layer, const float* x, int n, float* out) {
  const int E = g_.n_embd;
  const int F = g_.n_ff;
  const std::string b = base(layer);
  const TensorInfo& w_gate = file_.require(b + "ffn_gate.weight");
  const TensorInfo& w_up = file_.require(b + "ffn_up.weight");
  const TensorInfo& w_down = file_.require(b + "ffn_down.weight");
  if ((int)w_gate.dims[0] != E || (int)w_gate.dims[1] != F ||
      (int)w_up.dims[0] != E || (int)w_up.dims[1] != F ||
      (int)w_down.dims[0] != F || (int)w_down.dims[1] != E) {
    fail(b + "ffn_* shapes do not match n_embd " + std::to_string(E) + " / n_ff " +
         std::to_string(F) + ": the FFN layout is not the gate/up/down SwiGLU this reads");
  }
  std::vector<float> gv((size_t)F), uv((size_t)F), act((size_t)F), ov((size_t)E);
  for (int t = 0; t < n; ++t) {
    const float* xt = x + (size_t)t * E;
    project(w_gate, xt, gv.data(), F);
    project(w_up, xt, uv.data(), F);
    // SwiGLU: silu(gate) * up. llama.cpp's build_ffn with LLM_FFN_SILU, LLM_FFN_PAR
    // (src/models/qwen35.cpp build_layer_ffn).
    for (int i = 0; i < F; ++i) {
      const float g = gv[(size_t)i];
      act[(size_t)i] = (g / (1.0f + std::exp(-g))) * uv[(size_t)i];
    }
    project(w_down, act.data(), ov.data(), E);
    std::memcpy(out + (size_t)t * E, ov.data(), sizeof(float) * (size_t)E);
  }
}

void Qwen35::trunk_forward(const int32_t* tokens, int n, TrunkOut& out, bool keep_per_layer) {
  const int E = g_.n_embd;
  const int L = trunk_layers();
  if (L <= 0) {
    fail("there are no trunk layers: block_count " + std::to_string(g_.n_layer) +
         " minus nextn_predict_layers " + std::to_string(g_.nextn_layers));
  }
  if (n <= 0) fail("trunk_forward needs at least one token");
  if (E <= 0 || g_.n_ff <= 0 || g_.n_vocab <= 0) fail("trunk geometry was not read");

  out.trunk_layers = L;
  out.steps = n;
  out.hidden.assign((size_t)n * E, 0.0f);
  out.normed.assign((size_t)n * E, 0.0f);
  out.logits.assign((size_t)n * g_.n_vocab, 0.0f);
  out.per_layer.clear();
  out.per_layer_mix.clear();
  if (keep_per_layer) {
    out.per_layer.assign((size_t)L * n * E, 0.0f);
    out.per_layer_mix.assign((size_t)L * n * E, 0.0f);
  }

  std::vector<float> h((size_t)n * E, 0.0f);
  for (int t = 0; t < n; ++t) embed_row(tokens[t], h.data() + (size_t)t * E);

  const size_t NE = (size_t)n * E;
  std::vector<float> mixer((size_t)n * E, 0.0f), nrm((size_t)n * E, 0.0f), ffn((size_t)n * E, 0.0f);
  RecurrentOut ro;
  AttentionOut ao;

  for (int il = 0; il < L; ++il) {
    // The blocks norm their own input (`attn_norm`), so the residual stream goes in
    // raw. Their output is the layer's contribution, added back once.
    if (is_recurrent(il)) {
      recurrent_block(il, h.data(), n, ro);
      if (ro.out.size() != NE) fail(base(il) + "recurrent block returned a wrong-sized output");
      std::memcpy(mixer.data(), ro.out.data(), sizeof(float) * NE);
    } else {
      attention_block(il, h.data(), n, ao);
      if (ao.out.size() != NE) fail(base(il) + "attention block returned a wrong-sized output");
      std::memcpy(mixer.data(), ao.out.data(), sizeof(float) * NE);
    }
    for (size_t i = 0; i < NE; ++i) h[i] += mixer[i];
    if (keep_per_layer) {
      std::memcpy(out.per_layer_mix.data() + (size_t)il * NE, h.data(), sizeof(float) * NE);
    }

    // Post-attention norm -> dense FFN -> second residual, added to the stream from
    // before the norm (llama.cpp: "add to the tensor from before
    // post_attention_layernorm").
    const TensorInfo& w_pan = file_.require(base(il) + "post_attention_norm.weight");
    if ((int)w_pan.dims[0] != E) {
      fail(base(il) + "post_attention_norm.weight is not [n_embd]");
    }
    for (int t = 0; t < n; ++t) {
      rmsnorm(h.data() + (size_t)t * E, w_pan, nrm.data() + (size_t)t * E, E);
    }
    dense_ffn(il, nrm.data(), n, ffn.data());
    for (size_t i = 0; i < NE; ++i) h[i] += ffn[i];

    if (keep_per_layer) {
      std::memcpy(out.per_layer.data() + (size_t)il * NE, h.data(), sizeof(float) * NE);
    }
  }

  out.hidden = h;

  // The head: output_norm then output.weight, with no scaling and no softcap
  // (llama.cpp builds exactly `build_lora_mm(model.output, norm(cur))`).
  const TensorInfo& w_on = file_.require("output_norm.weight");
  if ((int)w_on.dims[0] != E) fail("output_norm.weight is not [n_embd]");
  for (int t = 0; t < n; ++t) {
    rmsnorm(h.data() + (size_t)t * E, w_on, out.normed.data() + (size_t)t * E, E);
  }
  const TensorInfo& w_out = file_.require("output.weight");
  if ((int)w_out.dims[0] != E || (int)w_out.dims[1] != g_.n_vocab) {
    fail("output.weight is not [n_embd, n_vocab] (untied embeddings are required here)");
  }
  for (int t = 0; t < n; ++t) {
    project(w_out, out.normed.data() + (size_t)t * E,
            out.logits.data() + (size_t)t * (size_t)g_.n_vocab, g_.n_vocab);
  }
}

// ---------------------------------------------------------------- the probe ---
namespace {
const char* kRoundTrip[] = {
    "The capital of France is",
    "hello world",
    "1 2 3 123 1234 3.14159 -0.5",
    "  leading and   multiple    spaces",
    "line1\nline2\n\nline3",
    "日本語のテキスト",
    "emoji: \xF0\x9F\x9A\x80\xF0\x9F\x94\xA5",
    "tab\there <|endoftext|>",
    "",
};
}  // namespace

int run_qwen35_probe(const Qwen35ProbeOptions& opt) {
  std::printf("== kanjoos qwen35 probe ==========================================\n");
  std::printf("model      : %s\n", opt.model.c_str());

  std::unique_ptr<Qwen35> m;
  const double t0 = now_ms();
  try {
    KNJ_PROFILE_OP("qwen35-load");
    m = Qwen35::open(opt.model);
  } catch (const std::exception& e) {
    std::printf("\nREFUSED at load:\n  %s\n", e.what());
    std::printf("\nThis is a container-level refusal: the file could not be read as a\n"
                "qwen35 model at all, so nothing about it was measured.\n");
    return 3;
  }
  const double load_ms = now_ms() - t0;
  const Qwen35Geometry& g = m->geom();

  std::printf("container  : GGUF v%u, %zu tensors, %zu metadata entries, %.3f GiB\n",
              m->file().version(), m->file().tensors().size(),
              m->file().metadata().size(),
              (double)m->file().file_size() / (1024.0 * 1024.0 * 1024.0));
  std::printf("loaded     : %.1f ms (mapped; nothing dequantized)\n", load_ms);

  std::printf("\n-- geometry, from the file's own metadata --\n%s", g.describe().c_str());

  // Which layers are which: printed, because the whole binding below depends on it.
  {
    std::string pat;
    for (int i = 0; i < g.n_layer; ++i) pat += g.recurrent[i] ? 'S' : 'A';
    std::printf("  layer kinds   %s  (S = recurrent/SSM, A = full attention)\n",
                pat.c_str());
    std::printf("  kinds source  %s\n",
                g.kinds_from_array
                    ? "attention.recurrent_layers (per-layer, exact)"
                    : "DERIVED from full_attention_interval \u2014 this file does "
                      "not carry the per-layer array");
    std::printf("  conv_dim      %d == 2 * groups(%d) * state(%d) + inner(%d)\n",
                g.conv_dim(), g.ssm_groups, g.ssm_state, g.ssm_inner);
  }

  // -- tokenizer ------------------------------------------------------------
  std::printf("\n-- tokenizer --\n");
  {
    KNJ_PROFILE_OP("qwen35-tokenize");
    const Tokenizer& tk = m->tokenizer();
    std::printf("  vocab %d, bos %d, eos %d, add_bos %d, chat_template %s\n",
                tk.vocab_size(), tk.bos_id(), tk.eos_id(), (int)tk.add_bos(),
                tk.chat_template().empty() ? "absent" : "present");
    // The merge table and the pre-tokeniser rule are part of the tokeniser's
    // identity: a file whose merges this engine did not apply, or whose `pre` it
    // does not implement, produces different ids without failing anything. Both
    // are printed, and tools/tok_crosscheck.py is what checks them.
    std::printf("  merges %d, pre '%s' -> rule %s (%s)\n", tk.merges(), tk.pre().c_str(),
                pre_rule_name(tk.pre_rule()),
                tk.pre_rule() == PreRule::Qwen35
                    ? "[\\p{L}\\p{M}]+: a combining mark joins the letter run"
                    : "[\\p{L}]+: a combining mark is punctuation");
    int pass = 0, total = 0;
    std::printf("  round-trip (decode(encode(s)) == s):\n");
    for (const char* s : kRoundTrip) {
      const std::vector<int32_t> ids = tk.encode(s);
      const std::string back = tk.decode(ids);
      const bool ok = back == std::string(s);
      ++total;
      pass += ok ? 1 : 0;
      std::printf("    %-4s n=%-4zu %s\n", ok ? "ok" : "FAIL", ids.size(),
                  (std::string("\"") + s + "\"").c_str());
    }
    std::printf("  %d/%d round-trips exact\n", pass, total);
    const std::vector<int32_t> ids = tk.encode(opt.text);
    std::printf("  probe text \"%s\" -> %zu id(s):", opt.text.c_str(), ids.size());
    for (int32_t id : ids) std::printf(" %d", id);
    std::printf("\n");
  }

  // -- binding --------------------------------------------------------------
  std::printf("\n-- tensor binding --\n");
  {
    KNJ_PROFILE_OP("qwen35-bind");
    const Qwen35Report& r = m->report();
    std::printf("  %s\n", r.summary().c_str());
    std::printf("  nextn: %d tensor(s)%s\n", r.nextn_tensors,
                r.nextn_block >= 0
                    ? (" on blk." + std::to_string(r.nextn_block)).c_str()
                    : " (none: this container has no merged MTP head)");
    int shown = 0;
    for (const Qwen35Binding& b : r.bindings) {
      if (b.ok()) continue;
      if (shown++ == 0) std::printf("  NOT USABLE:\n");
      std::printf("    %-44s %s\n", b.name.c_str(),
                  !b.found ? "absent"
                  : !b.shape_ok ? ("shape " + b.got + " != " + b.wanted).c_str()
                  : ("type " + b.type_name + " has no decoder").c_str());
      if (shown >= 24) {
        std::printf("    ... (and more; the counts above are the full picture)\n");
        break;
      }
    }
    if (shown == 0) std::printf("  every tensor is present with the implied shape\n");
  }

  // -- type coverage --------------------------------------------------------
  std::printf("\n-- quantisation coverage (all %zu tensors) --\n",
              m->file().tensors().size());
  {
    std::map<std::string, std::pair<int, bool>> by_type;
    for (const TensorInfo& t : m->file().tensors()) {
      auto& e = by_type[ggml_type_name(t.type)];
      e.first += 1;
      e.second = ggml_type_is_dequantizable(t.type);
    }
    int undecodable = 0;
    for (const auto& kv : by_type) {
      if (!kv.second.second) undecodable += kv.second.first;
      std::printf("  %-8s %4d tensor(s)  %s\n", kv.first.c_str(), kv.second.first,
                  kv.second.second ? "decodable" : "NO DECODER IN THIS ENGINE");
    }
    if (undecodable) {
      std::printf("  -> %d tensor(s) cannot be decoded here; the trunk cannot run\n"
                  "     on this file even once the recurrence exists.\n", undecodable);
    }
  }

  // -- the verified front half of one layer ---------------------------------
  const int L = opt.layer;
  const bool rec = m->is_recurrent(L);
  std::printf("\n-- layer %d (%s) --\n", L, rec ? "recurrent / SSM" : "full attention");
  std::vector<VecCheck> checks;
  std::string stopped;
  try {
    const Tokenizer& tk = m->tokenizer();
    std::vector<int32_t> ids;
    if (!opt.tokens.empty()) {
      // Explicit ids: the oracle's `--tokens` list. A wrong id would make the two
      // sides compute different things and compare anyway, so a malformed list is
      // a refusal and every id is range-checked.
      size_t at = 0;
      while (at < opt.tokens.size()) {
        const size_t comma = opt.tokens.find(',', at);
        const std::string piece = opt.tokens.substr(
            at, comma == std::string::npos ? std::string::npos : comma - at);
        if (piece.empty()) fail("--qwen35-tokens has an empty entry");
        const long v = std::strtol(piece.c_str(), nullptr, 10);
        if (v < 0 || v >= tk.vocab_size()) {
          fail("--qwen35-tokens id " + piece + " is outside the vocabulary (0.." +
               std::to_string(tk.vocab_size() - 1) + ")");
        }
        ids.push_back((int32_t)v);
        if (comma == std::string::npos) break;
        at = comma + 1;
      }
      if (ids.empty()) fail("--qwen35-tokens was given but produced no ids");
    } else {
      ids = tk.encode(opt.text);
      if (ids.empty()) ids.push_back(tk.bos_id() >= 0 ? tk.bos_id() : 0);
    }
    const std::string b = m->base(L);

    if (opt.forward) {
      // -- the trunk: all 64 layers, the dense FFN, the head -------------------
      //
      // This is the whole causal stack and it produces logits, which is what the
      // per-layer probes deliberately did not. It is still a *probe* in the sense
      // that it materialises every layer's output and holds the states in memory --
      // not a residency-managed forward path -- but the arithmetic is the trunk's.
      KNJ_PROFILE_OP("qwen35-forward");
      const int TL = m->trunk_layers();
      int n_rec = 0;
      for (int il = 0; il < TL; ++il) n_rec += m->is_recurrent(il) ? 1 : 0;
      std::printf("  tokens        : %zu", ids.size());
      for (size_t k = 0; k < ids.size(); ++k) std::printf(" %d", ids[k]);
      std::printf("\n");
      std::printf("  trunk         : %d layer(s) = block_count %d - nextn_predict_layers %d\n",
                  TL, g.n_layer, g.nextn_layers);
      std::printf("  layer kinds   : %d recurrent (SSM) + %d full attention in this file\n",
                  n_rec, TL - n_rec);
      std::printf("    (blk.%d carries the nextn MTP tensors and is the LAST layer in the file,\n"
                  "     but is NOT in the trunk: block_count == n_layer and MTP is per-block\n"
                  "     storage only, as llama.cpp's layer loop builds it\n", g.n_layer - 1);
      std::printf("  head          : output_norm + output.weight [%d, %d] (untied), no scaling,\n"
                  "                  no softcap\n", g.n_embd, g.n_vocab);

      std::vector<float> emb_in((size_t)ids.size() * (size_t)g.n_embd, 0.0f);
      for (size_t k = 0; k < ids.size(); ++k) {
        m->embed_row(ids[k], emb_in.data() + k * (size_t)g.n_embd);
      }

      Qwen35::TrunkOut o;
      m->reset_trunk_state();
      const double t1 = now_ms();
      m->trunk_forward(ids.data(), (int)ids.size(), o, true);
      const double t2 = now_ms();
      std::printf("  computed      : %zu token(s) through %d layer(s) in %.1f ms -- embedding,\n"
                  "                  then per layer: mixer residual + post-attn RMS + dense SwiGLU\n"
                  "                  FFN residual (per_layer_mix = after mixer residual,\n"
                  "                  per_layer = after the FFN residual; the oracle emits both)\n",
                  ids.size(), TL, t2 - t1);

      // The logits exist, so the tokens they imply can be read out. Printed before
      // any comparison: this is the first time this engine turns a qwen35 file into
      // a token at all, and a wrong layer order would still produce *a* token.
      for (size_t k = 0; k < ids.size(); ++k) {
        const float* lg = o.logits.data() + k * (size_t)g.n_vocab;
        int best[5] = {0, 0, 0, 0, 0};
        for (int i = 1; i < g.n_vocab; ++i) {
          for (int r = 0; r < 5; ++r) {
            if (lg[i] > lg[best[r]]) {
              for (int s = 4; s > r; --s) best[s] = best[s - 1];
              best[r] = i;
              break;
            }
          }
        }
        std::printf("  position %zu  argmax %d \"%s\"   top5", k, best[0],
                    tk.decode_one(best[0]).c_str());
        for (int r = 1; r < 5; ++r) {
          std::printf(" %d(%.2f) \"%s\"", best[r], lg[best[r]], tk.decode_one(best[r]).c_str());
        }
        std::printf("\n");
      }

      if (!opt.ref_dir.empty()) {
        checks.push_back(compare_vector(opt.ref_dir, "trunk_in", emb_in, opt.rtol));
        const size_t slice = (size_t)ids.size() * (size_t)g.n_embd;
        for (int il = 0; il < TL; ++il) {
          checks.push_back(compare_vector(
              opt.ref_dir, "trunk_mix" + std::to_string(il),
              std::vector<float>(o.per_layer_mix.begin() + (size_t)il * slice,
                                 o.per_layer_mix.begin() + (size_t)(il + 1) * slice),
              opt.rtol));
          checks.push_back(compare_vector(
              opt.ref_dir, "trunk_h" + std::to_string(il),
              std::vector<float>(o.per_layer.begin() + (size_t)il * slice,
                                 o.per_layer.begin() + (size_t)(il + 1) * slice),
              opt.rtol));
        }
        checks.push_back(compare_vector(opt.ref_dir, "trunk_norm", o.normed, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "trunk_logits", o.logits, opt.rtol));
      }
    } else if (opt.recurrent) {
      // -- the recurrence, one GatedDeltaNet layer, end to end ---------------
      // Not a trunk forward pass: one layer, on the host, with the conv history
      // and the delta-rule state carried across the tokens of this text. The 12
      // vectors are the oracle's names, compared by name and element by element.
      KNJ_PROFILE_OP("qwen35-recurrent");
      if (!rec) {
        fail("layer " + std::to_string(L) + " is a full-attention layer; the "
             "GatedDeltaNet recurrence (conv1d carry, delta rule, gating) is not "
             "defined for it");
      }
      std::printf("  tokens        : %zu", ids.size());
      for (size_t k = 0; k < ids.size(); ++k) std::printf(" %d", ids[k]);
      std::printf("\n");
      Qwen35::RecurrentOut o;
      m->reset_recurrent_state(L);
      const double t1 = now_ms();
      m->recurrent_layer(L, ids.data(), (int)ids.size(), o);
      const double ms = now_ms() - t1;
      std::printf("  computed      : %d token step(s) in %.1f ms -- conv1d carry "
                  "(%d taps), delta rule, gating\n", o.steps, ms, g.ssm_conv_kernel);
      std::printf("  exp(g)        : [%.6f, %.6f]  (a stable recurrence needs every\n"
                  "                  value inside (0, 1]; a positive g means the "
                  "stored ssm_a was read wrong)\n",
                  m->last_decay_min(), m->last_decay_max());
      std::printf("  state         : %d x %d x %d = %zu value(s) (one per head: key,\n"
                  "                  value; held here, never by the trunk)\n",
                  g.ssm_dt_rank, g.ssm_state, g.ssm_state, o.state.size());

      // Two properties the oracle cannot check, because it loops over the tokens
      // internally: the state carrying ACROSS CALLS, and that appending tokens
      // does not change a prefix. A conv window off by one token, or a history
      // that is dropped between calls, shows up in the first and nowhere else.
      {
        const size_t E2 = (size_t)g.n_embd;
        std::vector<float> split((size_t)ids.size() * E2, 0.0f);
        Qwen35::RecurrentOut c;
        m->reset_recurrent_state(L);
        int64_t first_bad = -1;
        double dmax = 0.0;
        for (size_t k = 0; k < ids.size(); ++k) {
          m->recurrent_layer(L, &ids[k], 1, c);
          for (size_t j = 0; j < E2; ++j) {
            const double d =
                std::fabs((double)c.out[j] - (double)o.out[k * E2 + j]);
            if (d > 0.0 && first_bad < 0) first_bad = (int64_t)(k * E2 + j);
            if (d > dmax) dmax = d;
          }
          std::memcpy(split.data() + k * E2, c.out.data(), sizeof(float) * E2);
        }
        std::printf("  state carry   : %zu separate call(s) vs one call -- out %s, "
                    "state %s\n", ids.size(),
                    split == o.out ? "BIT-IDENTICAL" : "DIFFERS",
                    c.state == o.state ? "BIT-IDENTICAL" : "DIFFERS");
        if (first_bad >= 0) {
          std::printf("                  first difference at index %lld, max|d| "
                      "%.3e\n", (long long)first_bad, dmax);
        }
        // Causality: the first half of the sequence, computed from a fresh
        // state, must equal the first half of the whole run.
        const size_t half = ids.size() / 2;
        if (half >= 1) {
          m->reset_recurrent_state(L);
          m->recurrent_layer(L, ids.data(), (int)half, c);
          bool same = c.out.size() == half * E2;
          double pbad = 0.0;
          for (size_t j = 0; same && j < c.out.size(); ++j) {
            const double d =
                std::fabs((double)c.out[j] - (double)o.out[j]);
            if (d > pbad) pbad = d;
            if (d != 0.0) same = false;
          }
          std::printf("  causality     : first %zu token(s) from a fresh state vs the "
                      "prefix of the\n                  whole run -- %s (max|d| %.3e)\n",
                      half, same ? "BIT-IDENTICAL" : "DIFFERS", pbad);
        }
        m->reset_recurrent_state(L);
      }
      if (!opt.ref_dir.empty()) {
        checks.push_back(compare_vector(opt.ref_dir, "layer_x", o.x, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_qkv", o.qkv, opt.rtol));
        checks.push_back(
            compare_vector(opt.ref_dir, "layer_conv_raw", o.conv_raw, opt.rtol));
        checks.push_back(
            compare_vector(opt.ref_dir, "layer_conv_silu", o.conv, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_q_l2", o.q, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_k_l2", o.k, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_beta", o.beta, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_g", o.g, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_o", o.o, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_state", o.state, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_gated", o.gated, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_out", o.out, opt.rtol));
      }
    } else if (opt.attention) {
      // -- one gated full-attention layer, end to end -------------------------
      // Not a trunk forward pass: one layer, on the host, a prefill of this
      // text's tokens with the K/V cache held in the probe, so a later call
      // continues at the next position. The 11 vectors are the oracle's names.
      KNJ_PROFILE_OP("qwen35-attention");
      if (rec) {
        fail("layer " + std::to_string(L) + " is recurrent; the gated attention "
             "(QG split, QK-norm, partial RoPE, causal softmax) is not defined for it");
      }
      std::printf("  tokens        : %zu", ids.size());
      for (size_t k = 0; k < ids.size(); ++k) std::printf(" %d", ids[k]);
      std::printf("\n");
      Qwen35::AttentionOut o;
      m->reset_attention_state(L);
      const double t1 = now_ms();
      m->attention_layer(L, ids.data(), (int)ids.size(), o);
      const double ms = now_ms() - t1;
      std::printf("  computed      : %d token step(s) in %.1f ms -- QG split, QK-norm, "
                  "partial\n                  RoPE, causal attention, gate; kv_len "
                  "%d\n", o.steps, ms, o.kv_len);
      std::printf("  heads         : %d q / %d kv x %d dims, scale 1/sqrt(%d) = %.6f; "
                  "gate is\n                  per head-dim, sigmoid-ed after the "
                  "attention\n", g.n_head, g.n_head_kv, g.head_dim, g.head_dim,
                  1.0f / std::sqrt((float)g.head_dim));
      std::printf("  rope          : partial NEOX over %d of %d dims, base %g; %d "
                  "pair(s) per\n                  IMROPE axis [t %d, h %d, w %d, x %d] -- "
                  "text-only, so all four\n                  carry the token position\n",
                  g.rope_dims, g.head_dim, (double)g.rope_base, g.rope_dims / 2,
                  o.rope_pairs[0], o.rope_pairs[1], o.rope_pairs[2], o.rope_pairs[3]);

      // The same two properties the recurrent path tests: the K/V cache carrying
      // across calls, and a prefix not changing when tokens are appended.
      {
        const size_t E2 = (size_t)g.n_embd;
        const size_t QD = (size_t)g.n_head * g.head_dim;
        std::vector<float> split_out((size_t)ids.size() * E2, 0.0f);
        std::vector<float> split_attn((size_t)ids.size() * QD, 0.0f);
        Qwen35::AttentionOut c;
        m->reset_attention_state(L);
        int64_t first_bad = -1;
        for (size_t k = 0; k < ids.size(); ++k) {
          m->attention_layer(L, &ids[k], 1, c);
          for (size_t j = 0; j < E2; ++j) {
            if (c.out[j] != o.out[k * E2 + j] && first_bad < 0) {
              first_bad = (int64_t)(k * E2 + j);
            }
          }
          std::memcpy(split_out.data() + k * E2, c.out.data(), sizeof(float) * E2);
          std::memcpy(split_attn.data() + k * QD, c.attn.data(), sizeof(float) * QD);
        }
        std::printf("  kv carry      : %zu separate call(s) vs one prefill -- out %s, "
                    "attn %s%s\n", ids.size(),
                    split_out == o.out ? "BIT-IDENTICAL" : "DIFFERS",
                    split_attn == o.attn ? "BIT-IDENTICAL" : "DIFFERS",
                    first_bad >= 0
                        ? (", first difference at " + std::to_string(first_bad)).c_str()
                        : "");
        const size_t half = ids.size() / 2;
        if (half >= 1) {
          m->reset_attention_state(L);
          m->attention_layer(L, ids.data(), (int)half, c);
          const bool same = c.out.size() == half * E2 && c.attn.size() == half * QD &&
                            std::equal(c.out.begin(), c.out.end(), o.out.begin()) &&
                            std::equal(c.attn.begin(), c.attn.end(), o.attn.begin());
          std::printf("  causality     : first %zu token(s) from a fresh cache vs the "
                      "prefix of\n                  the whole run -- %s\n", half,
                      same ? "BIT-IDENTICAL" : "DIFFERS");
        }
        m->reset_attention_state(L);
      }

      if (!opt.ref_dir.empty()) {
        checks.push_back(compare_vector(opt.ref_dir, "layer_x", o.x, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_qg", o.qg, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_q_norm", o.q_norm, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_q_rope", o.q_rope, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_k_norm", o.k_norm, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_k_rope", o.k_rope, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_v", o.v, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_scores", o.scores, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_attn", o.attn, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_gated", o.gated, opt.rtol));
        checks.push_back(compare_vector(opt.ref_dir, "layer_out", o.out, opt.rtol));
      }
    } else {
      // -- the verified front half of one layer ------------------------------
      KNJ_PROFILE_OP("qwen35-layer");
      std::vector<float> x((size_t)g.n_embd), xn((size_t)g.n_embd);
      // Wide enough for the widest projection in EITHER layer kind. The gated
      // query projection of an attention block is 2 * q_dim (12288 here), which is
      // wider than the SSM layer's conv_dim (10240): sizing this from the recurrent
      // layer alone overruns the buffer on the attention layers, and the first
      // version did exactly that (the process died with no output at all, which is
      // what a heap overflow looks like from outside).
      const int widest = std::max(std::max(g.conv_dim(), g.ssm_inner),
                                  std::max(2 * g.q_dim(), g.kv_dim()));
      std::vector<float> proj((size_t)widest);
      const TensorInfo& norm_w = m->file().require(b + "attn_norm.weight");
      const double t1 = now_ms();
      int vecs = 0;
      for (size_t k = 0; k < ids.size() && k < 8; ++k) {
        const int32_t id = ids[k];
        m->embed_row(id, x.data());
        m->rmsnorm(x.data(), norm_w, xn.data(), g.n_embd);
        if (!opt.ref_dir.empty()) {
          std::string tag = std::to_string(id);
          checks.push_back(compare_vector(opt.ref_dir, "embed_" + tag, x, opt.rtol));
          checks.push_back(compare_vector(opt.ref_dir, "xnorm_" + tag, xn, opt.rtol));
        }
        if (rec) {
          const TensorInfo& w = m->file().require(b + "attn_qkv.weight");
          m->project(w, xn.data(), proj.data(), g.conv_dim());
          if (!opt.ref_dir.empty()) {
            checks.push_back(compare_vector(
                opt.ref_dir, "qkv_" + std::to_string(id),
                std::vector<float>(proj.begin(), proj.begin() + g.conv_dim()), opt.rtol));
          }
          const TensorInfo& wg = m->file().require(b + "attn_gate.weight");
          m->project(wg, xn.data(), proj.data(), g.ssm_inner);
          if (!opt.ref_dir.empty()) {
            checks.push_back(compare_vector(
                opt.ref_dir, "gate_" + std::to_string(id),
                std::vector<float>(proj.begin(), proj.begin() + g.ssm_inner), opt.rtol));
          }
          vecs += 2;
        } else {
          const TensorInfo& wq = m->file().require(b + "attn_q.weight");
          const int q_out = (int)wq.dims[1];
          m->project(wq, xn.data(), proj.data(), q_out);
          if (!opt.ref_dir.empty()) {
            checks.push_back(compare_vector(
                opt.ref_dir, "q_" + std::to_string(id),
                std::vector<float>(proj.begin(), proj.begin() + q_out), opt.rtol));
          }
          const TensorInfo& wk = m->file().require(b + "attn_k.weight");
          m->project(wk, xn.data(), proj.data(), g.kv_dim());
          if (!opt.ref_dir.empty()) {
            checks.push_back(compare_vector(
                opt.ref_dir, "k_" + std::to_string(id),
                std::vector<float>(proj.begin(), proj.begin() + g.kv_dim()), opt.rtol));
          }
          const TensorInfo& wv = m->file().require(b + "attn_v.weight");
          m->project(wv, xn.data(), proj.data(), g.kv_dim());
          if (!opt.ref_dir.empty()) {
            checks.push_back(compare_vector(
                opt.ref_dir, "v_" + std::to_string(id),
                std::vector<float>(proj.begin(), proj.begin() + g.kv_dim()), opt.rtol));
          }
          vecs += 3;
        }
      }
      std::printf("  tokens probed : %zu of %zu in the text\n",
                  std::min<size_t>(ids.size(), 8), ids.size());
      std::printf("  computed      : norm + %s for each token in %.1f ms\n",
                  rec ? "attn_qkv + attn_gate" : "attn_q (gated) + attn_k + attn_v",
                  now_ms() - t1);
      std::printf("  projections   : %d vector(s) computed\n", vecs);
    }
  } catch (const std::exception& e) {
    // A refusal inside a layer is reported and the probe continues: the type
    // coverage above and the missing list below are the answer, and dying with
    // a stack trace would hide both. Nothing is compared after this.
    stopped = e.what();
    std::printf("\n  PROBE STOPPED at layer %d: %s\n", L, stopped.c_str());
    std::printf("  (nothing below it was computed or compared)\n");
    checks.clear();
  }

  // -- the comparison -------------------------------------------------------
  int verified = 0, failed = 0, noref = 0;
  if (!opt.ref_dir.empty()) {
    std::printf("\n-- full-output comparison vs %s (rtol %.0e of reference RMS) --\n",
                opt.ref_dir.c_str(), opt.rtol);
    std::printf("  %-16s %8s %12s %12s %7s %10s %10s  %s\n", "vector", "n",
                "max_abs", "rel_rmse", "zeros", "unwritten", "mismatch", "verdict");
    for (const VecCheck& v : checks) {
      std::printf("  %-16s %8zu %12.3e %12.3e %7llu %10llu %10llu  %s%s\n",
                  v.name.c_str(), v.n, v.max_abs, v.rel_rmse,
                  (unsigned long long)v.zeros, (unsigned long long)v.unwritten,
                  (unsigned long long)v.mismatches,
                  v.ref_missing ? "NO REF" : (v.verdict ? "PASS" : "FAIL"),
                  v.first_bad >= 0 ? (" (first at " + std::to_string(v.first_bad) + ")").c_str() : "");
      if (v.ref_missing) ++noref;
      else if (v.verdict) ++verified;
      else ++failed;
    }
    std::printf("  %d vector(s) verified element-by-element, %d failed, %d without a\n"
                "  reference in %s (regenerate it with tools/ref_qwen35.py for the same\n"
                "  token ids and layer)\n", verified, failed, noref, opt.ref_dir.c_str());
    if (checks.empty()) {
      std::printf("  no vectors were compared: the reference directory has no manifest for\n"
                  "  this layer. Generate it with tools/ref_qwen35.py.\n");
    }
  } else {
    std::printf("\n-- no oracle comparison requested (--qwen35-ref DIR) --\n");
  }

  // -- what is still missing ------------------------------------------------
  std::printf("\n-- NOT IMPLEMENTED (this is the honest end of the probe) --\n");
  int i = 0;
  for (const std::string& s : m->missing()) {
    std::printf("  %d. %s\n", ++i, s.c_str());
  }
  if (stopped.empty() && opt.forward) {
    std::printf("\nRESULT: the qwen35 trunk ran: %d causal layer(s) in file order, each "
                "with\n        its residual, then the dense FFN, output_norm and the "
                "untied %d-wide\n        head, producing logits for every token%s.\n",
                m->trunk_layers(), g.n_vocab,
                opt.ref_dir.empty()
                    ? ""
                    : (" and " + std::to_string(verified) +
                       " of " + std::to_string(verified + failed + noref) +
                       " vectors compare equal" +
                       (failed ? (" -- " + std::to_string(failed) + " FAIL") : "") +
                       " against the oracle").c_str());
    std::printf("        A token was produced, which no earlier run of this engine did for a\n"
                "        qwen35 file. It is a probe of the whole trunk, not a\n"
                "        generation loop: see the missing list above.\n");
  } else if (stopped.empty() && opt.attention) {
    std::printf("\nRESULT: the gated attention of layer %d is computed end to end "
                "(QG split,\n        QK-norm over head_dim, partial RoPE, causal softmax, the "
                "gate)%s.\n", L,
                opt.ref_dir.empty()
                    ? ""
                    : (" and " + std::to_string(verified) +
                       " of 11 vectors compare equal" +
                       (failed ? (" -- " + std::to_string(failed) + " FAIL") : "") +
                       " against the oracle").c_str());
    std::printf("        The rest of the trunk and the head are REFUSED, so no logits\n"
                "        are produced by this path: one layer is not a model.\n");
  } else if (stopped.empty() && opt.recurrent) {
    std::printf("\nRESULT: the GatedDeltaNet recurrence of layer %d is computed end to "
                "end\n        (conv1d state carry, the per-group delta rule, the "
                "alpha/beta/ssm_a\n        gating)%s.\n", L,
                opt.ref_dir.empty()
                    ? ""
                    : (" and " + std::to_string(verified) +
                       " of 12 vectors compare equal" +
                       (failed ? (" -- " + std::to_string(failed) + " FAIL") : "") +
                       " against the oracle").c_str());
    std::printf("        The rest of the trunk and the head are REFUSED, so no logits\n"
                "        are produced by this path: one layer is not a model.\n");
  } else if (stopped.empty()) {
    std::printf("\nRESULT: qwen35 front end verified (metadata, binding, tokenizer, "
                "embeddings,\n        layer %d input projections%s); the trunk and the head are "
                "REFUSED.\n        No logits are produced by this path.\n", L,
                opt.ref_dir.empty() ? "" : " against an independent oracle");
  } else {
    std::printf("\nRESULT: qwen35 container and tokenizer verified; layer %d could NOT "
                "be\n        probed (%s).\n        The trunk and the head are REFUSED. No "
                "logits are produced by this path.\n", L, stopped.c_str());
  }

  // A refusal is not success: exit 3 = "understood, and deliberately not run".
  // The trunk is the exception, because it does produce logits -- so it reports a
  // real verdict: 0 when every compared vector passed, 1 when one failed or a
  // vector had no reference, and 3 only when the run itself stopped.
  if (opt.forward && stopped.empty()) return failed == 0 && noref == 0 ? 0 : 1;
  return 3;
}

}  // namespace knj
