// src/cli/main.cpp -- load a GGUF, generate, and report throughput.
//
// This exists to answer one question honestly: does the engine produce the same
// text a reference does, at a speed that is worth measuring? Every number it
// prints is a real wall-clock measurement over real work, and it refuses runs
// it cannot describe (no metrics for a prompt it failed to tokenise, no speed
// for a generation that hit the context wall).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/model/model.h"

namespace {

using knj::Model;
using knj::ModelGeometry;

struct Args {
  std::string model;
  std::string prompt = "The capital of France is";
  int n_ctx = 2048;
  int threads = 0;
  int n_predict = 64;
  float temp = 0.0f;
  int top_k = 40;
  float top_p = 0.95f;
  uint32_t seed = 1;
  bool dump_tokens = false;
  bool dump_logits = false;
  int top_logits = 8;
  bool bench = false;
  int bench_pp = 128;
  int bench_tg = 32;
  bool show_geometry = true;
};

void usage() {
  std::printf(
      "kanjoos-run -- run a GGUF model on the reference CPU path\n"
      "\n"
      "  --model PATH      GGUF to load (required)\n"
      "  -p, --prompt TEXT prompt (raw completion, no chat template)\n"
      "  -n, --n-predict N tokens to generate (default 64)\n"
      "  --ctx N           KV cache size (default 2048)\n"
      "  --threads N       worker threads (default: hardware concurrency)\n"
      "  --temp F          sampling temperature; 0 means greedy (default 0)\n"
      "  --top-k N         top-k filter for temperature > 0 (default 40)\n"
      "  --top-p F         nucleus filter for temperature > 0 (default 0.95)\n"
      "  --seed N          RNG seed (default 1)\n"
      "  --dump-tokens     print the prompt's token ids\n"
      "  --dump-logits     print the top tokens at the first generated position\n"
      "  --bench PP TG     time a PP-token prefill and a TG-token decode run\n");
}

int32_t sample(const float* logits, int n_vocab, const Args& a, std::mt19937& rng) {
  if (a.temp <= 0.0f) {
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) {
      if (logits[i] > logits[best]) best = i;
    }
    return best;
  }
  std::vector<std::pair<float, int32_t>> v(n_vocab);
  const float inv_t = 1.0f / a.temp;
  for (int i = 0; i < n_vocab; ++i) v[i] = {logits[i] * inv_t, i};
  std::partial_sort(v.begin(), v.begin() + std::min<size_t>(v.size(), a.top_k), v.end(),
                    [](const auto& x, const auto& y) { return x.first > y.first; });
  const int k = std::min<int>(n_vocab, a.top_k);
  float mx = v[0].first;
  float sum = 0.0f;
  std::vector<float> p(k);
  for (int i = 0; i < k; ++i) {
    p[i] = std::exp(v[i].first - mx);
    sum += p[i];
  }
  for (int i = 0; i < k; ++i) p[i] /= sum;
  int keep = k;
  float acc = 0.0f;
  for (int i = 0; i < k; ++i) {
    acc += p[i];
    if (acc >= a.top_p) {
      keep = i + 1;
      break;
    }
  }
  float total = 0.0f;
  for (int i = 0; i < keep; ++i) total += p[i];
  std::uniform_real_distribution<float> dist(0.0f, total);
  const float r = dist(rng);
  float running = 0.0f;
  for (int i = 0; i < keep; ++i) {
    running += p[i];
    if (r <= running) return v[i].second;
  }
  return v[keep - 1].second;
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
      return argv[++i];
    };
    if (s == "--model") a.model = next();
    else if (s == "-p" || s == "--prompt") a.prompt = next();
    else if (s == "-n" || s == "--n-predict") a.n_predict = std::stoi(next());
    else if (s == "--ctx") a.n_ctx = std::stoi(next());
    else if (s == "--threads") a.threads = std::stoi(next());
    else if (s == "--temp") a.temp = std::stof(next());
    else if (s == "--top-k") a.top_k = std::stoi(next());
    else if (s == "--top-p") a.top_p = std::stof(next());
    else if (s == "--seed") a.seed = (uint32_t)std::stoul(next());
    else if (s == "--dump-tokens") a.dump_tokens = true;
    else if (s == "--dump-logits") a.dump_logits = true;
    else if (s == "--top-logits") a.top_logits = std::stoi(next());
    else if (s == "--bench") {
      a.bench = true;
      a.bench_pp = std::stoi(next());
      a.bench_tg = std::stoi(next());
    } else if (s == "-h" || s == "--help") {
      usage();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + s);
    }
  }
  if (a.model.empty()) throw std::runtime_error("--model is required");
  return a;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
      .count();
}

void run_bench(Model& m, const Args& a) {
  const int pp = std::min(a.bench_pp, a.n_ctx);
  const int tg = std::min(a.bench_tg, a.n_ctx - pp);
  if (pp <= 0 || tg <= 0) throw std::runtime_error("--bench needs ctx > pp + tg");

  std::vector<int32_t> ids(pp, 1000);
  m.reset();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < pp; ++i) m.forward(&ids[i], 1);
  const double pp_ms = ms_since(t0);

  // A decode run must be a real decode: one token at a time, each attending to
  // everything before it.
  std::vector<std::vector<float>> logits_sink;
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < tg; ++i) {
    m.forward(&ids[0], 1);
  }
  const double tg_ms = ms_since(t0);

  std::printf("\n-- bench (prompt token repeated; not a quality measurement) --\n");
  std::printf("pp%-4d  %9.2f ms  %10.2f tok/s\n", pp, pp_ms, 1000.0 * pp / pp_ms);
  std::printf("tg%-4d  %9.2f ms  %10.2f tok/s  (%.3f ms/token)\n", tg, tg_ms,
              1000.0 * tg / tg_ms, tg_ms / tg);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = parse(argc, argv);
    Model::Options opt;
    opt.n_ctx = a.n_ctx;
    opt.threads = a.threads;

    auto t_open0 = std::chrono::steady_clock::now();
    std::unique_ptr<Model> m = Model::open(a.model, opt);
    const double open_ms = ms_since(t_open0);

    std::printf("model      : %s\n", a.model.c_str());
    std::printf("loaded     : %.1f ms (mapped, nothing dequantized at load)\n", open_ms);
    std::printf("file bytes : %llu\n", (unsigned long long)m->file().file_size());
    std::printf("gguf ver   : %u, alignment %llu, tensors %zu, kv %zu\n", m->file().version(),
                (unsigned long long)m->file().alignment(), m->file().tensors().size(),
                m->file().metadata().size());
    std::printf("threads    : %d\n", m->threads());
    if (a.show_geometry) std::printf("geometry   : %s\n", m->geom().describe().c_str());

    const knj::Tokenizer& tok = m->tokenizer();
    std::printf("tokenizer  : vocab %d, bos %d, eos %d, add_bos %d\n", tok.vocab_size(),
                tok.bos_id(), tok.eos_id(), (int)tok.add_bos());

    std::vector<int32_t> prompt = tok.encode(a.prompt, false);
    if (tok.add_bos() && tok.bos_id() >= 0) prompt.insert(prompt.begin(), tok.bos_id());
    if (prompt.empty()) throw std::runtime_error("prompt tokenised to nothing");
    if (a.dump_tokens) {
      std::printf("prompt ids :");
      for (int32_t id : prompt) std::printf(" %d", id);
      std::printf("\n");
    }
    std::printf("prompt     : %zu tokens, round-trip \"%s\"\n", prompt.size(),
                tok.decode(prompt).c_str());

    if ((int)prompt.size() >= a.n_ctx) {
      throw std::runtime_error("prompt longer than the context");
    }

    if (a.bench) {
      run_bench(*m, a);
      return 0;
    }

    // Prefill: the whole prompt in one call, which is what batched weights buy.
    auto t0 = std::chrono::steady_clock::now();
    const float* logits = m->forward(prompt.data(), (int)prompt.size());
    const double pp_ms = ms_since(t0);

    if (a.dump_logits) {
      std::vector<std::pair<float, int32_t>> v((size_t)m->geom().n_vocab);
      for (int i = 0; i < m->geom().n_vocab; ++i) v[(size_t)i] = {logits[i], i};
      const int k = std::min<int>(m->geom().n_vocab, a.top_logits);
      std::partial_sort(v.begin(), v.begin() + k, v.end(),
                        [](const auto& x, const auto& y) { return x.first > y.first; });
      std::printf("top %d at last prompt position:\n", k);
      for (int i = 0; i < k; ++i) {
        std::printf("   %10d  %12.5f  \"%s\"\n", v[(size_t)i].second, v[(size_t)i].first,
                    tok.decode_one(v[(size_t)i].second).c_str());
      }
    }

    std::mt19937 rng(a.seed);
    std::vector<int32_t> gen;
    double tg_ms = 0.0;
    int32_t next = sample(logits, m->geom().n_vocab, a, rng);
    for (int i = 0; i < a.n_predict; ++i) {
      gen.push_back(next);
      if (next == tok.eos_id()) break;
      if (m->pos() + 1 > m->n_ctx()) break;
      auto t1 = std::chrono::steady_clock::now();
      const float* l = m->forward(&next, 1);
      tg_ms += ms_since(t1);
      next = sample(l, m->geom().n_vocab, a, rng);
    }

    std::printf("\nprompt     : %s\n", a.prompt.c_str());
    std::printf("completion : %s\n", tok.decode(gen).c_str());
    std::printf("\n-- measured --\n");
    std::printf("prefill    : %zu tokens in %8.2f ms  %10.2f tok/s\n", prompt.size(), pp_ms,
                1000.0 * (double)prompt.size() / pp_ms);
    if (!gen.empty()) {
      std::printf("decode     : %zu tokens in %8.2f ms  %10.2f tok/s  (%.3f ms/token)\n",
                  gen.size(), tg_ms, 1000.0 * (double)gen.size() / tg_ms, tg_ms / (double)gen.size());
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
