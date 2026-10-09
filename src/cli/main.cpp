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
#include "src/model/qwen35.h"
#include "src/platform/memprobe.h"
#include "src/profiler/profiler.h"

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
  bool tokenize_only = false;
  bool dump_logits = false;
  int top_logits = 8;
  bool bench = false;
  int bench_pp = 128;
  int bench_tg = 32;
  bool show_geometry = true;
  // docs/06-profiling.md section 1. `--profiling` never changes what the engine
  // computes; it changes what it measures, and the report says which clock it
  // measured on.
  bool profiling = false;
  std::string profile_format = "table";   // table | json | csv
  std::string profile_detail = "class";   // class | layer | kernel
  std::string profile_dir;
  int profile_floor = 256;
  int profile_warmup = 8;
  bool profile_no_subtract = false;
  // Prompt read from a UTF-8 file instead of argv. On Windows the process is
  // handed its arguments in the *ANSI code page*, not UTF-8, so a CJK or emoji
  // prompt arrives mangled -- or, when a conversion ends on a byte the CRT reads
  // as a delimiter, as several arguments. A file is read as bytes and has no
  // such step: `--prompt-file` is the interface for non-ASCII text.
  std::string prompt_file;
  // qwen35 probe (see src/model/qwen35.h). Not a forward pass: a front-end
  // verification with an explicit refusal at the end.
  std::string qwen35_ref;
  std::string qwen35_tokens;
  int qwen35_layer = 0;
  double qwen35_rtol = 2e-3;
  bool qwen35_recurrent = false;
  bool qwen35_attention = false;
  bool qwen35_forward = false;
};

void usage() {
  std::printf(
      "kanjoos-run -- run a GGUF model on the reference CPU path\n"
      "\n"
      "  --model PATH      GGUF to load (required)\n"
      "  -p, --prompt TEXT prompt (raw completion, no chat template)\n"
      "  --prompt-file F   read the prompt from a UTF-8 file instead (use this for\n"
      "                    non-ASCII text: Windows gives argv in the ANSI code page,\n"
      "                    so a CJK or emoji prompt cannot cross that boundary intact)\n"
      "  -n, --n-predict N tokens to generate (default 64)\n"
      "  --ctx N           KV cache size (default 2048)\n"
      "  --threads N       worker threads (default: hardware concurrency)\n"
      "  --temp F          sampling temperature; 0 means greedy (default 0)\n"
      "  --top-k N         top-k filter for temperature > 0 (default 40)\n"
      "  --top-p F         nucleus filter for temperature > 0 (default 0.95)\n"
      "  --seed N          RNG seed (default 1)\n"
      "  --dump-tokens     print the prompt's token ids\n"
      "  --tokenize-only   print token ids and exit without a model forward pass\n"
      "  --dump-logits     print the top tokens at the first generated position\n"
      "  --bench PP TG     time a PP-token prefill and a TG-token decode run\n"
      "  --profiling[=FMT] profile the run: table (default), json, csv, no-subtract, off\n"
      "  --profile-dir DIR also write profile.{txt,json,csv} into DIR\n"
      "  --profile-floor N empty ops timed through the same begin/end path (default 256)\n"
      "  --profile-warmup N discard the first N steps (default 8)\n"
      "  --profile-detail=D component detail: class (default), layer, kernel\n"
      "\n"
      "qwen35 (a different trunk, not a configuration of this one):\n"
      "  --qwen35-ref DIR  compare full outputs against tools/ref_qwen35.py's oracle\n"
      "  --qwen35-layer N  the layer whose input projections are verified (default 0)\n"
      "  --qwen35-tokens L comma-separated token ids to run instead of tokenising\n"
      "                    --prompt; the oracle is invoked with an explicit --tokens\n"
      "                    list, so a comparison needs the same ids on both sides\n"
      "  --qwen35-rtol F   comparison tolerance, relative to reference RMS (default 2e-3)\n"
      "  --qwen35-recurrent  run the layer's GatedDeltaNet recurrence end to end:\n"
      "                    conv1d state carry, the per-group delta rule and the\n"
      "                    alpha/beta/ssm_a gating, compared as the 12 layer_* vectors\n"
      "                    tools/ref_qwen35.py --recurrent writes (implies the layer\n"
      "                    must be recurrent; a full-attention layer is refused)\n"
      "  --qwen35-attention  run the layer's gated full attention end to end: the QG\n"
      "                    split, QK-norm over head_dim, partial RoPE, causal softmax\n"
      "                    and the sigmoid gate, compared as the 11 layer_* vectors\n"
      "                    tools/ref_qwen35.py --attention writes (implies the layer\n"
      "                    must be full attention; a recurrent layer is refused)\n"
      "  --qwen35-forward    run the whole trunk -- all 64 causal layers in file order\n"
      "                    (48 SSM + 16 attention), each with its residual; the dense\n"
      "                    SwiGLU FFN; output_norm; and the untied 248320-wide head --\n"
      "                    and compare the residual stream after every layer, the\n"
      "                    head's input and the logits against tools/ref_qwen35.py\n"
      "                    --forward. This path produces logits and exits 0 when it\n"
      "                    verifies\n");
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
    else if (s == "--prompt-file") a.prompt_file = next();
    else if (s == "-n" || s == "--n-predict") a.n_predict = std::stoi(next());
    else if (s == "--ctx") a.n_ctx = std::stoi(next());
    else if (s == "--threads") a.threads = std::stoi(next());
    else if (s == "--temp") a.temp = std::stof(next());
    else if (s == "--top-k") a.top_k = std::stoi(next());
    else if (s == "--top-p") a.top_p = std::stof(next());
    else if (s == "--seed") a.seed = (uint32_t)std::stoul(next());
    else if (s == "--dump-tokens") a.dump_tokens = true;
    else if (s == "--tokenize-only") { a.dump_tokens = true; a.tokenize_only = true; }
    else if (s == "--dump-logits") a.dump_logits = true;
    else if (s == "--top-logits") a.top_logits = std::stoi(next());
    else if (s == "--qwen35-ref") a.qwen35_ref = next();
    else if (s == "--qwen35-tokens") a.qwen35_tokens = next();
    else if (s == "--qwen35-layer") a.qwen35_layer = std::stoi(next());
    else if (s == "--qwen35-rtol") a.qwen35_rtol = std::stod(next());
    else if (s == "--qwen35-recurrent") a.qwen35_recurrent = true;
    else if (s == "--qwen35-attention") a.qwen35_attention = true;
    else if (s == "--qwen35-forward") a.qwen35_forward = true;
    else if (s == "--profile-dir") a.profile_dir = next();
    else if (s == "--profile-floor") a.profile_floor = std::stoi(next());
    else if (s == "--profile-warmup") a.profile_warmup = std::stoi(next());
    else if (s.rfind("--profile-detail=", 0) == 0) {
      a.profile_detail = s.substr(std::strlen("--profile-detail="));
      if (a.profile_detail != "class" && a.profile_detail != "layer" &&
          a.profile_detail != "kernel") {
        throw std::runtime_error("--profile-detail must be class, layer or kernel");
      }
    } else if (s.rfind("--profiling", 0) == 0) {
      a.profiling = true;
      const size_t eq = s.find('=');
      if (eq != std::string::npos) a.profile_format = s.substr(eq + 1);
      if (a.profile_format == "off") {
        a.profiling = false;
        a.profile_format = "table";
      } else if (a.profile_format == "no-subtract") {
        a.profile_no_subtract = true;
        a.profile_format = "table";
      } else if (a.profile_format != "table" && a.profile_format != "json" &&
                 a.profile_format != "csv") {
        // `counters` and `full` are in the document's list and are NOT
        // implemented here. Refuse rather than quietly print something else.
        throw std::runtime_error("--profiling=" + a.profile_format +
                                 " is not implemented; this build implements "
                                 "off|table|json|csv|no-subtract (docs/06-profiling.md s1)");
      }
    } else if (s == "--bench") {
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
  {
    KNJ_PROFILE_OP("prefill");
    for (int i = 0; i < pp; ++i) {
      m.forward(&ids[i], 1);
      knj::Profiler::get().step_done();
    }
  }
  const double pp_ms = ms_since(t0);

  // A decode run must be a real decode: one token at a time, each attending to
  // everything before it.
  std::vector<std::vector<float>> logits_sink;
  t0 = std::chrono::steady_clock::now();
  {
    KNJ_PROFILE_OP("decode");
    for (int i = 0; i < tg; ++i) {
      m.forward(&ids[0], 1);
      knj::Profiler::get().step_done();
    }
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
    Args a = parse(argc, argv);
    if (!a.prompt_file.empty()) {
      // Bytes, not text: the file is the UTF-8 the caller wrote, and decoding is
      // the tokenizer's job. A file that cannot be read is a named refusal.
      std::FILE* f = std::fopen(a.prompt_file.c_str(), "rb");
      if (!f) {
        std::printf("error: --prompt-file cannot be opened: %s\n", a.prompt_file.c_str());
        return 1;
      }
      std::string text;
      char buf[4096];
      size_t got = 0;
      while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, got);
      const bool bad = std::ferror(f) != 0;
      std::fclose(f);
      if (bad) {
        std::printf("error: --prompt-file failed while reading: %s\n", a.prompt_file.c_str());
        return 1;
      }
      a.prompt = text;
    }
    Model::Options opt;
    opt.n_ctx = a.n_ctx;
    opt.threads = a.threads;

    // C21. Configured before anything is loaded so `load` is itself an op in the
    // table, and the floor is measured before the first real op so the empty-op
    // cost belongs to this run rather than to a previous one.
    knj::ProfileConfig pc;
    pc.on = a.profiling;
    pc.format = a.profile_format;
    pc.detail = a.profile_detail;
    pc.dir = a.profile_dir;
    pc.floor_ops = a.profile_floor;
    pc.warmup_steps = a.profile_warmup;
    pc.subtract_floor = !a.profile_no_subtract;
    knj::Profiler& prof = knj::Profiler::get();
    prof.configure(pc);
    prof.set_domain(knj::ClockDomain::Host);
    prof.measure_floor();

    // Which architecture is this, before the forward pass decides what to do
    // with it. `qwen35` is a different trunk (hybrid SSM + gated attention), not
    // a configuration of `qwen3moe`, so it gets a front-end probe that verifies
    // what can be verified and refuses the rest by name. `dflash` is a draft
    // model for a target this engine cannot run yet; it is refused with the
    // reason rather than with the generic architecture message.
    {
      std::string arch;
      try {
        knj::GgufFile sniff = knj::GgufFile::open(a.model);
        if (sniff.has("general.architecture")) {
          arch = sniff.meta_string("general.architecture");
        }
      } catch (const std::exception& e) {
        std::printf("model      : %s\n", a.model.c_str());
        std::printf("REFUSED    : the container could not be read: %s\n", e.what());
        std::printf("             Nothing was measured. tools/ggufmeta/gguf_meta.py reads\n"
                    "             the same file and prints its header and tensor table.\n");
        return 3;
      }
      if (arch == "qwen35" && a.tokenize_only) {
        // The qwen35 probe normally returns 3 after its front-end report because
        // it does not produce logits. Tokenizer cross-checks need an honest 0
        // from tokenization alone, without entering that probe or its forward path.
        try {
          const knj::GgufFile file = knj::GgufFile::open(a.model);
          const knj::Tokenizer tok = knj::Tokenizer::from_gguf(file);
          const std::vector<int32_t> ids = tok.encode(a.prompt, false);
          std::printf("probe text \\\"%s\\\" -> %zu id(s):", a.prompt.c_str(), ids.size());
          for (int32_t id : ids) std::printf(" %d", id);
          std::printf("\\n");
          return 0;
        } catch (const std::exception& e) {
          std::fprintf(stderr, "error: %s\\n", e.what());
          return 1;
        }
      }
      if (arch == "qwen35") {
        knj::Qwen35ProbeOptions q;
        q.model = a.model;
        q.ref_dir = a.qwen35_ref;
        q.tokens = a.qwen35_tokens;
        q.layer = a.qwen35_layer;
        q.rtol = a.qwen35_rtol;
        q.recurrent = a.qwen35_recurrent;
        q.attention = a.qwen35_attention;
        q.forward = a.qwen35_forward;
        q.text = a.prompt;
        const int rc = knj::run_qwen35_probe(q);
        if (a.profiling) {
          knj::Profiler& p = knj::Profiler::get();
          p.print_header(stdout, "0.1.0",
                         "qwen35 front-end probe (no device backend)",
                         "vram NOT MEASURED here   ram NOT MEASURED here");
          p.report(stdout);
          p.write_dir();
        }
        return rc;
      }
      if (arch == "dflash") {
        std::printf("model      : %s\n", a.model.c_str());
        std::printf("REFUSED    : architecture 'dflash' is a *draft* model, not a language\n"
                    "             model. It carries no token_embd and no output.weight: it\n"
                    "             conditions on a target model's hidden states and emits a\n"
                    "             block of tokens. There is nothing here for it to draft\n"
                    "             against, and the C20 loader is not implemented.\n");
        std::printf("             Measured layouts: docs/10-dflash-draft-models.md\n"
                    "             Owning worksheet: ai-coder/c20-speculation.md\n");
        return 3;
      }
    }

    auto t_open0 = std::chrono::steady_clock::now();
    std::unique_ptr<Model> m;
    {
      KNJ_PROFILE_OP("load");
      m = Model::open(a.model, opt);
    }
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

    std::vector<int32_t> prompt;
    {
      KNJ_PROFILE_OP("tokenize");
      prompt = tok.encode(a.prompt, false);
    }
    if (tok.add_bos() && tok.bos_id() >= 0) prompt.insert(prompt.begin(), tok.bos_id());
    if (prompt.empty()) throw std::runtime_error("prompt tokenised to nothing");
    if (a.dump_tokens) {
      std::printf("prompt ids :");
      for (int32_t id : prompt) std::printf(" %d", id);
      std::printf("\n");
    }
    std::printf("prompt     : %zu tokens, round-trip \"%s\"\n", prompt.size(),
                tok.decode(prompt).c_str());

    // One report path for every exit, so a profiled run can never end without
    // its table. `finish` is defined here because it needs the loaded model.
    auto finish = [&]() {
      if (!a.profiling) return;
      knj::Profiler& p = knj::Profiler::get();
      const knj::HostMem hm = knj::host_mem();
      char dev[256];
      std::snprintf(dev, sizeof(dev),
                    "host CPU reference path (no device backend)   threads %d   ctx %d   "
                    "vocab %d",
                    m->threads(), m->n_ctx(), m->geom().n_vocab);
      char bud[384];
      std::snprintf(bud, sizeof(bud),
                    "vram NOT MEASURED here (no device on this path)   ram %.2f/%.2f GiB "
                    "available   rss peak %.0f MiB   platform %s   cpus %d",
                    (double)hm.phys_avail_bytes / 1073741824.0,
                    (double)hm.phys_total_bytes / 1073741824.0,
                    (double)hm.peak_rss_bytes / 1048576.0, knj::platform_name(),
                    knj::logical_cpus());
      p.print_header(stdout, "0.1.0", dev, bud);
      p.report(stdout);
      p.write_dir();
    };

    // Tokenization is a complete result: ids printed, report emitted, exit 0
    // BEFORE the context check and before prefill. The context bound and the
    // forward pass belong to generation; a tokenizer cross-check must not pay
    // for either (and must not fail on them).
    if (a.tokenize_only) {
      finish();
      return 0;
    }

    if ((int)prompt.size() >= a.n_ctx) {
      throw std::runtime_error("prompt longer than the context");
    }

    if (a.bench) {
      run_bench(*m, a);
      finish();
      return 0;
    }

    // Prefill: the whole prompt in one call, which is what batched weights buy.
    auto t0 = std::chrono::steady_clock::now();
    const float* logits = nullptr;
    {
      KNJ_PROFILE_OP("prefill");
      logits = m->forward(prompt.data(), (int)prompt.size());
      prof.step_done();
    }
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
    int32_t next = 0;
    {
      KNJ_PROFILE_OP("sample");
      next = sample(logits, m->geom().n_vocab, a, rng);
    }
    for (int i = 0; i < a.n_predict; ++i) {
      gen.push_back(next);
      if (next == tok.eos_id()) break;
      if (m->pos() + 1 > m->n_ctx()) break;
      auto t1 = std::chrono::steady_clock::now();
      const float* l = nullptr;
      {
        KNJ_PROFILE_OP("decode");
        l = m->forward(&next, 1);
        prof.step_done();
      }
      tg_ms += ms_since(t1);
      {
        KNJ_PROFILE_OP("sample");
        next = sample(l, m->geom().n_vocab, a, rng);
      }
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
    finish();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
