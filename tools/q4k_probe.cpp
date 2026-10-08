// tools/q4k_probe.cpp -- dump f32 weights decoded by kernels/knj_q4k.h from REAL
// model bytes, so an independent decoder (gguf-py) can disagree with it.
//
// WHY THIS EXISTS
// ---------------
// knj_q4k.h is the device-side unpacker for the quantised blocks the models on
// this machine actually carry. It is a transcription of ggml's own formula, and
// a transcription is exactly the kind of thing that "looks right" and is wrong:
// the packed 6-bit scales steal their high bits from their neighbours, and a
// Q6_K block is 210 bytes long (odd), so nothing can be assumed aligned.
//
// A device GEMM measured on top of a wrong unpacker would produce a shape-correct
// number and nothing else. So this probe runs FIRST, on the host, on the real
// bytes at the real file offsets from build/q4k/*.job, and hand the result to
// tools/q4k_unpack_crosscheck.py.
//
// It deliberately does NOT link the engine or HIP: the header under test is the
// only thing in the loop, so a disagreement can only mean the header is wrong.
//
// Usage:
//   q4k_probe <jobfile> <outdir>
// writes <outdir>/<role>.f32 for role in {gate, up, down}.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/knj_q4k.h"

namespace {

struct Row {
  std::string role;
  std::string type;
  long long off = 0;
  long long k = 0, n = 0, e = 0, bpb = 0, wpb = 0;
};

// Minimal key/value reader for the job descriptor. The file is authored by
// tools/q4k_job.py and is a flat list of `key value` lines plus `#` comments.
std::vector<std::pair<std::string, std::string>> read_job(const std::string& path) {
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) throw std::runtime_error("cannot open job " + path);
  std::vector<std::pair<std::string, std::string>> kv;
  char line[1024];
  while (std::fgets(line, sizeof(line), fp)) {
    if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
    char key[256] = {0};
    char val[768] = {0};
    if (std::sscanf(line, "%255s %767s", key, val) != 2) continue;
    kv.emplace_back(key, val);
  }
  std::fclose(fp);
  return kv;
}

std::string get(const std::vector<std::pair<std::string, std::string>>& kv,
                const std::string& key) {
  for (const auto& p : kv)
    if (p.first == key) return p.second;
  throw std::runtime_error("job is missing key: " + key);
}

long long get_int(const std::vector<std::pair<std::string, std::string>>& kv,
                  const std::string& key) {
  return std::strtoll(get(kv, key).c_str(), nullptr, 10);
}

// Read exactly `nbytes` from the model at `off`. The whole tensor is pulled into
// host memory rather than mmapped: the probe must not take a page-fault on the
// path it is meant to be measuring for correctness, and 7-10 MB is nothing.
std::vector<uint8_t> read_at(const std::string& model, long long off, long long nbytes) {
  std::FILE* fp = std::fopen(model.c_str(), "rb");
  if (!fp) throw std::runtime_error("cannot open model " + model);
  if (std::fseek(fp, (long)off, SEEK_SET) != 0) {
    std::fclose(fp);
    throw std::runtime_error("seek failed at " + std::to_string(off));
  }
  std::vector<uint8_t> buf((size_t)nbytes);
  const size_t got = std::fread(buf.data(), 1, (size_t)nbytes, fp);
  std::fclose(fp);
  if (got != (size_t)nbytes)
    throw std::runtime_error("short read at " + std::to_string(off));
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <jobfile> <outdir>\n", argv[0]);
    return 2;
  }
  try {
    const std::string job_path = argv[1];
    const std::string outdir = argv[2];
    const auto kv = read_job(job_path);
    const std::string model = get(kv, "model");

    std::printf("job         %s\n", job_path.c_str());
    std::printf("model       %s\n", model.c_str());
    std::printf("layer       %lld\n", get_int(kv, "layer"));

    int failed = 0;
    for (const char* role : {"gate", "up", "down"}) {
      Row r;
      r.role = role;
      r.type = get(kv, std::string(role) + "_type");
      r.off = get_int(kv, std::string(role) + "_off");
      r.k = get_int(kv, std::string(role) + "_k");
      r.n = get_int(kv, std::string(role) + "_n");
      r.e = get_int(kv, "experts");
      r.bpb = get_int(kv, std::string(role) + "_bpb");
      r.wpb = get_int(kv, std::string(role) + "_wpb");

      const long long n_weights = r.k * r.n * r.e;
      if (n_weights % r.wpb != 0)
        throw std::runtime_error(r.role + ": weight count is not a multiple of the block");
      const long long n_blocks = n_weights / r.wpb;
      const long long nbytes = n_blocks * r.bpb;

      const std::vector<uint8_t> buf = read_at(model, r.off, nbytes);

      if (r.type != "q4_k" && r.type != "q6_k")
        throw std::runtime_error(r.role + ": unsupported type " + r.type);

      std::vector<float> out((size_t)n_weights);
      for (long long i = 0; i < n_weights; ++i) {
        const uint8_t* blk = buf.data() + (size_t)(i / r.wpb) * (size_t)r.bpb;
        const int t = (int)(i % r.wpb);
        out[(size_t)i] = (r.type == "q4_k") ? knj_q4k_weight(blk, t)
                                            : knj_q6k_weight(blk, t);
      }

      // Span-vs-element self-consistency. The device kernel does not call the
      // per-element decoders; it calls the 16-wide span decoders and hoists the
      // sub-block constants out of the loop. That hoist is only legal because a
      // 16-wide span starting at a multiple of 16 cannot cross a sub-block
      // boundary, and "only legal because" is a claim, so it is checked here on
      // every block of every real tensor: same bits, or this run fails.
      if (r.wpb != 256)
        throw std::runtime_error(r.role + ": span check assumes a 256-weight superblock");
      long long span_diff = 0, span_checked = 0;
      {
        float span[16];
        for (long long b = 0; b < n_blocks; ++b) {
          const uint8_t* blk = buf.data() + (size_t)b * (size_t)r.bpb;
          for (int r0 = 0; r0 < 256; r0 += 16) {
            if (r.type == "q4_k")
              knj_q4k_span16(blk, r0, span);
            else
              knj_q6k_span16(blk, r0, span);
            for (int i = 0; i < 16; ++i) {
              const float elem = (r.type == "q4_k") ? knj_q4k_weight(blk, r0 + i)
                                                    : knj_q6k_weight(blk, r0 + i);
              uint32_t a, c;
              __builtin_memcpy(&a, &span[i], 4);
              __builtin_memcpy(&c, &elem, 4);
              if (a != c) ++span_diff;
              ++span_checked;
            }
          }
        }
      }
      if (span_diff) ++failed;

      const std::string path = outdir + "/" + r.role + ".f32";
      std::FILE* fp = std::fopen(path.c_str(), "wb");
      if (!fp) throw std::runtime_error("cannot write " + path);
      const size_t wrote = std::fwrite(out.data(), sizeof(float), out.size(), fp);
      std::fclose(fp);
      if (wrote != out.size()) throw std::runtime_error("short write to " + path);

      // A zero or unwritten dump is a result too -- it is just not a passing one.
      size_t zeros = 0;
      for (float v : out) if (v == 0.0f) ++zeros;

      std::printf("  %-5s %-5s off=%-10lld %lld x %lld x %lld  blocks=%-8lld bytes=%-9lld"
                  " zeros=%zu/%lld  -> %s\n",
                  r.role.c_str(), r.type.c_str(), r.off, r.k, r.n, r.e, n_blocks, nbytes,
                  zeros, n_weights, path.c_str());
      std::printf("        span16 vs element: %lld/%lld bit-identical%s\n",
                  span_checked - span_diff, span_checked,
                  span_diff ? "  <-- MISMATCH" : "");
      if (wrote == 0) ++failed;
    }
    return failed == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
