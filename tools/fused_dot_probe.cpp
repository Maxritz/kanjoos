// tools/fused_dot_probe.cpp -- exercise knj::dequant_dot_f32 on real tensors.
//
// Purpose: the engine's matrix-vector product calls dequant_dot_f32(type,
// column, x, n_in) once per output column. That fused path (dot_q4_k, dot_q6_k)
// never materialises an f32 copy of the weight, so dequant_crosscheck.py -- which
// only covers dequant_row_f32 -- cannot see it. A fused dot that is wrong would
// be invisible to every existing check.
//
// This probe calls exactly the engine entry point the model uses and writes both
// operands and the results, so tools/fused_dot_crosscheck.py can compute the same
// dots from gguf-py's dequantizer and numpy and disagree at a named index.
//
// Usage:
//   fused_dot_probe <model.gguf> <outdir> <tensor> [<tensor> ...]
// writes <outdir>/<tensor>.fdot_x.f32   (nvec x n_in, row-major)
//        <outdir>/<tensor>.fdot_y.f32   (nvec x rows, row-major)
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/loader/dequant.h"
#include "src/loader/gguf.h"

namespace {

// Deterministic, well-conditioned, and reproducible in numpy: integer-valued
// operands in [-8, 8]. Integer x makes cancellation-free dot products, so a
// mismatch is a real mismatch and not the probe's own rounding.
constexpr int kNVec = 3;

void fill_x(std::vector<float>& x, int vec, uint64_t n_in) {
  uint32_t s = 0x9E3779B9u ^ (uint32_t)(vec * 2654435761u);
  for (uint64_t i = 0; i < n_in; ++i) {
    s = s * 1664525u + 1013904223u;
    x[vec * n_in + i] = (float)((int)((s >> 13) % 17) - 8);
  }
}

void write_f32(const std::string& path, const float* p, size_t n) {
  // FILE*, not std::ofstream: this MinGW's ofstream faults under -O2.
  std::FILE* fp = std::fopen(path.c_str(), "wb");
  if (!fp) throw std::runtime_error("cannot write " + path);
  const size_t wrote = std::fwrite(p, sizeof(float), n, fp);
  std::fclose(fp);
  if (wrote != n) throw std::runtime_error("short write to " + path);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <model.gguf> <outdir> <tensor>...\n", argv[0]);
    return 2;
  }
  try {
    knj::GgufFile f = knj::GgufFile::open(argv[1]);
    for (int a = 3; a < argc; ++a) {
      const std::string name = argv[a];
      const knj::TensorInfo& t = f.require(name);
      const uint64_t n_in = t.dims[0];
      uint64_t rows = 1;
      for (size_t i = 1; i < t.dims.size(); ++i) rows *= t.dims[i];
      const uint64_t stride = knj::column_stride_bytes(t.type, n_in);
      const uint8_t* base = f.data_of(t);

      std::vector<float> x((size_t)kNVec * (size_t)n_in);
      for (int v = 0; v < kNVec; ++v) fill_x(x, v, n_in);

      std::vector<float> y((size_t)kNVec * (size_t)rows);
      for (int v = 0; v < kNVec; ++v) {
        const float* xv = x.data() + (size_t)v * (size_t)n_in;
        float* yv = y.data() + (size_t)v * (size_t)rows;
        for (uint64_t r = 0; r < rows; ++r) {
          yv[r] = knj::dequant_dot_f32(t.type, base + r * stride, xv, n_in);
        }
      }

      const std::string dir = argv[2];
      write_f32(dir + "/" + name + ".fdot_x.f32", x.data(), x.size());
      write_f32(dir + "/" + name + ".fdot_y.f32", y.data(), y.size());
      std::printf("%-28s type=%-6s n_in=%llu rows=%llu nvec=%d stride=%llu\n",
                  name.c_str(), knj::ggml_type_name(t.type),
                  (unsigned long long)n_in, (unsigned long long)rows, kNVec,
                  (unsigned long long)stride);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
