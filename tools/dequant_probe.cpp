// tools/dequant_probe.cpp -- dump the engine's dequantised f32 for named tensors.
//
// Purpose: give a *second*, independent implementation something to disagree
// with. The engine's dequantisers and gguf-py's were written by different people
// from the same reference, so a byte-for-byte agreement on real model tensors is
// evidence, and any disagreement names the tensor and the index.
//
// Usage:
//   dequant_probe <model.gguf> <outdir> <tensor> [<tensor> ...]
// writes <outdir>/<tensor>.f32
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "src/loader/dequant.h"
#include "src/loader/gguf.h"

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
      const int64_t n_in = (int64_t)t.dims[0];
      int64_t rows = 1;
      for (size_t i = 1; i < t.dims.size(); ++i) rows *= (int64_t)t.dims[i];
      const uint64_t stride = knj::column_stride_bytes(t.type, (uint64_t)n_in);
      const uint8_t* base = f.data_of(t);

      std::vector<float> out((size_t)rows * (size_t)n_in);
      std::vector<float> tmp((size_t)n_in);
      for (int64_t r = 0; r < rows; ++r) {
        knj::dequant_row_f32(t.type, base + (uint64_t)r * stride, tmp.data(), (uint64_t)n_in);
        std::memcpy(out.data() + (size_t)r * (size_t)n_in, tmp.data(),
                    sizeof(float) * (size_t)n_in);
      }
      const std::string path = std::string(argv[2]) + "/" + name + ".f32";
      // FILE* and not std::ofstream: this MinGW's ofstream faults under -O2 in
      // this translation unit, and the probe must not be the thing under test.
      std::FILE* fp = std::fopen(path.c_str(), "wb");
      if (!fp) throw std::runtime_error("cannot write " + path);
      const size_t wrote = std::fwrite(out.data(), sizeof(float), out.size(), fp);
      std::fclose(fp);
      if (wrote != out.size()) throw std::runtime_error("short write to " + path);
      std::printf("%-28s type=%-6s dims=", name.c_str(), knj::ggml_type_name(t.type));
      for (size_t i = 0; i < t.dims.size(); ++i) std::printf("%llu%s", (unsigned long long)t.dims[i],
                                                            i + 1 == t.dims.size() ? "" : "x");
      std::printf("  rows=%lld n_in=%lld stride=%llu bytes=%zu\n", (long long)rows,
                  (long long)n_in, (unsigned long long)stride, out.size() * sizeof(float));
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
