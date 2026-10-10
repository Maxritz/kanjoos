// tools/dequant_bytes_probe.cpp -- decode raw block bytes to f32.
//
// Purpose: validate a decoder against crafted bytes (random, edge, exact)
// without needing a model file. The companion script builds block bytes in
// python, this probe runs knj::dequant_row_f32 over them, and the script
// compares against gguf-py's decoder on the same bytes.
//
// Usage:
//   dequant_bytes_probe <TYPE> <bytes-file> <n-weights> <out-f32>
// TYPE is the ggml type name (e.g. Q4_1, NVFP4, IQ4_XS). Exit 1 on refusal,
// exit 2 on usage error.
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/loader/dequant.h"
#include "src/loader/gguf.h"

namespace {

knj::GgmlType parse_type(const char* s) {
  // Upper bound covers fork-experimental ids (ROCMFP family at 100+).
  for (uint32_t id = 0; id < 256; ++id) {
    const knj::GgmlType t = static_cast<knj::GgmlType>(id);
    if (std::strcmp(knj::ggml_type_name(t), s) == 0) return t;
  }
  throw std::runtime_error(std::string("unknown type name: ") + s);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s <TYPE> <bytes> <n> <out-f32>\n", argv[0]);
    return 2;
  }
  try {
    const knj::GgmlType t = parse_type(argv[1]);
    const uint64_t n = std::stoull(argv[3]);
    std::FILE* fp = std::fopen(argv[2], "rb");
    if (!fp) throw std::runtime_error("cannot read bytes file");
    std::fseek(fp, 0, SEEK_END);
    const long nb = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> bytes((size_t)(nb > 0 ? nb : 0));
    if (!bytes.empty() &&
        std::fread(bytes.data(), 1, bytes.size(), fp) != bytes.size())
      throw std::runtime_error("short read on bytes file");
    std::fclose(fp);
    std::vector<float> out((size_t)n);
    knj::dequant_row_f32(t, bytes.data(), out.data(), n);
    std::FILE* fo = std::fopen(argv[4], "wb");
    if (!fo) throw std::runtime_error("cannot write output");
    const size_t wrote = std::fwrite(out.data(), sizeof(float), out.size(), fo);
    std::fclose(fo);
    if (wrote != out.size()) throw std::runtime_error("short write");
    std::printf("%s n=%llu bytes=%ld\n", argv[1], (unsigned long long)n, nb);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
