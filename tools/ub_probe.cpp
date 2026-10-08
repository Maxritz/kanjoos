// tools/ub_probe.cpp -- isolate the -O2 crash in the loader.
#include <cstdio>
#include <cstring>
#include <vector>

#include "src/loader/dequant.h"
#include "src/loader/gguf.h"

#define STEP(msg) do { std::fprintf(stderr, "[step] %s\n", msg); std::fflush(stderr); } while (0)

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  STEP("open");
  knj::GgufFile f = knj::GgufFile::open(argv[1]);
  STEP("require");
  const knj::TensorInfo& t = f.require("blk.0.attn_v.weight");
  STEP("dims");
  std::fprintf(stderr, "[step] dims[0]=%llu dims[1]=%llu size=%zu\n",
               (unsigned long long)t.dims[0], (unsigned long long)t.dims[1], t.dims.size());
  STEP("names");
  std::fprintf(stderr, "[step] type=%s bw=%zu bb=%zu\n", knj::ggml_type_name(t.type),
               knj::ggml_type_block_weights(t.type), knj::ggml_type_block_bytes(t.type));
  STEP("stride");
  const uint64_t stride = knj::column_stride_bytes(t.type, t.dims[0]);
  std::fprintf(stderr, "[step] stride=%llu\n", (unsigned long long)stride);
  STEP("data_of");
  const uint8_t* base = f.data_of(t);
  std::fprintf(stderr, "[step] base=%p\n", (const void*)base);
  STEP("nbytes");
  std::fprintf(stderr, "[step] nbytes=%llu\n", (unsigned long long)t.nbytes());
  STEP("alloc");
  std::vector<float> tmp((size_t)t.dims[0]);
  STEP("dequant");
  knj::dequant_row_f32(t.type, base, tmp.data(), t.dims[0]);
  STEP("sum");
  double s = 0;
  for (uint64_t i = 0; i < t.dims[0]; ++i) s += tmp[i];
  std::fprintf(stderr, "[step] row0 sum=%.6f v[0]=%.6f v[511]=%.6f\n", s, tmp[0], tmp[511]);
  STEP("dot");
  const float d = knj::dequant_dot_f32(t.type, base, tmp.data(), t.dims[0]);
  std::fprintf(stderr, "[step] dot=%.6f\n", d);
  STEP("done");
  return 0;
}
