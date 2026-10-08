// src/loader/dequant.h -- GGML block formats to f32.
//
// The engine never materialises a whole model in f32: a 1.5 B model is ~6 GB
// dequantized, and the streaming target is far larger. Everything here works on
// a *contiguous run of weights*, which for a ggml 2-D tensor [ne0, ne1] is one
// output column -- ne0 is the fastest-varying dimension. So a caller that wants
// "column j of this weight" gets it with one call and no striding.
#pragma once

#include <cstddef>
#include <cstdint>

#include "src/loader/gguf.h"

namespace knj {

// IEEE half -> float, handling subnormals, inf and nan.
float fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);
float bf16_to_fp32(uint16_t b);

// Decode `n` consecutive weights starting at `src` into `dst`.
//
// `n` counts weights, not bytes, and must be a whole number of blocks for the
// type. Note that for the K-quants a "superblock" is 256 weights and the
// run must be a multiple of that.
void dequant_row_f32(GgmlType type, const uint8_t* src, float* dst, uint64_t n);

// Decode `n` weights and return their dot product with `x` in one pass.
//
// This is the shape every matrix-vector product actually needs, and it avoids
// writing an f32 copy of the weight anywhere: the decoded block lives in a
// stack buffer and is consumed immediately. For the 256-weight K-quants that
// matters -- a fused dot is ~1 pass over memory where decode-then-dot is 2.
float dequant_dot_f32(GgmlType type, const uint8_t* src, const float* x, uint64_t n);

// Byte distance between consecutive output columns of a tensor whose fastest
// dimension is `ne0`. This is the only stride the loader needs.
inline uint64_t column_stride_bytes(GgmlType type, uint64_t ne0) {
  return ne0 / ggml_type_block_weights(type) * ggml_type_block_bytes(type);
}

// Bytes occupied by one whole tensor payload.
inline uint64_t tensor_bytes(GgmlType type, uint64_t ne0, uint64_t ne1) {
  return column_stride_bytes(type, ne0) * ne1;
}

}  // namespace knj
