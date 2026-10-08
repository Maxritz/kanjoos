// src/loader/dequant.cpp -- GGML block formats to f32.
//
// Every routine here is a transcription of ggml's reference dequantiser, which
// is the only definition that matters: these are the bytes llama.cpp wrote.
// Where a formula looks like it "should" be simpler, the reference is followed
// anyway, because the packing is not uniform across sub-blocks.
#include "src/loader/dequant.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace knj {
namespace {

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("dequant: " + m); }

uint16_t rd16(const uint8_t* p) {
  uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}

// Q4_K packs eight 6-bit sub-block scales and eight 6-bit minima into 12 bytes.
// The first four scales live in the low nibbles of scales[0..3]; the second set
// straddles bytes 4..11. Transcribed from ggml's get_scale_min_k4.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

void dequant_q4_0(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 32;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    const uint8_t* qs = src + 2;
    for (int l = 0; l < 16; ++l) {
      dst[l]      = d * (float)((int)(qs[l] & 0x0F) - 8);
      dst[l + 16] = d * (float)((int)(qs[l] >> 4) - 8);
    }
    src += 18;
    dst += 32;
  }
}

void dequant_q8_0(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 32;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    const int8_t* qs = reinterpret_cast<const int8_t*>(src + 2);
    for (int l = 0; l < 32; ++l) dst[l] = d * (float)qs[l];
    src += 34;
    dst += 32;
  }
}

// block_q4_K: { half d; half dmin; uint8 scales[12]; uint8 qs[128]; } = 144 B
// Sixteen 32-weight sub-blocks; the loop below consumes 32 qs bytes per step
// (four steps = 128 bytes) and emits 64 weights per step.
void dequant_q4_k(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d    = fp16_to_fp32(rd16(src + 0));
    const float dmin = fp16_to_fp32(rd16(src + 2));
    const uint8_t* scales = src + 4;
    const uint8_t* q = src + 16;

    for (int j = 0; j < 4; ++j) {
      uint8_t sc, m;
      get_scale_min_k4(2 * j, scales, &sc, &m);
      const float d1  = d * (float)sc;
      const float dm1 = dmin * (float)m;
      get_scale_min_k4(2 * j + 1, scales, &sc, &m);
      const float d2  = d * (float)sc;
      const float dm2 = dmin * (float)m;
      for (int l = 0; l < 32; ++l) dst[l]      = d1 * (float)(q[l] & 0x0F) - dm1;
      for (int l = 0; l < 32; ++l) dst[l + 32] = d2 * (float)(q[l] >> 4)   - dm2;
      q += 32;
      dst += 64;
    }
    src += 144;
  }
}

// block_q6_K: { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d; } = 210 B
void dequant_q6_k(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* ql = src;
    const uint8_t* qh = src + 128;
    const int8_t*  sc = reinterpret_cast<const int8_t*>(src + 192);
    const float d = fp16_to_fp32(rd16(src + 208));

    for (int nn = 0; nn < 256; nn += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int q1 = (int)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int q3 = (int)((ql[l +  0] >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int q4 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
        dst[l +  0] = d * (float)sc[is + 0] * (float)q1;
        dst[l + 32] = d * (float)sc[is + 2] * (float)q2;
        dst[l + 64] = d * (float)sc[is + 4] * (float)q3;
        dst[l + 96] = d * (float)sc[is + 6] * (float)q4;
      }
      dst += 128;
      ql  += 64;
      qh  += 32;
      sc  += 8;
    }
    src += 210;
  }
}

// --- fused dots --------------------------------------------------------
//
// Both K-quants scale a whole sub-block by one factor, so the factor can be
// applied to the *sum* instead of to each weight:
//
//   Q4_K:  w_l = d1*q_l - dm1   =>  sum(w*x) = d1*sum(q*x) - dm1*sum(x)
//   Q6_K:  w_l = d*sc*q_l       =>  sum(w*x) = d*sc*sum(q*x)
//
// That is one pass over the packed bytes with no 256-float staging buffer, and
// it is what ggml's own vec_dot does (integer sub-sums, scale applied last) --
// the decode-then-dot version was the odd one out. Rounding differs from the
// staged form only by association order, in a sum over 32 small integers.

float dot_q4_k(const uint8_t* src, const float* x, uint64_t nb) {
  float acc = 0.0f;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d    = fp16_to_fp32(rd16(src + 0));
    const float dmin = fp16_to_fp32(rd16(src + 2));
    const uint8_t* scales = src + 4;
    const uint8_t* q = src + 16;

    for (int j = 0; j < 4; ++j) {
      uint8_t sc, m;
      get_scale_min_k4(2 * j, scales, &sc, &m);
      const float d1 = d * (float)sc, dm1 = dmin * (float)m;
      get_scale_min_k4(2 * j + 1, scales, &sc, &m);
      const float d2 = d * (float)sc, dm2 = dmin * (float)m;

      const uint8_t* qb = q + 32 * j;
      const float* x0 = x + 64 * j;
      float sq0 = 0.0f, sx0 = 0.0f, sq1 = 0.0f, sx1 = 0.0f;
      for (int l = 0; l < 32; ++l) {
        sq0 += (float)(qb[l] & 0x0F) * x0[l];
        sx0 += x0[l];
        sq1 += (float)(qb[l] >> 4) * x0[l + 32];
        sx1 += x0[l + 32];
      }
      acc += (d1 * sq0 - dm1 * sx0) + (d2 * sq1 - dm2 * sx1);
    }
    src += 144;
    x   += 256;
  }
  return acc;
}

float dot_q6_k(const uint8_t* src, const float* x, uint64_t nb) {
  float acc = 0.0f;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* ql = src;
    const uint8_t* qh = src + 128;
    const int8_t*  sc = reinterpret_cast<const int8_t*>(src + 192);
    const float d = fp16_to_fp32(rd16(src + 208));

    // Two 128-weight halves; each half is four groups of 32 whose scale varies
    // across the 16-weight rows (sc[2g] for l < 16, sc[2g+1] for l >= 16).
    for (int half = 0; half < 2; ++half) {
      const int so = half * 8;
      for (int g = 0; g < 4; ++g) {
        const float* xg = x + half * 128 + g * 32;
        float s0 = 0.0f, s1 = 0.0f;
        for (int l = 0; l < 32; ++l) {
          int qv;
          switch (g) {
            case 0:  qv = (int)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32; break;
            case 1:  qv = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32; break;
            case 2:  qv = (int)((ql[l +  0] >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32; break;
            default: qv = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32; break;
          }
          const float t = (float)qv * xg[l];
          if (l < 16) s0 += t;
          else        s1 += t;
        }
        acc += d * ((float)sc[so + 2 * g] * s0 + (float)sc[so + 2 * g + 1] * s1);
      }
      ql += 64;
      qh += 32;
    }
    src += 210;
    x   += 256;
  }
  return acc;
}

float dot_f32(const float* a, const float* b, uint64_t n) {
  float acc = 0.0f;
  for (uint64_t i = 0; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

}  // namespace

float fp16_to_fp32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  const uint32_t exp  = (h >> 10) & 0x1F;
  const uint32_t man  = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;  // +-zero
    } else {
      // Subnormal: renormalise into a normal float.
      uint32_t e = 127 - 15 + 1, m = man;
      while ((m & 0x400) == 0) { m <<= 1; --e; }
      m &= 0x3FF;
      bits = sign | (e << 23) | (m << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (man << 13);  // inf / nan
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

uint16_t fp32_to_fp16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t man = x & 0x7FFFFF;
  if (exp <= 0) return (uint16_t)sign;
  if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00);
  return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

float bf16_to_fp32(uint16_t b) {
  uint32_t bits = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

float dequant_dot_f32(GgmlType type, const uint8_t* src, const float* x, uint64_t n) {
  // Unquantized types have no block loop worth the indirection.
  if (type == GgmlType::F32) {
    return dot_f32(reinterpret_cast<const float*>(src), x, n);
  }
  if (type == GgmlType::F16) {
    float acc = 0.0f;
    for (uint64_t i = 0; i < n; ++i) acc += fp16_to_fp32(rd16(src + 2 * i)) * x[i];
    return acc;
  }
  if (type == GgmlType::BF16) {
    float acc = 0.0f;
    for (uint64_t i = 0; i < n; ++i) acc += bf16_to_fp32(rd16(src + 2 * i)) * x[i];
    return acc;
  }

  const uint64_t bw = ggml_type_block_weights(type);
  const uint64_t bb = ggml_type_block_bytes(type);
  if (bw == 0 || bw > 256) fail(std::string("unusable block size for type ") + ggml_type_name(type));
  if (n % bw) fail(std::string("run is not a whole number of blocks for type ") + ggml_type_name(type));

  const uint64_t nb = n / bw;

  // The two K-quants that carry the model's weight mass take the fused path;
  // everything else stages a block and dots it.
  if (bw == 256) {
    if (type == GgmlType::Q4_K) return dot_q4_k(src, x, nb);
    if (type == GgmlType::Q6_K) return dot_q6_k(src, x, nb);
  }

  float buf[256];
  float acc = 0.0f;
  for (uint64_t b = 0; b < nb; ++b) {
    dequant_row_f32(type, src + b * bb, buf, bw);
    acc += dot_f32(buf, x + b * bw, bw);
  }
  return acc;
}

void dequant_row_f32(GgmlType type, const uint8_t* src, float* dst, uint64_t n) {
  switch (type) {
    case GgmlType::F32:
      std::memcpy(dst, src, n * 4);
      return;
    case GgmlType::F16:
      for (uint64_t i = 0; i < n; ++i) dst[i] = fp16_to_fp32(rd16(src + 2 * i));
      return;
    case GgmlType::BF16:
      for (uint64_t i = 0; i < n; ++i) dst[i] = bf16_to_fp32(rd16(src + 2 * i));
      return;
    case GgmlType::Q4_0:
      if (n % 32) fail("Q4_0 run is not a multiple of 32");
      dequant_q4_0(src, dst, n);
      return;
    case GgmlType::Q8_0:
      if (n % 32) fail("Q8_0 run is not a multiple of 32");
      dequant_q8_0(src, dst, n);
      return;
    case GgmlType::Q4_K:
      if (n % 256) fail("Q4_K run is not a multiple of 256");
      dequant_q4_k(src, dst, n);
      return;
    case GgmlType::Q6_K:
      if (n % 256) fail("Q6_K run is not a multiple of 256");
      dequant_q6_k(src, dst, n);
      return;
    default:
      fail(std::string("no f32 decoder for type ") + ggml_type_name(type));
  }
}

}  // namespace knj
