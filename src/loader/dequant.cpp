// src/loader/dequant.cpp -- GGML block formats to f32.
//
// Every routine here is a transcription of ggml's reference dequantiser, which
// is the only definition that matters: these are the bytes llama.cpp wrote.
// Where a formula looks like it "should" be simpler, the reference is followed
// anyway, because the packing is not uniform across sub-blocks.
#include "src/loader/dequant.h"
#include "src/loader/iq_grids.h"

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

float rd32f(const uint8_t* p) {
  float v;
  std::memcpy(&v, p, 4);
  return v;
}

// E8M0 -> float, halved. Transcribed from ggml_e8m0_to_fp32_half
// (ggml-impl.h): the halving matches the doubled kvalues_fp4 table.
float e8m0_to_fp32_half(uint8_t x) {
  uint32_t bits;
  if (x < 2) {
    bits = (uint32_t)0x00200000 << x;
  } else {
    bits = (uint32_t)(x - 1) << 23;
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Unsigned E4M3 (bias 7) -> float, halved. Transcribed from
// ggml_ue4m3_to_fp32 (ggml-impl.h), which returns value * 0.5 for the
// same doubled-kvalues convention.
float ue4m3_to_fp32(uint8_t x) {
  if (x == 0 || x == 0x7F) return 0.0f;
  const int exp = (x >> 3) & 0xF;
  const int man = x & 0x7;
  float raw;
  if (exp == 0) {
    raw = std::ldexp((float)man, -9);
  } else {
    raw = std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
  }
  return raw * 0.5f;
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

// block_q5_K: { half d; half dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]; } = 176 B
//
// Eight 32-weight sub-blocks, and the fifth bit of weight w lives in
// qh[w % 32] at bit (w / 32): the 32 qh bytes carry exactly one bit per weight,
// which is where the 5 in Q5_K comes from. That bit mapping was DERIVED from
// gguf-py's decoder on this machine's own Q5_K tensors (recover the integer q
// from gguf-py's output, then match it against the raw bytes) and is verified
// byte-for-byte against gguf-py by tools/dequant_crosscheck.py -- it is not a
// transcription from memory. The scale/min packing is Q4_K's get_scale_min_k4.
void dequant_q5_k(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d    = fp16_to_fp32(rd16(src + 0));
    const float dmin = fp16_to_fp32(rd16(src + 2));
    const uint8_t* scales = src + 4;
    const uint8_t* qh = src + 16;
    const uint8_t* qs = src + 48;

    for (int sb = 0; sb < 8; ++sb) {
      uint8_t sc, m;
      get_scale_min_k4(sb, scales, &sc, &m);
      const float d1  = d * (float)sc;
      const float dm1 = dmin * (float)m;
      const uint8_t* q = qs + (sb / 2) * 32;
      const bool high = (sb & 1) != 0;
      for (int l = 0; l < 32; ++l) {
        const int lo = high ? (int)(q[l] >> 4) : (int)(q[l] & 0x0F);
        const int hi = (int)((qh[l] >> sb) & 1);
        dst[l] = d1 * (float)(lo | (hi << 4)) - dm1;
      }
      dst += 32;
    }
    src += 176;
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

// block_q4_1: { half d; half m; uint8 qs[16]; } = 20 B
// Transcribed from dequantize_row_q4_1 (ggml-quants.c).
void dequant_q4_1(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 32;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    const float m = fp16_to_fp32(rd16(src + 2));
    const uint8_t* qs = src + 4;
    for (int l = 0; l < 16; ++l) {
      dst[l]      = (float)(qs[l] & 0x0F) * d + m;
      dst[l + 16] = (float)(qs[l] >> 4) * d + m;
    }
    src += 20;
    dst += 32;
  }
}

// block_q5_0: { half d; uint8 qh[4]; uint8 qs[16]; } = 22 B
// The 5th bit of weight j lives in qh bit j (low half) or bit j-4+16
// (high half). Transcribed from dequantize_row_q5_0.
void dequant_q5_0(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 32;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    uint32_t qh;
    std::memcpy(&qh, src + 2, 4);
    const uint8_t* qs = src + 6;
    for (int l = 0; l < 16; ++l) {
      const int xh0 = (int)(((qh >> (l +  0)) << 4) & 0x10);
      const int xh1 = (int)(((qh >> (l + 12))     ) & 0x10);
      dst[l]      = (float)(((int)(qs[l] & 0x0F) | xh0) - 16) * d;
      dst[l + 16] = (float)(((int)(qs[l] >> 4)   | xh1) - 16) * d;
    }
    src += 22;
    dst += 32;
  }
}

// block_q5_1: { half d; half m; uint8 qh[4]; uint8 qs[16]; } = 24 B
// Transcribed from dequantize_row_q5_1.
void dequant_q5_1(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 32;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    const float m = fp16_to_fp32(rd16(src + 2));
    uint32_t qh;
    std::memcpy(&qh, src + 4, 4);
    const uint8_t* qs = src + 8;
    for (int l = 0; l < 16; ++l) {
      const int xh0 = (int)(((qh >> (l +  0)) << 4) & 0x10);
      const int xh1 = (int)(((qh >> (l + 12))     ) & 0x10);
      dst[l]      = (float)((int)(qs[l] & 0x0F) | xh0) * d + m;
      dst[l + 16] = (float)((int)(qs[l] >> 4)   | xh1) * d + m;
    }
    src += 24;
    dst += 32;
  }
}

// block_q8_1: { half d; half s; int8 qs[32]; } = 36 B
// Upstream ggml has NO scalar dequantizer for Q8_1 (its header declares it
// commented-out), so there is nothing to transcribe. The value semantics are
// DERIVED from the reference quantizer (quantize_row_q8_1_ref,
// ggml-quants.c): qs[j] = round(x[j]/d) and s = sum(qs)*d, hence
// y[j] = d*qs[j] reconstructs x and s is a dot-product shortcut that decode
// must ignore. Q8_1 is all but absent from real GGUFs; this decoder exists so
// the type is supported, not skipped, and it is validated by exact
// construction (hand-built blocks must decode bit-exactly).
void dequant_q8_1(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 32;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    const int8_t* qs = reinterpret_cast<const int8_t*>(src + 4);
    for (int l = 0; l < 32; ++l) dst[l] = d * (float)qs[l];
    src += 36;
    dst += 32;
  }
}

// block_q1_0: { half d; uint8 qs[16]; } = 18 B, 128 weights, 1 bit each.
// Transcribed from dequantize_row_q1_0.
void dequant_q1_0(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 128;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    const uint8_t* qs = src + 2;
    for (int l = 0; l < 128; ++l) {
      const int bit = (qs[l / 8] >> (l % 8)) & 1;
      dst[l] = bit ? d : -d;
    }
    src += 18;
    dst += 128;
  }
}

// block_q2_K: { uint8 scales[16]; uint8 qs[64]; half d; half dmin; } = 84 B
// NOTE the field order: scales and qs come FIRST, the super-block scales
// last. This differs from the Q4_K-family layout and is exactly what
// ggml-common.h's block_q2_K (plus gguf-py's splitter) says. Transcribed
// from dequantize_row_q2_K (ggml-quants.c): sixteen 16-weight sub-blocks,
// each sub-block's scale/min nibble from scales[is++], the 2-bit quants
// for the two 16-weight halves 16 bytes apart.
void dequant_q2_k(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* scales = src;
    const uint8_t* qs = src + 16;
    const float d   = fp16_to_fp32(rd16(src + 80));
    const float min = fp16_to_fp32(rd16(src + 82));
    int is = 0;
    for (int nn = 0; nn < 256; nn += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        uint8_t sc = scales[is++];
        float dl = d * (float)(sc & 0xF), ml = min * (float)(sc >> 4);
        for (int l = 0; l < 16; ++l) dst[l] = dl * (float)((qs[l] >> shift) & 3) - ml;
        dst += 16;
        sc = scales[is++];
        dl = d * (float)(sc & 0xF); ml = min * (float)(sc >> 4);
        for (int l = 0; l < 16; ++l) dst[l] = dl * (float)((qs[l + 16] >> shift) & 3) - ml;
        dst += 16;
        shift += 2;
      }
      qs += 32;
    }
    src += 84;
  }
}

// block_q3_K: { uint8 hmask[32]; uint8 qs[64]; uint8 scales[12]; half d; }
// = 110 B. Transcribed from dequantize_row_q3_K. The 12 scale bytes unpack
// into sixteen 6-bit scales (bias 32) through the aux dance below -- copied
// literally, because the packing is not stride-regular.
void dequant_q3_k(const uint8_t* src, float* dst, uint64_t n) {
  static const uint32_t kmask1 = 0x03030303;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d_all = fp16_to_fp32(rd16(src + 108));
    const uint8_t* hm = src;         // ggml block_q3_K: hmask[32] FIRST ...
    const uint8_t* qs = src + 32;    // ... then qs[64] (x+32 in ggml), scales[12], d
    uint32_t aux[4];
    std::memcpy(aux, src + 96, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
    int is = 0;
    uint8_t m = 1;
    for (int nn = 0; nn < 256; nn += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        float dl = d_all * (float)(scales[is++] - 32);
        for (int l = 0; l < 16; ++l)
          dst[l] = dl * (float)((int)((qs[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
        dst += 16;
        dl = d_all * (float)(scales[is++] - 32);
        for (int l = 0; l < 16; ++l)
          dst[l] = dl * (float)((int)((qs[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
        dst += 16;
        shift += 2;
        m <<= 1;
      }
      qs += 32;
    }
    src += 110;
  }
}

// block_q8_K: { float d; int8 qs[256]; int16 bsums[16]; } = 292 B.
// Transcribed from dequantize_row_q8_K. bsums are a dot-product shortcut;
// decode ignores them. Note d is float here, not fp16.
void dequant_q8_k(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = rd32f(src);
    const int8_t* qs = reinterpret_cast<const int8_t*>(src + 4);
    for (int l = 0; l < 256; ++l) dst[l] = d * (float)qs[l];
    src += 292;
    dst += 256;
  }
}

// block_mxfp4: { uint8 e (E8M0); uint8 qs[16]; } = 17 B, 32 weights.
// Transcribed from dequantize_row_mxfp4. Values come from the shared
// doubled E2M1 table (knj_kvalues_fp4); the single E8M0 scale is halved to
// match the doubling.
void dequant_mxfp4(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 32;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = e8m0_to_fp32_half(src[0]);
    const uint8_t* qs = src + 1;
    for (int l = 0; l < 16; ++l) {
      dst[l]      = (float)knj_kvalues_fp4[qs[l] & 0x0F] * d;
      dst[l + 16] = (float)knj_kvalues_fp4[qs[l] >> 4] * d;
    }
    src += 17;
    dst += 32;
  }
}

// block_nvfp4: { uint8 d[4] (UE4M3, one per 16-weight sub-block);
// uint8 qs[32]; } = 36 B, 64 weights. Transcribed from
// dequantize_row_nvfp4. Same value table as MXFP4; per-sub-block scales.
void dequant_nvfp4(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 64;
  for (uint64_t i = 0; i < nb; ++i) {
    for (int s = 0; s < 4; ++s) {
      const float d = ue4m3_to_fp32(src[s]);
      const uint8_t* qs = src + 4 + s * 8;
      float* yb = dst + s * 16;
      for (int l = 0; l < 8; ++l) {
        yb[l]     = (float)knj_kvalues_fp4[qs[l] & 0x0F] * d;
        yb[l + 8] = (float)knj_kvalues_fp4[qs[l] >> 4] * d;
      }
    }
    src += 36;
    dst += 64;
  }
}

// --- ROCmFP family decoders + scale tables ---
static const float knj_rocmfp4_scale_ue4m3_half[127] = {
  0x0.0000000000000p+0, 0x1.0000000000000p-10, 0x1.0000000000000p-9, 0x1.8000000000000p-9,
  0x1.0000000000000p-8, 0x1.4000000000000p-8, 0x1.8000000000000p-8, 0x1.c000000000000p-8,
  0x1.0000000000000p-7, 0x1.2000000000000p-7, 0x1.4000000000000p-7, 0x1.6000000000000p-7,
  0x1.8000000000000p-7, 0x1.a000000000000p-7, 0x1.c000000000000p-7, 0x1.e000000000000p-7,
  0x1.0000000000000p-6, 0x1.2000000000000p-6, 0x1.4000000000000p-6, 0x1.6000000000000p-6,
  0x1.8000000000000p-6, 0x1.a000000000000p-6, 0x1.c000000000000p-6, 0x1.e000000000000p-6,
  0x1.0000000000000p-5, 0x1.2000000000000p-5, 0x1.4000000000000p-5, 0x1.6000000000000p-5,
  0x1.8000000000000p-5, 0x1.a000000000000p-5, 0x1.c000000000000p-5, 0x1.e000000000000p-5,
  0x1.0000000000000p-4, 0x1.2000000000000p-4, 0x1.4000000000000p-4, 0x1.6000000000000p-4,
  0x1.8000000000000p-4, 0x1.a000000000000p-4, 0x1.c000000000000p-4, 0x1.e000000000000p-4,
  0x1.0000000000000p-3, 0x1.2000000000000p-3, 0x1.4000000000000p-3, 0x1.6000000000000p-3,
  0x1.8000000000000p-3, 0x1.a000000000000p-3, 0x1.c000000000000p-3, 0x1.e000000000000p-3,
  0x1.0000000000000p-2, 0x1.2000000000000p-2, 0x1.4000000000000p-2, 0x1.6000000000000p-2,
  0x1.8000000000000p-2, 0x1.a000000000000p-2, 0x1.c000000000000p-2, 0x1.e000000000000p-2,
  0x1.0000000000000p-1, 0x1.2000000000000p-1, 0x1.4000000000000p-1, 0x1.6000000000000p-1,
  0x1.8000000000000p-1, 0x1.a000000000000p-1, 0x1.c000000000000p-1, 0x1.e000000000000p-1,
  0x1.0000000000000p+0, 0x1.2000000000000p+0, 0x1.4000000000000p+0, 0x1.6000000000000p+0,
  0x1.8000000000000p+0, 0x1.a000000000000p+0, 0x1.c000000000000p+0, 0x1.e000000000000p+0,
  0x1.0000000000000p+1, 0x1.2000000000000p+1, 0x1.4000000000000p+1, 0x1.6000000000000p+1,
  0x1.8000000000000p+1, 0x1.a000000000000p+1, 0x1.c000000000000p+1, 0x1.e000000000000p+1,
  0x1.0000000000000p+2, 0x1.2000000000000p+2, 0x1.4000000000000p+2, 0x1.6000000000000p+2,
  0x1.8000000000000p+2, 0x1.a000000000000p+2, 0x1.c000000000000p+2, 0x1.e000000000000p+2,
  0x1.0000000000000p+3, 0x1.2000000000000p+3, 0x1.4000000000000p+3, 0x1.6000000000000p+3,
  0x1.8000000000000p+3, 0x1.a000000000000p+3, 0x1.c000000000000p+3, 0x1.e000000000000p+3,
  0x1.0000000000000p+4, 0x1.2000000000000p+4, 0x1.4000000000000p+4, 0x1.6000000000000p+4,
  0x1.8000000000000p+4, 0x1.a000000000000p+4, 0x1.c000000000000p+4, 0x1.e000000000000p+4,
  0x1.0000000000000p+5, 0x1.2000000000000p+5, 0x1.4000000000000p+5, 0x1.6000000000000p+5,
  0x1.8000000000000p+5, 0x1.a000000000000p+5, 0x1.c000000000000p+5, 0x1.e000000000000p+5,
  0x1.0000000000000p+6, 0x1.2000000000000p+6, 0x1.4000000000000p+6, 0x1.6000000000000p+6,
  0x1.8000000000000p+6, 0x1.a000000000000p+6, 0x1.c000000000000p+6, 0x1.e000000000000p+6,
  0x1.0000000000000p+7, 0x1.2000000000000p+7, 0x1.4000000000000p+7, 0x1.6000000000000p+7,
  0x1.8000000000000p+7, 0x1.a000000000000p+7, 0x1.c000000000000p+7
};

static const float knj_rocmfpx_scale_ue4m3[127] = {
  0x0.0000000000000p+0, 0x1.0000000000000p-10, 0x1.0000000000000p-9, 0x1.8000000000000p-9,
  0x1.0000000000000p-8, 0x1.4000000000000p-8, 0x1.8000000000000p-8, 0x1.c000000000000p-8,
  0x1.0000000000000p-7, 0x1.2000000000000p-7, 0x1.4000000000000p-7, 0x1.6000000000000p-7,
  0x1.8000000000000p-7, 0x1.a000000000000p-7, 0x1.c000000000000p-7, 0x1.e000000000000p-7,
  0x1.0000000000000p-6, 0x1.2000000000000p-6, 0x1.4000000000000p-6, 0x1.6000000000000p-6,
  0x1.8000000000000p-6, 0x1.a000000000000p-6, 0x1.c000000000000p-6, 0x1.e000000000000p-6,
  0x1.0000000000000p-5, 0x1.2000000000000p-5, 0x1.4000000000000p-5, 0x1.6000000000000p-5,
  0x1.8000000000000p-5, 0x1.a000000000000p-5, 0x1.c000000000000p-5, 0x1.e000000000000p-5,
  0x1.0000000000000p-4, 0x1.2000000000000p-4, 0x1.4000000000000p-4, 0x1.6000000000000p-4,
  0x1.8000000000000p-4, 0x1.a000000000000p-4, 0x1.c000000000000p-4, 0x1.e000000000000p-4,
  0x1.0000000000000p-3, 0x1.2000000000000p-3, 0x1.4000000000000p-3, 0x1.6000000000000p-3,
  0x1.8000000000000p-3, 0x1.a000000000000p-3, 0x1.c000000000000p-3, 0x1.e000000000000p-3,
  0x1.0000000000000p-2, 0x1.2000000000000p-2, 0x1.4000000000000p-2, 0x1.6000000000000p-2,
  0x1.8000000000000p-2, 0x1.a000000000000p-2, 0x1.c000000000000p-2, 0x1.e000000000000p-2,
  0x1.0000000000000p-1, 0x1.2000000000000p-1, 0x1.4000000000000p-1, 0x1.6000000000000p-1,
  0x1.8000000000000p-1, 0x1.a000000000000p-1, 0x1.c000000000000p-1, 0x1.e000000000000p-1,
  0x1.0000000000000p+0, 0x1.2000000000000p+0, 0x1.4000000000000p+0, 0x1.6000000000000p+0,
  0x1.8000000000000p+0, 0x1.a000000000000p+0, 0x1.c000000000000p+0, 0x1.e000000000000p+0,
  0x1.0000000000000p+1, 0x1.2000000000000p+1, 0x1.4000000000000p+1, 0x1.6000000000000p+1,
  0x1.8000000000000p+1, 0x1.a000000000000p+1, 0x1.c000000000000p+1, 0x1.e000000000000p+1,
  0x1.0000000000000p+2, 0x1.2000000000000p+2, 0x1.4000000000000p+2, 0x1.6000000000000p+2,
  0x1.8000000000000p+2, 0x1.a000000000000p+2, 0x1.c000000000000p+2, 0x1.e000000000000p+2,
  0x1.0000000000000p+3, 0x1.2000000000000p+3, 0x1.4000000000000p+3, 0x1.6000000000000p+3,
  0x1.8000000000000p+3, 0x1.a000000000000p+3, 0x1.c000000000000p+3, 0x1.e000000000000p+3,
  0x1.0000000000000p+4, 0x1.2000000000000p+4, 0x1.4000000000000p+4, 0x1.6000000000000p+4,
  0x1.8000000000000p+4, 0x1.a000000000000p+4, 0x1.c000000000000p+4, 0x1.e000000000000p+4,
  0x1.0000000000000p+5, 0x1.2000000000000p+5, 0x1.4000000000000p+5, 0x1.6000000000000p+5,
  0x1.8000000000000p+5, 0x1.a000000000000p+5, 0x1.c000000000000p+5, 0x1.e000000000000p+5,
  0x1.0000000000000p+6, 0x1.2000000000000p+6, 0x1.4000000000000p+6, 0x1.6000000000000p+6,
  0x1.8000000000000p+6, 0x1.a000000000000p+6, 0x1.c000000000000p+6, 0x1.e000000000000p+6,
  0x1.0000000000000p+7, 0x1.2000000000000p+7, 0x1.4000000000000p+7, 0x1.6000000000000p+7,
  0x1.8000000000000p+7, 0x1.a000000000000p+7, 0x1.c000000000000p+7
};

// --- ROCmFP family (llama.cpp/ROCmFPX fork, GGML ids 100/101/102/103/104/107).
//
// Transcribed from ggml/rocmfp4/rocmfp4.c (rocmfp4_dequantize_row_q4_0[_fast],
// rocmfp4_decode, rocmfp4_ue4m3_to_fp32_half) and ggml/rocmfpx/rocmfpx.c
// (rocmfpx_dequantize_row_fp2/fp3/fp6/fp8, the unpack helpers, the code
// tables, rocmfpx_scale_lookup). UE4M3 scale tables below are bit-exact
// hexfloat dumps of the fork's compiled tables (tmp/attnrep/dump_tables.c),
// not re-derivations: e > 0x7E decodes as 0.0 in both tables, as in the fork.
//
// Layout note: low nibbles / first-half codes always pair with e[0], high
// nibbles / second-half codes with e[1] (fp8 has a single scale).

inline float rocmfp4_scale(uint8_t e) {
  return e <= 0x7E ? knj_rocmfp4_scale_ue4m3_half[e] : 0.0f;
}

inline float rocmfpx_scale(uint8_t e) {
  return e <= 0x7E ? knj_rocmfpx_scale_ue4m3[e] : 0.0f;
}

// E2M1-derived nibble with the largest magnitude retuned 12 -> 10
// (rocmfp4_decode; identical to the rocmfp4_codebook table).
inline int rocmfp4_icode(uint8_t q) {
  const int m3 = q & 7;
  const int mag = m3 <= 4 ? m3 : 2 * m3 - 4;
  return (q & 8) ? -mag : mag;
}

// block_q4_0_rocmfp4: { uint8 qs[16]; uint8 e[2]; } = 18 B, 32 weights.
void dequant_q4_0_rocmfp4(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 32;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d0 = rocmfp4_scale(src[16]);
    const float d1 = rocmfp4_scale(src[17]);
    for (int j = 0; j < 16; ++j) {
      dst[j]      = (float)rocmfp4_icode(src[j] & 0x0F) * d0;
      dst[j + 16] = (float)rocmfp4_icode(src[j] >> 4) * d1;
    }
    src += 18;
    dst += 32;
  }
}

// block_q4_0_rocmfp4_fast: { uint8 qs[16]; uint8 e; } = 17 B, 32 weights.
void dequant_q4_0_rocmfp4_fast(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 32;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = rocmfp4_scale(src[16]);
    for (int j = 0; j < 16; ++j) {
      dst[j]      = (float)rocmfp4_icode(src[j] & 0x0F) * d;
      dst[j + 16] = (float)rocmfp4_icode(src[j] >> 4) * d;
    }
    src += 17;
    dst += 32;
  }
}

// block_q2_0_rocmfpx: { uint8 qs[8]; uint8 e[2]; } = 10 B, 32 weights.
// Frozen codebook {-4, -1, 1, 4} (rocmfpx_decode_fp2_code; PR #42 freezes it).
void dequant_q2_0_rocmfpx(const uint8_t* src, float* dst, uint64_t n) {
  static const int vals[4] = {-4, -1, 1, 4};
  const uint64_t nb = n / 32;
  for (uint64_t i = 0; i < nb; ++i) {
    for (int half = 0; half < 2; ++half) {
      const float d = rocmfpx_scale(src[8 + half]);
      for (int j = 0; j < 16; ++j) {
        const uint8_t code = (src[half * 4 + j / 4] >> (2 * (j % 4))) & 3u;
        dst[half * 16 + j] = (float)vals[code] * d;
      }
    }
    src += 10;
    dst += 32;
  }
}

// FP3 unpack: 8 x 3-bit codes <- 3 bytes (rocmfpx_fp3_unpack8).
inline void rocmfpx_unpack8(const uint8_t* s, uint8_t* c) {
  c[0] =  s[0]        & 7u;
  c[1] = (s[0] >> 3)  & 7u;
  c[2] = ((s[0] >> 6) & 3u) | ((s[1] & 1u) << 2);
  c[3] = (s[1] >> 1)  & 7u;
  c[4] = (s[1] >> 4)  & 7u;
  c[5] = ((s[1] >> 7) & 1u) | ((s[2] & 3u) << 1);
  c[6] = (s[2] >> 2)  & 7u;
  c[7] = (s[2] >> 5)  & 7u;
}

// block_q3_0_rocmfpx: { uint8 qs[12]; uint8 e[2]; } = 14 B, 32 weights.
// 3-bit code: mag table {0,1,2,4}, bit 2 is the sign (rocmfpx_decode_fp3_code).
void dequant_q3_0_rocmfpx(const uint8_t* src, float* dst, uint64_t n) {
  static const int mag[4] = {0, 1, 2, 4};
  const uint64_t nb = n / 32;
  for (uint64_t i = 0; i < nb; ++i) {
    uint8_t codes[32];
    rocmfpx_unpack8(src, codes);
    rocmfpx_unpack8(src + 3, codes + 8);
    rocmfpx_unpack8(src + 6, codes + 16);
    rocmfpx_unpack8(src + 9, codes + 24);
    for (int half = 0; half < 2; ++half) {
      const float d = rocmfpx_scale(src[12 + half]);
      for (int j = 0; j < 16; ++j) {
        const int idx = half * 16 + j;
        const int v = mag[codes[idx] & 3u];
        dst[idx] = (float)((codes[idx] & 4u) ? -v : v) * d;
      }
    }
    src += 14;
    dst += 32;
  }
}

// FP6 unpack: 4 x 6-bit codes <- 3 bytes (rocmfpx_fp6_unpack4).
inline void rocmfpx_unpack4(const uint8_t* s, uint8_t* c) {
  c[0] =  s[0]         & 0x3Fu;
  c[1] = ((s[0] >> 6)  & 0x03u) | ((s[1] & 0x0Fu) << 2);
  c[2] = ((s[1] >> 4)  & 0x0Fu) | ((s[2] & 0x03u) << 4);
  c[3] =  (s[2] >> 2)  & 0x3Fu;
}

// block_q6_0_rocmfpx: { uint8 qs[24]; uint8 e[2]; } = 26 B, 32 weights.
// 6-bit code: low 5 bits magnitude, bit 5 sign; -0 decodes as -32
// (rocmfpx_decode_fp6_code: -(mag == 0 ? 32 : mag)).
void dequant_q6_0_rocmfpx(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 32;
  for (uint64_t i = 0; i < nb; ++i) {
    uint8_t codes[32];
    for (int g = 0; g < 8; ++g) rocmfpx_unpack4(src + 3 * g, codes + 4 * g);
    for (int half = 0; half < 2; ++half) {
      const float d = rocmfpx_scale(src[24 + half]);
      for (int j = 0; j < 16; ++j) {
        const int idx = half * 16 + j;
        const int m = codes[idx] & 31u;
        const int v = (codes[idx] & 32u) ? -(m == 0 ? 32 : m) : m;
        dst[idx] = (float)v * d;
      }
    }
    src += 26;
    dst += 32;
  }
}

// block_q8_0_rocmfpx: { int8 qs[32]; uint8 e; } = 33 B, 32 weights.
// Codes are the signed values themselves; one scale for the block.
void dequant_q8_0_rocmfpx(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 32;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = rocmfpx_scale(src[32]);
    const int8_t* qs = reinterpret_cast<const int8_t*>(src);
    for (int l = 0; l < 32; ++l) dst[l] = (float)qs[l] * d;
    src += 33;
    dst += 32;
  }
}

// block_tq1_0: { uint8 qs[48]; uint8 qh[4]; half d; } = 54 B, 256 weights.
// Ternary: 5 values per byte (3^5 = 243 < 256). Transcribed from
// dequantize_row_tq1_0. sizeof(qs)=48: the first loop covers bytes 0..31
// (32-wide), the second bytes 32..47 (16-wide); qh covers the last 16.
void dequant_tq1_0(const uint8_t* src, float* dst, uint64_t n) {
  static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src + 52));
    const uint8_t* qs = src;
    const uint8_t* qh = src + 48;
    for (int j = 0; j < 32; j += 32)
      for (int nn = 0; nn < 5; ++nn)
        for (int mm = 0; mm < 32; ++mm) {
          const uint8_t q = (uint8_t)(qs[j + mm] * pow3[nn]);
          dst[0] = (float)(((uint16_t)q * 3 >> 8) - 1) * d;
          ++dst;
        }
    for (int j = 32; j < 48; j += 16)
      for (int nn = 0; nn < 5; ++nn)
        for (int mm = 0; mm < 16; ++mm) {
          const uint8_t q = (uint8_t)(qs[j + mm] * pow3[nn]);
          dst[0] = (float)(((uint16_t)q * 3 >> 8) - 1) * d;
          ++dst;
        }
    for (int nn = 0; nn < 4; ++nn)
      for (int j = 0; j < 4; ++j) {
        const uint8_t q = (uint8_t)(qh[j] * pow3[nn]);
        dst[0] = (float)(((uint16_t)q * 3 >> 8) - 1) * d;
        ++dst;
      }
    src += 54;
  }
}

// block_tq2_0: { uint8 qs[64]; half d; } = 66 B, 256 weights, 2 bits each.
// Values are q-1, i.e. 00=-1, 01=0, 10=+1, 11=+2. Transcribed from
// dequantize_row_tq2_0.
void dequant_tq2_0(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src + 64));
    const uint8_t* qs = src;
    for (int j = 0; j < 64; j += 32)
      for (int l = 0; l < 4; ++l)
        for (int mm = 0; mm < 32; ++mm) {
          const int8_t q = (int8_t)((qs[j + mm] >> (l * 2)) & 3);
          dst[0] = (float)(q - 1) * d;
          ++dst;
        }
    src += 66;
  }
}

// --- IQ family --------------------------------------------------------
// All transcribed from the dequantize_row_iq* scalar path (ggml-quants.c).
// Codebooks come from src/loader/iq_grids.h, extracted mechanically from
// ggml-common.h (see tools/extract_iq_grids.py), never hand-copied. The
// grids are packed little-endian integers cast to bytes, exactly as ggml
// indexes them; this host is little-endian (x86_64).

void dequant_iq2_xxs(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    const uint8_t* qs = src + 2;
    for (int b = 0; b < 8; ++b) {
      uint32_t aux32[2];
      std::memcpy(aux32, qs + 8 * b, 8);
      const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
      const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
      for (int l = 0; l < 4; ++l) {
        const uint8_t* grid = reinterpret_cast<const uint8_t*>(knj_iq2xxs_grid + aux8[l]);
        const uint8_t signs = knj_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
        for (int j = 0; j < 8; ++j)
          dst[j] = db * (float)grid[j] * ((signs & knj_kmask_iq2xs[j]) ? -1.0f : 1.0f);
        dst += 8;
      }
    }
    src += 66;
  }
}

void dequant_iq2_xs(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    const uint8_t* qs = src + 2;
    const uint8_t* scales = src + 66;
    for (int b = 0; b < 8; ++b) {
      const float db0 = d * (0.5f + (float)(scales[b] & 0xF)) * 0.25f;
      const float db1 = d * (0.5f + (float)(scales[b] >> 4)) * 0.25f;
      for (int l = 0; l < 4; ++l) {
        uint16_t idx;
        std::memcpy(&idx, qs + 8 * b + 2 * l, 2);
        const uint8_t* grid =
            reinterpret_cast<const uint8_t*>(knj_iq2xs_grid + (idx & 511));
        const uint8_t signs = knj_ksigns_iq2xs[idx >> 9];
        const float db = (l / 2) ? db1 : db0;
        for (int j = 0; j < 8; ++j)
          dst[j] = db * (float)grid[j] * ((signs & knj_kmask_iq2xs[j]) ? -1.0f : 1.0f);
        dst += 8;
      }
    }
    src += 74;
  }
}

void dequant_iq2_s(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    const uint8_t* qs = src + 2;
    const uint8_t* qh = qs + 64;
    const uint8_t* signs0 = qs + 32;
    const uint8_t* scales = qh + 8;
    for (int b = 0; b < 8; ++b) {
      const float db0 = d * (0.5f + (float)(scales[b] & 0xF)) * 0.25f;
      const float db1 = d * (0.5f + (float)(scales[b] >> 4)) * 0.25f;
      for (int l = 0; l < 4; ++l) {
        const float dl = (l / 2) ? db1 : db0;
        const uint8_t* grid = reinterpret_cast<const uint8_t*>(
            knj_iq2s_grid + (qs[l] | ((qh[b] << (8 - 2 * l)) & 0x300)));
        for (int j = 0; j < 8; ++j)
          dst[j] = dl * (float)grid[j] *
                   ((signs0[l] & knj_kmask_iq2xs[j]) ? -1.0f : 1.0f);
        dst += 8;
      }
      qs += 4;
      signs0 += 4;
    }
    src += 82;
  }
}

void dequant_iq3_xxs(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    const uint8_t* qs = src + 2;
    const uint8_t* sands = qs + 64;
    for (int b = 0; b < 8; ++b) {
      uint32_t aux32;
      std::memcpy(&aux32, sands + 4 * b, 4);
      const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
      for (int l = 0; l < 4; ++l) {
        const uint8_t signs = knj_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
        const uint8_t* grid1 =
            reinterpret_cast<const uint8_t*>(knj_iq3xxs_grid + qs[2 * l]);
        const uint8_t* grid2 =
            reinterpret_cast<const uint8_t*>(knj_iq3xxs_grid + qs[2 * l + 1]);
        for (int j = 0; j < 4; ++j) {
          dst[j]     = db * (float)grid1[j] *
                       ((signs & knj_kmask_iq2xs[j]) ? -1.0f : 1.0f);
          dst[j + 4] = db * (float)grid2[j] *
                       ((signs & knj_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
        }
        dst += 8;
      }
      qs += 8;
    }
    src += 98;
  }
}

void dequant_iq3_s(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    const uint8_t* qs = src + 2;
    const uint8_t* qh = qs + 64;
    const uint8_t* signs0 = qh + 8;
    const uint8_t* scales = signs0 + 32;
    for (int b = 0; b < 8; b += 2) {
      const float db1 = d * (1.0f + 2.0f * (float)(scales[b / 2] & 0xF));
      const float db2 = d * (1.0f + 2.0f * (float)(scales[b / 2] >> 4));
      for (int half = 0; half < 2; ++half) {
        const float db = half ? db2 : db1;
        for (int l = 0; l < 4; ++l) {
          const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(
              knj_iq3s_grid + (qs[2 * l] | ((qh[half] << (8 - 2 * l)) & 256)));
          const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(
              knj_iq3s_grid + (qs[2 * l + 1] | ((qh[half] << (7 - 2 * l)) & 256)));
          for (int j = 0; j < 4; ++j) {
            dst[j]     = db * (float)grid1[j] *
                         ((signs0[l] & knj_kmask_iq2xs[j]) ? -1.0f : 1.0f);
            dst[j + 4] = db * (float)grid2[j] *
                         ((signs0[l] & knj_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
          }
          dst += 8;
        }
        qs += 8;
        signs0 += 4;
      }
      qh += 2;
    }
    src += 110;
  }
}

void dequant_iq1_s(const uint8_t* src, float* dst, uint64_t n) {
  static const float kDelta = 0.125f;
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    const uint8_t* qs = src + 2;
    const uint8_t* qh8 = qs + 32;
    for (int b = 0; b < 8; ++b) {
      uint16_t qh;
      std::memcpy(&qh, qh8 + 2 * b, 2);
      const float dl = d * (float)(2 * ((qh >> 12) & 7) + 1);
      const float delta = (qh & 0x8000) ? -kDelta : kDelta;
      for (int l = 0; l < 4; ++l) {
        const int8_t* grid = reinterpret_cast<const int8_t*>(
            knj_iq1s_grid + (qs[l] | (((qh >> (3 * l)) & 7) << 8)));
        for (int j = 0; j < 8; ++j) dst[j] = dl * ((float)grid[j] + delta);
        dst += 8;
      }
      qs += 4;
    }
    src += 50;
  }
}

void dequant_iq1_m(const uint8_t* src, float* dst, uint64_t n) {
  static const float kDelta = 0.125f;
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const uint8_t* qs = src;
    const uint8_t* qh = src + 32;
    const uint8_t* sc8 = src + 48;
    uint16_t sc[4];
    for (int k = 0; k < 4; ++k) std::memcpy(&sc[k], sc8 + 2 * k, 2);
    uint16_t su16 = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) |
                               ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
    const float d = fp16_to_fp32(su16);
    for (int b = 0; b < 8; ++b) {
      const float dl1 = d * (float)(2 * ((sc[b / 2] >> (6 * (b % 2))) & 0x7) + 1);
      const float dl2 = d * (float)(2 * ((sc[b / 2] >> (6 * (b % 2) + 3)) & 0x7) + 1);
      uint16_t idx[4];
      idx[0] = (uint16_t)(qs[0] | ((qh[0] << 8) & 0x700));
      idx[1] = (uint16_t)(qs[1] | ((qh[0] << 4) & 0x700));
      idx[2] = (uint16_t)(qs[2] | ((qh[1] << 8) & 0x700));
      idx[3] = (uint16_t)(qs[3] | ((qh[1] << 4) & 0x700));
      float delta[4];
      delta[0] = (qh[0] & 0x08) ? -kDelta : kDelta;
      delta[1] = (qh[0] & 0x80) ? -kDelta : kDelta;
      delta[2] = (qh[1] & 0x08) ? -kDelta : kDelta;
      delta[3] = (qh[1] & 0x80) ? -kDelta : kDelta;
      for (int l = 0; l < 4; ++l) {
        const float dl = (l < 2) ? dl1 : dl2;
        const int8_t* grid =
            reinterpret_cast<const int8_t*>(knj_iq1s_grid + idx[l]);
        for (int j = 0; j < 8; ++j) dst[j] = dl * ((float)grid[j] + delta[l]);
        dst += 8;
      }
      qs += 4;
      qh += 2;
    }
    src += 56;
  }
}

void dequant_iq4_nl(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 32;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    const uint8_t* qs = src + 2;
    for (int l = 0; l < 16; ++l) {
      dst[l]      = d * (float)knj_kvalues_iq4nl[qs[l] & 0x0F];
      dst[l + 16] = d * (float)knj_kvalues_iq4nl[qs[l] >> 4];
    }
    src += 18;
    dst += 32;
  }
}

void dequant_iq4_xs(const uint8_t* src, float* dst, uint64_t n) {
  const uint64_t nb = n / 256;
  for (uint64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(rd16(src));
    uint16_t sh;
    std::memcpy(&sh, src + 2, 2);
    const uint8_t* sl = src + 4;
    const uint8_t* qs = src + 8;
    for (int b = 0; b < 8; ++b) {
      const int ls = (int)(((sl[b / 2] >> (4 * (b % 2))) & 0xF) |
                           (((sh >> (2 * b)) & 3) << 4));
      const float dl = d * (float)(ls - 32);
      for (int l = 0; l < 16; ++l) {
        dst[l]      = dl * (float)knj_kvalues_iq4nl[qs[l] & 0x0F];
        dst[l + 16] = dl * (float)knj_kvalues_iq4nl[qs[l] >> 4];
      }
      dst += 32;
      qs += 16;
    }
    src += 136;
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
    case GgmlType::Q5_K:
      if (n % 256) fail("Q5_K run is not a multiple of 256");
      dequant_q5_k(src, dst, n);
      return;
    case GgmlType::Q6_K:
      if (n % 256) fail("Q6_K run is not a multiple of 256");
      dequant_q6_k(src, dst, n);
      return;
    case GgmlType::Q4_1:
      if (n % 32) fail("Q4_1 run is not a multiple of 32");
      dequant_q4_1(src, dst, n);
      return;
    case GgmlType::Q5_0:
      if (n % 32) fail("Q5_0 run is not a multiple of 32");
      dequant_q5_0(src, dst, n);
      return;
    case GgmlType::Q5_1:
      if (n % 32) fail("Q5_1 run is not a multiple of 32");
      dequant_q5_1(src, dst, n);
      return;
    case GgmlType::Q8_1:
      if (n % 32) fail("Q8_1 run is not a multiple of 32");
      dequant_q8_1(src, dst, n);
      return;
    case GgmlType::Q1_0:
      if (n % 128) fail("Q1_0 run is not a multiple of 128");
      dequant_q1_0(src, dst, n);
      return;
    case GgmlType::Q2_K:
      if (n % 256) fail("Q2_K run is not a multiple of 256");
      dequant_q2_k(src, dst, n);
      return;
    case GgmlType::Q3_K:
      if (n % 256) fail("Q3_K run is not a multiple of 256");
      dequant_q3_k(src, dst, n);
      return;
    case GgmlType::Q8_K:
      if (n % 256) fail("Q8_K run is not a multiple of 256");
      dequant_q8_k(src, dst, n);
      return;
    case GgmlType::Q4_0_ROCMFP4:
      if (n % 32) fail("Q4_0_ROCMFP4 run is not a multiple of 32");
      dequant_q4_0_rocmfp4(src, dst, n);
      return;
    case GgmlType::Q4_0_ROCMFP4_FAST:
      if (n % 32) fail("Q4_0_ROCMFP4_FAST run is not a multiple of 32");
      dequant_q4_0_rocmfp4_fast(src, dst, n);
      return;
    case GgmlType::Q2_0_ROCMFPX:
      if (n % 32) fail("Q2_0_ROCMFPX run is not a multiple of 32");
      dequant_q2_0_rocmfpx(src, dst, n);
      return;
    case GgmlType::Q3_0_ROCMFPX:
      if (n % 32) fail("Q3_0_ROCMFPX run is not a multiple of 32");
      dequant_q3_0_rocmfpx(src, dst, n);
      return;
    case GgmlType::Q6_0_ROCMFPX:
      if (n % 32) fail("Q6_0_ROCMFPX run is not a multiple of 32");
      dequant_q6_0_rocmfpx(src, dst, n);
      return;
    case GgmlType::Q8_0_ROCMFPX:
      if (n % 32) fail("Q8_0_ROCMFPX run is not a multiple of 32");
      dequant_q8_0_rocmfpx(src, dst, n);
      return;
    case GgmlType::MXFP4:
      if (n % 32) fail("MXFP4 run is not a multiple of 32");
      dequant_mxfp4(src, dst, n);
      return;
    case GgmlType::NVFP4:
      if (n % 64) fail("NVFP4 run is not a multiple of 64");
      dequant_nvfp4(src, dst, n);
      return;
    case GgmlType::TQ1_0:
      if (n % 256) fail("TQ1_0 run is not a multiple of 256");
      dequant_tq1_0(src, dst, n);
      return;
    case GgmlType::TQ2_0:
      if (n % 256) fail("TQ2_0 run is not a multiple of 256");
      dequant_tq2_0(src, dst, n);
      return;
    case GgmlType::IQ2_XXS:
      if (n % 256) fail("IQ2_XXS run is not a multiple of 256");
      dequant_iq2_xxs(src, dst, n);
      return;
    case GgmlType::IQ2_XS:
      if (n % 256) fail("IQ2_XS run is not a multiple of 256");
      dequant_iq2_xs(src, dst, n);
      return;
    case GgmlType::IQ2_S:
      if (n % 256) fail("IQ2_S run is not a multiple of 256");
      dequant_iq2_s(src, dst, n);
      return;
    case GgmlType::IQ3_XXS:
      if (n % 256) fail("IQ3_XXS run is not a multiple of 256");
      dequant_iq3_xxs(src, dst, n);
      return;
    case GgmlType::IQ3_S:
      if (n % 256) fail("IQ3_S run is not a multiple of 256");
      dequant_iq3_s(src, dst, n);
      return;
    case GgmlType::IQ1_S:
      if (n % 256) fail("IQ1_S run is not a multiple of 256");
      dequant_iq1_s(src, dst, n);
      return;
    case GgmlType::IQ1_M:
      if (n % 256) fail("IQ1_M run is not a multiple of 256");
      dequant_iq1_m(src, dst, n);
      return;
    case GgmlType::IQ4_NL:
      if (n % 32) fail("IQ4_NL run is not a multiple of 32");
      dequant_iq4_nl(src, dst, n);
      return;
    case GgmlType::IQ4_XS:
      if (n % 256) fail("IQ4_XS run is not a multiple of 256");
      dequant_iq4_xs(src, dst, n);
      return;
    default:
      fail(std::string("no f32 decoder for type ") + ggml_type_name(type));
  }
}

}  // namespace knj
