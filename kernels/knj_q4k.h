// kernels/knj_q4k.h -- unpack GGUF Q4_K / Q6_K weights on the device.
//
// WHY THIS EXISTS, AND HOW IT DIFFERS FROM knj_pack.h
// ---------------------------------------------------
// knj_pack.h describes the PROJECT's own W4 g128 expert pack: 4 bits, group 128,
// 3 B/group metadata, a 68-byte group stride. That format is what the W3/W4
// design point is built on, and tools/bench/rocwmma_moe.hip proves a rocWMMA
// grouped expert GEMM reads it correctly.
//
// But the format the models on this machine ACTUALLY carry is GGUF Q4_K / Q6_K.
// Measuring a GEMM on a synthetic W4-g128 tensor of approximately the right
// shape is the "passes without measuring anything" class: the shape looks right
// and the bytes are not the bytes. So a device prefill number has to be produced
// from the real quantised blocks.
//
// The two formats are structurally different and it matters:
//
//   W4 g128   group 128, one f16 scale + one u8 zero point per group, unsigned
//             codes, w = (q - zp) * s                 -> 0.5313 B/weight @ 4 bits
//   Q4_K      256-weight superblock = f16 d + f16 dmin + 12 B of packed 6-bit
//             scales/mins + 128 B of 4-bit codes; EIGHT 32-weight sub-blocks,
//             each with its own scale AND its own min:
//                 w = d1*q - m1,  d1 = d*sc, m1 = dmin*m
//             -> 144 B / 256 weights = 0.5625 B/weight
//   Q6_K      256-weight superblock = 128 B ql + 64 B qh + 16 B int8 scales
//             + f16 d; SIX bits per weight as (4 low bits | 2 high bits), the
//             code biased by -32:
//                 w = d * sc * q,  q in [-32, 31]
//             -> 210 B / 256 weights = 0.8203 B/weight
//
// Q4_K is ASYMMETRIC (a real per-sub-block min, not a zero point) and per-32
// rather than per-128. Neither the scale nor the grouping transfers from the W4
// pack, which is exactly why reusing the W4 kernel here would have been wrong.
//
// DELIBERATELY HIP-FREE. The repo keeps its tier-A-compilable headers free of
// <hip/hip_runtime.h> so the host tier can include them, and fp16 decode is done
// here in integer bit manipulation rather than via __half2float. That also makes
// the reference implementation of the formula available to host-side checks
// without a GPU in the loop.
//
// "HIP-FREE" MEANS NO HIP HEADER, NOT HOST-ONLY. The decoders are the ones the
// DEVICE kernels call -- that is the whole point: one implementation of the
// format, exercised on the host by tools/q4k_probe.cpp and on the device by
// tools/bench/q4k_moe_ffn.hip, so a host/device disagreement is impossible by
// construction and a wrong decode cannot be masked by a second, different one.
// So the functions carry KNJ_HD, which expands to __host__ __device__ under
// hipcc and to plain `static inline` everywhere else. A host-only compiler
// never sees a device attribute and still compiles this file; hipcc gets a
// device-callable version of exactly the same source. Nothing here includes
// <hip/hip_runtime.h>, so tier A's `g++ -std=c++17 -I. tools/q4k_probe.cpp`
// keeps working unchanged.
//
// The offsets below are normative and were transcribed from ggml's own
// dequantise_row_q4_K / dequantise_row_q6_K (llama.cpp ggml/src/ggml-quants.c)
// and cross-checked against gguf-py's quants.py (Q4_K at :476, Q6_K at :553).

#ifndef KNJ_Q4K_H
#define KNJ_Q4K_H

#include <stdint.h>

// Superblock geometry. QK_K is 256 for both K-quants here.
#define KNJ_QK_K      256
#define KNJ_Q4K_BYTES 144
#define KNJ_Q6K_BYTES 210

// Q4_K block offsets.
#define KNJ_Q4K_OFF_D     0    // f16  d    (super-block scale for the scales)
#define KNJ_Q4K_OFF_DMIN  2    // f16  dmin (super-block scale for the mins)
#define KNJ_Q4K_OFF_SCALE 4    // u8[12]    packed 6-bit scales and mins
#define KNJ_Q4K_OFF_QS    16   // u8[128]   4-bit codes, low nibble first

// Q6_K block offsets.
#define KNJ_Q6K_OFF_QL    0    // u8[128]  low 4 bits
#define KNJ_Q6K_OFF_QH    128  // u8[64]   high 2 bits
#define KNJ_Q6K_OFF_SC    192  // i8[16]   per-16 sub-block scales
#define KNJ_Q6K_OFF_D     208  // f16      super-block scale

// ---------------------------------------------------------------------------
// Host/device decoration. Under hipcc the decoders must be callable from
// __global__ code; everywhere else the attribute does not exist. `static inline`
// keeps them internal-linkage in both worlds.
// ---------------------------------------------------------------------------
#if defined(__HIPCC__) || defined(__CUDACC__)
#define KNJ_HD __host__ __device__ static inline
#else
#define KNJ_HD static inline
#endif

// ---------------------------------------------------------------------------
// fp16 -> fp32, exact, integer bit manipulation. Round-trip identical to
// __half2float for every bit pattern including subnormals and NaN/Inf payloads.
// ---------------------------------------------------------------------------
KNJ_HD float knj_half_bits_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exp  = (uint32_t)((h >> 10) & 0x1f);
    const uint32_t man  = (uint32_t)(h & 0x3ff);
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;                       // +-0
        } else {
            // Subnormal: normalise so the implicit leading 1 lands at bit 10.
            int e = -1;
            uint32_t m = man;
            do { e++; m <<= 1; } while ((m & 0x400u) == 0);
            bits = sign | (uint32_t)(127 - 15 - e) << 23 | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);  // Inf / NaN
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

// Unaligned little-endian u16 load. The quantised payloads are packed, so the
// metadata is NOT 2-byte aligned in general (a Q6_K block is 210 bytes long).
KNJ_HD uint16_t knj_load_u16(const uint8_t *p) {
    uint16_t v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

// ---------------------------------------------------------------------------
// Q4_K: the packed 6-bit scale/min pair for sub-block j (0..7).
//
// The 12 bytes hold 8 six-bit scales in scales[0..7] and 8 six-bit mins in
// scales[4..11], but the high bits of scales 4..7 are STOLEN from the top two
// bits of scales 0..3 (and likewise for the mins). That is the whole reason this
// function exists rather than a plain array lookup.
// ---------------------------------------------------------------------------
KNJ_HD void knj_q4k_scale_min(const uint8_t *scales, int j,
                              uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = (uint8_t)(scales[j] & 63u);
        *m = (uint8_t)(scales[j + 4] & 63u);
    } else {
        *d = (uint8_t)((scales[j + 4] & 0x0Fu) | (uint8_t)((scales[j - 4] >> 6) << 4));
        *m = (uint8_t)((scales[j + 4] >> 4)    | (uint8_t)((scales[j - 0] >> 6) << 4));
    }
}

// Q4_K weight at superblock-local index r in [0, 256).
KNJ_HD float knj_q4k_weight(const uint8_t *blk, int r) {
    const float d    = knj_half_bits_to_float(knj_load_u16(blk + KNJ_Q4K_OFF_D));
    const float dmin = knj_half_bits_to_float(knj_load_u16(blk + KNJ_Q4K_OFF_DMIN));
    const uint8_t *scales = blk + KNJ_Q4K_OFF_SCALE;
    const uint8_t *qs     = blk + KNJ_Q4K_OFF_QS;

    const int j = r >> 5;   // sub-block 0..7 (32 weights each, 8 per superblock)
    const int l = r & 31;   // position inside the sub-block

    uint8_t sc, mn;
    knj_q4k_scale_min(scales, j, &sc, &mn);

    // Codes are stored as 4 groups of 32 bytes; group (j/2) holds sub-blocks
    // (2g, 2g+1) as its low and high nibbles respectively.
    const uint8_t byte = qs[(j >> 1) * 32 + l];
    const float q = (float)((j & 1) ? (uint32_t)(byte >> 4) : (uint32_t)(byte & 0x0f));

    const float d1 = d * (float)sc;
    const float m1 = dmin * (float)mn;
    return d1 * q - m1;
}

// Q6_K weight at superblock-local index r in [0, 256).
KNJ_HD float knj_q6k_weight(const uint8_t *blk, int r) {
    const float d = knj_half_bits_to_float(knj_load_u16(blk + KNJ_Q6K_OFF_D));
    const uint8_t *ql = blk + KNJ_Q6K_OFF_QL;
    const uint8_t *qh = blk + KNJ_Q6K_OFF_QH;
    const int8_t  *sc = (const int8_t *)(blk + KNJ_Q6K_OFF_SC);

    // The 256 weights are processed as two 128-blocks; each consumes 64 ql
    // bytes, 32 qh bytes and 8 scales.
    const int half = r >> 7;      // 0 or 1
    const int t    = r & 127;
    const int l    = t & 31;
    const int qtr  = t >> 5;      // 0..3, the four 32-wide output groups
    const int is   = l >> 4;      // 0 or 1, the 16-wide scale granularity

    const uint8_t *qlb = ql + half * 64;
    const uint8_t *qhb = qh + half * 32;
    const int8_t  *scb = sc + half * 8;

    // groups 0 and 2 read the low nibble of ql[l] / ql[l+32]; groups 1 and 3 the
    // high nibble. Which of ql[l] vs ql[l+32] depends on the group.
    const uint32_t lo  = (qtr & 1) ? (uint32_t)qlb[l + 32] : (uint32_t)qlb[l];
    const uint32_t nib = (qtr < 2) ? (lo & 0x0fu) : (lo >> 4);
    const uint32_t hi  = (uint32_t)((qhb[l] >> (2 * qtr)) & 3u);

    const int q = (int)(nib | (hi << 4)) - 32;   // biased by -32, range [-32, 31]
    return d * (float)scb[is + 2 * qtr] * (float)q;
}

// ---------------------------------------------------------------------------
// The 16-wide span decoders. A rocWMMA tile is 16x16, and a `load_matrix_sync`
// needs 16 consecutive weights per row, so the kernel must produce 16 weights at
// a time -- always starting at an even multiple of 16. That is what makes the
// hoist below correct rather than lucky:
//
//   Q4_K  sub-blocks are 32 wide and a span starts at r0 with r0 % 16 == 0, so
//         r0 % 32 is 0 or 16 and the span [r0, r0+16) never leaves its sub-block.
//         One (scale, min) pair therefore covers the whole span.
//   Q6_K  the 16-wide scale granularity has the same property: l = r0 % 32 is 0
//         or 16, so `is` is constant across the span and one scale covers it.
//
// These functions are the ones the device kernel calls, and q4k_probe.cpp
// checks them element-by-element against the per-element decoders above on every
// block of every real tensor -- on the host, with no GPU involved. So a device
// disagreement is a tile-layout disagreement, not a misread of the format.
// ---------------------------------------------------------------------------
KNJ_HD void knj_q4k_span16(const uint8_t *blk, int r0, float *out) {
    const float d    = knj_half_bits_to_float(knj_load_u16(blk + KNJ_Q4K_OFF_D));
    const float dmin = knj_half_bits_to_float(knj_load_u16(blk + KNJ_Q4K_OFF_DMIN));
    const uint8_t *scales = blk + KNJ_Q4K_OFF_SCALE;
    const int j = r0 >> 5;
    const int l = r0 & 31;
    uint8_t sc, mn;
    knj_q4k_scale_min(scales, j, &sc, &mn);
    const float d1 = d * (float)sc;
    const float m1 = dmin * (float)mn;
    const uint8_t *base = blk + KNJ_Q4K_OFF_QS + (j >> 1) * 32;
    for (int i = 0; i < 16; ++i) {
        const uint8_t byte = base[l + i];
        const uint32_t q = (j & 1) ? (uint32_t)(byte >> 4) : (uint32_t)(byte & 0x0f);
        out[i] = d1 * (float)q - m1;
    }
}

KNJ_HD void knj_q6k_span16(const uint8_t *blk, int r0, float *out) {
    const float d = knj_half_bits_to_float(knj_load_u16(blk + KNJ_Q6K_OFF_D));
    const uint8_t *ql = blk + KNJ_Q6K_OFF_QL;
    const uint8_t *qh = blk + KNJ_Q6K_OFF_QH;
    const int8_t  *sc = (const int8_t *)(blk + KNJ_Q6K_OFF_SC);

    const int hlf = r0 >> 7;
    const int t   = r0 & 127;
    const int qtr = t >> 5;
    const int l   = t & 31;
    const int is  = l >> 4;

    const uint8_t *qlb = ql + hlf * 64;
    const uint8_t *qhb = qh + hlf * 32;
    const float s = d * (float)sc[hlf * 8 + is + 2 * qtr];

    for (int i = 0; i < 16; ++i) {
        const uint32_t lo  = (qtr & 1) ? (uint32_t)qlb[l + i + 32] : (uint32_t)qlb[l + i];
        const uint32_t nib = (qtr < 2) ? (lo & 0x0fu) : (lo >> 4);
        const uint32_t hi  = (uint32_t)((qhb[l + i] >> (2 * qtr)) & 3u);
        const int q = (int)(nib | (hi << 4)) - 32;
        out[i] = s * (float)q;
    }
}

#undef KNJ_HD

#endif  // KNJ_Q4K_H
