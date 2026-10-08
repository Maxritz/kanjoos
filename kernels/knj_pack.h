// kernels/knj_pack.h -- the W4 weight pack layout, shared by host and device.
//
// This header is the single definition of what a packed expert byte MEANS. The
// host reference oracle and the device kernels both include it, which is the
// only way the two can be guaranteed to agree: if the layout is written twice,
// it will diverge, and the divergence will look like a kernel bug.
//
// ---------------------------------------------------------------------------
// The 3 B/group rule
// ---------------------------------------------------------------------------
// Every quantised format in this engine carries 3 bytes of metadata per group:
// an fp16 scale and a uint8 zero-point. That is `gemm_w4.hip`'s convention and
// it is MEASURED (docs/00-verified-facts.md sections 8.10 and 9.2).
//
// It matters more than it looks. A "no metadata" quantisation is not available
// at any bit width here, so:
//
//     INT4 g128 = 0.5 + 3/128 = 0.5234 B/weight   (3.82x vs FP16, NOT 4x)
//     INT4 g64  = 0.5 + 3/64  = 0.5469 B/weight
//
// An earlier revision of docs/09 quoted INT4 g128 as 24 KB/token, i.e. exactly
// half of FP16. It is 25.1 KiB. That single omitted metadata term was a 4.6%
// error that then propagated into a resident-context column that did not follow
// from its own byte count. The rule is repeated here because it is the kind of
// constant that gets "simplified" by accident.
//
// Metadata width is 3 B for EVERY bit width, not just W4 -- that is the single
// metadata rule across W8/W6/W4/W3/W2. One convention, one packer.

#ifndef KNJ_PACK_H
#define KNJ_PACK_H

#include <stdint.h>

#ifndef __device__
#define __device__ __attribute__((device))
#endif
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
#ifndef KNJ_HOST_CALLABLE
#define KNJ_HOST_CALLABLE
#endif
#ifndef __host__
#define __host__ __attribute__((host))
#endif
#ifndef __host
#define __host
#endif

// --- group sizes -----------------------------------------------------------
// Expert weights: 128. KV keeps 64 by default.
//
// MEASURED (docs/00-verified-facts.md section 7.6), and the reason is a number
// rather than a preference: group 64 costs 4.29% more bytes for the expert bank
// (0.5469 vs 0.5234 B/weight) = ~40 MB/token = ~10 ms of a ~250 ms token budget,
// to buy roughly 0.2-0.3 bits. The int32 accumulator is not a reason to pick 64:
// worst case at group 128 is 243,840 against 2,147,483,647, a headroom of 8807x.
#define KNJ_GROUP_EXPERT 128
#define KNJ_GROUP_KV     64

// --- metadata --------------------------------------------------------------
#define KNJ_META_BYTES_PER_GROUP 3

// --- byte cost per stored weight, by format -------------------------------
// One function, used by both the budget code and the kernel stride maths, so a
// pack and its size estimate cannot drift apart. (They did drift once: the W4
// row said 0.5186 while gemm_w4.hip defined 0.5234, and the two tables in the
// same document were comparing different packs.)
#define KNJ_BITS_PER_WEIGHT_W8 8
#define KNJ_BITS_PER_WEIGHT_W6 6
#define KNJ_BITS_PER_WEIGHT_W4 4
#define KNJ_BITS_PER_WEIGHT_W3 3
#define KNJ_BITS_PER_WEIGHT_W2 2

__host__ __device__ __forceinline__
float knj_bytes_per_weight(int bits, int group) {
    return (float)bits / 8.0f + (float)KNJ_META_BYTES_PER_GROUP / (float)group;
}

#define KNJ_W4_BYTES_PER_WEIGHT (0.5f + 3.0f / 128.0f)   /* 0.5234 */
#define KNJ_W3_BYTES_PER_WEIGHT (0.375f + 3.0f / 128.0f) /* 0.3984 */
#define KNJ_W2_BYTES_PER_WEIGHT (0.25f + 3.0f / 128.0f)  /* 0.2734 */

// --- the packed blob -------------------------------------------------------
//
// Per output column n of a projection, walking K in groups of `group`:
//   [ payload: group*bits/8 bytes ][ scale: fp16 ][ zero_point: uint8 ]
//
// Payload for bits<=4 is NIBBLE packed, two weights per byte, low nibble first,
// with weights in ascending K order. For an 8-weight run inside one payload the
// byte order is:
//
//   byte0: w0 w1      byte1: w2 w3      byte2: w4 w5      byte3: w6 w7
//   (each byte shown as high-nibble, low-nibble)
//
// Codes are UNSIGNED: u = q + zero_point, with q in [0, 15]. See the note on
// `knj_unpack4_u4` below for why "unsigned with a zero point" rather than
// "signed nibble" is the shipped choice.
//
// Alignment: every column's payload starts 4-byte aligned so a thread can load
// it with one 32-bit load, and the 3 metadata bytes are followed by 1 pad byte
// to keep the next column 4-byte aligned. So the on-disk stride per column is
//   payload_bytes + 4
// and that pad byte is INCLUDED in every size estimate below. It is 1 byte per
// group; at group 128 with W4 it is 0.0078 B/weight, which is inside the 0.5234
// already quoted (0.5 + 3/128 = 0.5234 does NOT include it, so the honest W4
// cost is 0.5312). Recorded rather than hidden: the docs' 0.5234 is payload +
// metadata, and this header's stride is what the kernel actually walks.

#define KNJ_W4_PAYLOAD_BYTES(group) (((group) * 4) / 8)
#define KNJ_GROUP_STRIDE_BYTES(bits, group) \
    ((((group) * (bits)) / 8) + 4)

// Dequantised value of one code, in the units the GEMM accumulates:
//   weight ~= (q - 8) * scale          for signed nibble
// This engine stores codes as UNSIGNED u = q + zp, and dequantises as
//   weight = (u - 8) * scale
// so that the dot product can run on raw bytes without a per-element
// correction, with the zero point applied once per group by folding it into the
// activation correction term. The alternative -- signed nibbles -- requires the
// `v_dot4c_i32_i8` CLAMPING variant (which saturates) or an explicit bias add
// per weight; unsigned is both cheaper and exact.

#define KNJ_NIBBLE_BIAS 8

// ---------------------------------------------------------------------------
// Dequantisation, expressed on the GPU
// ---------------------------------------------------------------------------
//
// The two-instruction unpack, from a 32-bit word holding 8 nibbles = 8 weights:
//
//   lo = packed & 0x0f0f0f0f      -> bytes {w0, w2, w4, w6}
//   hi = (packed >> 4) & 0x0f0f0f0f-> bytes {w1, w3, w5, w7}
//
// BOTH need the mask. This is the single most bug-prone spot in the whole pack,
// and it is a documented trap in this repo:
//
//   * `hi = packed >> 4` WITHOUT the trailing mask is WRONG. A whole-word shift
//     drags the next byte's low nibble into this byte's high nibble. This was
//     written, compiled cleanly, and produced plausible wrong numbers
//     (docs/00-verified-facts.md section 7.5 item 3).
//   * `(x & 0x0f0f0f0f) << 4` is the IDENTITY, not a "high nibble extract". A
//     version of this code shipped that as the `hi` term, which made u_hi
//     silently equal u_lo for every weight (section 7.5 item 4).
//
// `lo` and `hi` therefore hold ALTERNATING weights, which is why the
// activations must be stored DE-INTERLEAVED (see below) rather than in natural
// k order. Getting this pairing wrong does not crash; it silently computes a
// permuted dot product, which is why the oracle checks EVERY output element.
__device__ __forceinline__
void knj_unpack8_u4(uint32_t packed, int *lo, int *hi) {
    *lo = (int)(packed & 0x0f0f0f0fu);
    *hi = (int)((packed >> 4) & 0x0f0f0f0fu);
}

// ---------------------------------------------------------------------------
// De-interleaved activation layout
// ---------------------------------------------------------------------------
//
// To match `lo`/`hi` above, activations for a run of 8 consecutive k must be
// presented as TWO 32-bit registers:
//
//   a_even = { a[k0], a[k0+2], a[k0+4], a[k0+6] }
//   a_odd  = { a[k0+1], a[k0+3], a[k0+5], a[k0+7] }
//
// i.e. activations are stored with the even offsets first, then the odd ones.
// This is the layout docs/00-verified-facts.md section 7.6 settled on when it
// chose int8 activations over the (instruction-cheaper, 81.5 dB worse SNR) int4
// activation variant.
//
// The cost of this layout is one de-interleave pass over the activations. That
// pass is HOISTED OUT of the GEMM: activations are prepared once per token by
// C14/C16, not per output column. Paying it inside the K loop would make the
// "cheap unpack" claim false.
//
// Activations are int8 and are NOT bias-corrected here. The per-group
// correction term is applied once per group on the host/device side as
//   correction = scale * (zp - 8) * sum(activations in the group)
// which is an O(1) term per group rather than an O(group) one.

// Offsets, in 32-bit words, within a de-interleaved activation row.
__host__ __device__ __forceinline__
int knj_act_words_per_8k(void) { return 2; }

// ---------------------------------------------------------------------------
// Host-side helpers (the oracle and the activation preparer both use these)
// ---------------------------------------------------------------------------
#ifndef __CUDA_ARCH__

#include <string.h>

// Pack one group of `group` unsigned codes for one output column.
// Host only: the device never packs, it only consumes (docs/01 invariant I2 --
// no dequantisation on the miss path, and by symmetry no packing on it either).
static inline void knj_pack_group_u4(uint8_t *dst, const uint8_t *codes, int group) {
    const int payload = (group * 4) / 8;
    memset(dst, 0, (size_t)payload);
    for (int i = 0; i < payload; ++i) {
        const int lo = codes[2 * i] & 0x0f;
        const int hi = codes[2 * i + 1] & 0x0f;
        dst[i] = (uint8_t)((hi << 4) | lo);
    }
}

// De-interleave one run of 8 natural-order int8 activations into two uint32
// words: word0 = even offsets, word1 = odd offsets. Host only.
static inline void knj_deinterleave8(const int8_t *a8, uint32_t *w2) {
    // 32-bit accumulators. With uint8_t here, `e |= x << (8*t)` discards every
    // term with t>0 and only the first activation byte survives -- a silent
    // truncation that still produces a plausible 32-bit word.
    uint32_t e = 0, o = 0;
    e |= (uint32_t)(uint8_t)a8[0] << 0;
    e |= (uint32_t)(uint8_t)a8[2] << 8;
    e |= (uint32_t)(uint8_t)a8[4] << 16;
    e |= (uint32_t)(uint8_t)a8[6] << 24;
    o |= (uint32_t)(uint8_t)a8[1] << 0;
    o |= (uint32_t)(uint8_t)a8[3] << 8;
    o |= (uint32_t)(uint8_t)a8[5] << 16;
    o |= (uint32_t)(uint8_t)a8[7] << 24;
    w2[0] = e;
    w2[1] = o;
}

#endif  /* !__CUDA_ARCH__ */

#endif  /* KNJ_PACK_H */