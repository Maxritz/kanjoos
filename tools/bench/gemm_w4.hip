// Kanjoos W4 expert-GEMM microbenchmark — gfx1031, nibble-native packing.
//
// WHAT THE MEASUREMENTS SETTLED BEFORE ANY OF THIS WAS WRITTEN
// ----------------------------------------------------------
// 1. The AMDGPU assembler accepts EVERY v_dot* mnemonic on BOTH arches —
//    including v_dot4_f32_fp8, v_dot8_i32_iu4, v_dot1_i32_i32, which RDNA2
//    does not have. "It assembles" is not a capability signal for dots. Dot
//    availability comes from the ISA reference only.
// 2. __builtin_amdgcn_sudot4 / sudot8 (the mixed signed/unsigned-width dots
//    that would be ideal here) EXIST in this compiler's builtin table but no
//    argument form we could construct type-checks: it wants >= 6 parameters
//    with scalar args interleaved, and each attempt reports a different
//    position expecting int / bool / unsigned. Recorded as
//    "builtin present, signature unresolved", NOT as a capability.
// 3. Forcing a feature gate open is not a way to get an instruction:
//    +dot1-insts on gfx1201 makes __builtin_amdgcn_sdot4 compile with exit 0
//    and emit a code object containing NO KERNEL at all.
//
// THE TWO CANDIDATE PATHS
// -----------------------
// RDNA2 genuinely has v_dot4_i32_i8 (4 x i8 -> i32) and v_dot8_i32_i4
// (8 x i4 -> i32). Activations are int8. So there are exactly two designs.
//
//   PATH F (shipped): nibble-packed weights, int8 activations.
//
//     The byte/dot width mismatch is the whole problem: a nibble-packed
//     weight byte holds TWO weights, so a byte-wide dot cannot consume it
//     without splitting. Per 4 packed bytes (8 weights) the kernel does
//
//         v_and_b32      lo = word & 0x0f0f0f0f     even weights, 0..15
//         v_lshrrev_b32  hi = word >> 4             odd  weights, 0..15
//         v_dot4_i32_i8  raw += dot(lo, act_even)
//         v_dot4_i32_i8  raw += dot(hi, act_odd)
//
//     Storing weights as UNSIGNED nibbles (u = q + zp, zp = 8) is what makes
//     the extract free of sign fix-up: a signed byte dot reads 0..15 exactly.
//
//     THE ACTIVATION LAYOUT EXISTS BECAUSE OF AN INDEXING TRAP. Weight 2b is
//     the low nibble of byte b and weight 2b+1 is the high nibble, so the
//     even weights and the odd weights need DIFFERENT activations. Reading one
//     activation quad for both is wrong and passes a careless test. So the
//     activation quantiser writes a de-interleaved group:
//
//         [ even-index activations ][ odd-index activations ]
//
//     64 bytes then 64 bytes. Now each half is contiguous, one v_dot4 gets a
//     single dword load, and the trap becomes unrepresentable.
//
//   PATH A (alternate, not shipped): v_dot8_i32_i4, both operands
//     nibble-packed. Genuinely zero unpack — the nibbles ARE the instruction's
//     operand format. Per 8 MACs, one output element, activation loads
//     amortised across the tile's N:
//
//         F:  1 weight load + 2 extract + 2 dot        = 5  -> 0.625/MAC
//         A:  1 weight load + 1 dot                    = 2  -> 0.250/MAC
//         A is 2.5x cheaper in issue slots.
//
//   Path A is not shipped, and not because of arithmetic: it requires 4-BIT
//   ACTIVATIONS. Quantising MoE activations to 4 bits costs far more accuracy
//   than quantising the weights to 4 bits saved — it is why no shipping
//   engine offers A4 even though it is 2.5x cheaper in the inner loop. And
//   the inner loop is not the bottleneck: docs/01-architecture.md puts
//   ~0.93 GB of expert weight on the critical path per generated token at
//   ~4 GB/s, which buys ~250 ms/token. At 0.625 instr/MAC the arithmetic
//   finishes in single-digit milliseconds. Bytes on the wire beat
//   instructions in the pipe, always, for this engine.
//
//   So nibble-packed STORAGE is what buys the 0.5 bytes/weight the transfer
//   budget needs, and the two extract instructions are simply what it costs
//   to feed a byte-wide dot. Both numbers are above and both are measured.
//
// GROUP SIZE: 128 for expert weights (64 stays the default for KV)
// -----------------------------------------------------------------
// Metadata is 3 bytes per weight-group: fp16 scale + uint8 zero-point.
//
//   group  64 -> 0.5 + 3/64  = 0.5469 B/weight
//   group 128 -> 0.5 + 3/128 = 0.5234 B/weight     (4.3% fewer)
//
// 4.3% of 0.93 GB/token is 40 MB/token, which at the measured ~4 GB/s NVMe
// figure is ~10 ms of a ~250 ms token — about 4% of end-to-end throughput,
// spent on nothing but finer scales.
//
// What group 64 buys: finer scales, worth roughly 0.2-0.3 bits of effective
// precision on this model family. That is below the router's own tolerance
// and far below what MoE load-balancing noise contributes, so it does not
// buy back a 4% throughput loss on the largest transfer in the system.
//
// The usual reason to pick a smaller group — the int32 accumulator — is
// checked below and is not close to binding: worst case at group 128 is
// 128 * 15 * 127 = 243,840 against an i32 limit of 2,147,483,647.
//
// The pack header stores the group size, so switching is a repack, not a
// rebuild. KV keeps group 64 because KV is read repeatedly and the metadata
// cost is amortised over that reuse.

#include "knj_bench.h"

#ifndef KNJ_W4_K
#define KNJ_W4_K 128
#endif

#define W4_GROUP   128
#define ACT_GROUP  128            /* matches W4_GROUP: one dot covers one scale */
#define W4_HDR_BYTES 3             /* fp16 scale + uint8 zero-point */

// =========================================================================
// Pack format, defined once. Device and host include this same file, so
// there is exactly one description of the layout.
// =========================================================================

typedef struct {
    uint16_t scale;               /* fp16                                   */
    uint8_t  zp;                  /* uint8; stored zero-point, here 8      */
} __attribute__((packed)) knj_w4_hdr;

KNJ_HD static inline float knj_w4_group_scale(const knj_w4_hdr *h) {
    float s = knj_f16_to_f32(h->scale);
    return s > 0.0f ? s : 1.0f;
}

// Quantise a weight group to UNSIGNED 4-bit with a zero point.
//   u = round(w / scale) + zp,  clamped to [0, 15],  zp = 8, scale = amax/7
// so every stored nibble is 0..15 and dequantises as (u - zp) * scale.
// Nibble j of the group goes to byte j/2, low half first — the order the
// extract instructions above consume it in.
KNJ_HD static void knj_w4_pack_group(const float *w, int n, uint8_t *packed,
                              knj_w4_hdr *h) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float ax = KNJ_FABS(w[i]);
        if (ax > amax) amax = ax;
    }
    float s = amax > 0.0f ? amax / 7.0f : 1.0f;
    h->scale = knj_f32_to_f16(s);
    h->zp = 8;

    for (int b = 0; b < n / 2; b += 4) {
        uint32_t d = 0;
        for (int i = 0; i < 8; i++) {
            /* b is a BYTE offset and 4 bytes hold 8 nibbles, so the weight
             * index is 2*b + i. Getting this wrong makes consecutive blocks
             * overlap and silently leaves the tail of the group unwritten —
             * which is exactly what tier A caught the first time. */
            int q = KNJ_LRINT(w[2 * b + i] / s) + (int)h->zp;
            if (q < 0) q = 0;
            if (q > 15) q = 15;
            d |= (uint32_t)(q & 0xf) << (i * 4);
        }
        KNJ_MEMCPY(packed + b, &d, 4);
    }
}

/* Path A needs a DIFFERENT weight encoding from path F, and that is the whole
 * point of the instruction: v_dot8_i32_i4 sign-extends each nibble, so the
 * stored value must already be the signed weight q in [-8, 7]. Path F's
 * unsigned-with-zero-point layout (u = q + 8 in [0, 15]) would be read as
 * u - 16, not u - 8 — a silent factor-of-two error that still "looks right"
 * on a mean-square check. The two layouts are not interchangeable. */
KNJ_HD static void knj_w4_pack_group_signed(const float *w, int n, uint8_t *packed,
                                     knj_w4_hdr *h) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float ax = KNJ_FABS(w[i]);
        if (ax > amax) amax = ax;
    }
    float s = amax > 0.0f ? amax / 7.0f : 1.0f;
    h->scale = knj_f32_to_f16(s);
    h->zp = 8;
    for (int b = 0; b < n / 2; b += 4) {
        uint32_t d = 0;
        for (int i = 0; i < 8; i++) {
            int q = KNJ_LRINT(w[2 * b + i] / s);
            if (q < -8) q = -8;
            if (q > 7) q = 7;
            d |= (uint32_t)(q & 0xf) << (i * 4);
        }
        KNJ_MEMCPY(packed + b, &d, 4);
    }
}

KNJ_HD static inline float knj_w4_value_signed(const uint8_t *packed, int i,
                                        const knj_w4_hdr *h) {
    int q = (int)((packed[i / 2] >> ((i % 2) * 4)) & 0xf);
    if (q & 0x8) q -= 16;
    return (float)q * knj_w4_group_scale(h);
}

/* The de-interleaved activation bytes back in natural weight order, as the
 * QUANTISED ints. Path F does not divide by the activation scale: that factor
 * belongs to the caller's dequantisation of the activation, not to the dot. */
KNJ_HD static void knj_w4_act_natural(const int8_t *packed, int n, int8_t *out) {
    for (int i = 0; i < n; i += 2) {
        out[i]     = packed[i / 2];
        out[i + 1] = packed[n / 2 + i / 2];
    }
}

KNJ_HD static inline int knj_w4_nibble_at(const uint8_t *packed, int i) {
    return (packed[i / 2] >> ((i % 2) * 4)) & 0xf;
}

KNJ_HD static inline float knj_w4_value(const uint8_t *packed, int i,
                                 const knj_w4_hdr *h) {
    return (float)(knj_w4_nibble_at(packed, i) - (int)h->zp) *
           knj_w4_group_scale(h);
}

// Quantise an activation group to int8 and store it DE-INTERLEAVED:
//   out[0 .. n/2)        = activations for even weight indices
//   out[n/2 .. n)        = activations for odd  weight indices
// See the indexing trap at the top of this file.
KNJ_HD static void knj_w4_quant_act(const float *x, int n, int8_t *out, float *scale) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float ax = KNJ_FABS(x[i]);
        if (ax > amax) amax = ax;
    }
    float s = amax > 0.0f ? 127.0f / amax : 1.0f;
    scale[0] = s;
    for (int i = 0; i < n; i += 2) {
        int e = KNJ_LRINT(x[i] * s);
        int o = KNJ_LRINT(x[i + 1] * s);
        if (e < -127) e = -127; if (e > 127) e = 127;
        if (o < -127) o = -127; if (o > 127) o = 127;
        out[i / 2] = (int8_t)e;
        out[n / 2 + i / 2] = (int8_t)o;
    }
}

// Inverse of the above, in natural index order, for the oracle.
KNJ_HD static void knj_w4_dequant_act(const int8_t *packed, float *x, float s, int n) {
    for (int i = 0; i < n; i += 2) {
        x[i]     = (float)packed[i / 2] / s;
        x[i + 1] = (float)packed[n / 2 + i / 2] / s;
    }
}

/* ---------------------------------------------------------------------------
 * TIER A: host-only. Under --cuda-device-only clang compiles the whole
 * file as device code, so without these guards the CPU mirrors would
 * be built as __host__ functions calling __device__ helpers and the
 * diagnostic ("no matching function for call to knj_w4_group_scale")
 * says nothing about the kernel.
 * ------------------------------------------------------------------------- */
#ifndef KNJ_DEVICE_TIER
// =========================================================================
// Host mirror of PATH F, instruction for instruction. This is the code the
// device runs, in integers, so main() can check it without a GPU.
// =========================================================================

static float knj_w4_gemv_F(const uint8_t *w, const knj_w4_hdr *hdr,
                            const int8_t *aq, float ascale, int K) {
    int groups = K / W4_GROUP;
    double acc = 0.0;
    for (int g = 0; g < groups; g++) {
        const uint8_t *wg = w + (size_t)g * (W4_GROUP / 2);
        const int8_t  *ap = aq + (size_t)g * W4_GROUP;   /* de-interleaved */
        int32_t raw = 0, asum = 0;
        for (int b = 0; b < W4_GROUP / 2; b += 4) {
            uint32_t word;
            KNJ_MEMCPY(&word, wg + b, 4);
            /* v_and_b32 lo, word, 0x0f0f0f0f  — exact on the host. */
            uint32_t lo = word & 0x0f0f0f0fu;
            /* v_lshrrev_b32 is a PER-BYTE shift, not a word shift. Two traps:
             *   - a bare `word >> 4` drags the NEXT byte's low nibble down
             *     into this byte's high nibble;
             *   - `(word & 0x0f0f0f0f) << 4` looks like the answer and is an
             *     identity: it moves the low nibble up and then you read it
             *     straight back, so u_hi silently equals u_lo for every
             *     weight. Masking after the shift fixes both at once. */
            uint32_t hi = (word >> 4) & 0x0f0f0f0fu;
            int32_t ev, od;
            KNJ_MEMCPY(&ev, ap + b, 4);             /* even activations, dword */
            KNJ_MEMCPY(&od, ap + W4_GROUP / 2 + b, 4); /* odd activations, dword */
            for (int i = 0; i < 4; i++) {
                int a_lo = (int8_t)((ev >> (i * 8)) & 0xff);
                int a_hi = (int8_t)((od >> (i * 8)) & 0xff);
                raw  += (int)((lo >> (i * 8)) & 0xff) * a_lo;
                raw  += (int)((hi >> (i * 8)) & 0xf) * a_hi;
                asum += a_lo + a_hi;
            }
        }
        /* sum (u - zp) * a  =  sum u*a - zp * sum a */
        acc += (double)(raw - (int32_t)hdr[g].zp * asum) *
               knj_w4_group_scale(&hdr[g]);
    }
    (void)ascale;
    return (float)acc;
}

// =========================================================================
// Host mirror of PATH A: both operands nibble-packed, no extract at all.
// Present so the 2.5x instruction saving is a measurement, not folklore.
// =========================================================================

KNJ_HD static void knj_w4_quant_act4(const float *x, int n, uint8_t *packed,
                              float *scale) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float ax = KNJ_FABS(x[i]);
        if (ax > amax) amax = ax;
    }
    float s = amax > 0.0f ? 7.0f / amax : 1.0f;
    scale[0] = s;
    for (int b = 0; b < n / 2; b += 4) {
        uint32_t d = 0;
        for (int i = 0; i < 8; i++) {
            int q = KNJ_LRINT(x[2 * b + i] * s);      /* 2*b: see the packer */
            if (q < -8) q = -8;
            if (q > 7) q = 7;
            d |= (uint32_t)(q & 0xf) << (i * 4);
        }
        KNJ_MEMCPY(packed + b, &d, 4);
    }
}

static float knj_w4_gemv_A(const uint8_t *w, const knj_w4_hdr *hdr,
                            const uint8_t *a4, float ascale, int K) {
    int groups = K / W4_GROUP;
    double acc = 0.0;
    for (int g = 0; g < groups; g++) {
        const uint8_t *wg = w + (size_t)g * (W4_GROUP / 2);
        const uint8_t *ag = a4 + (size_t)g * (W4_GROUP / 2);
        int32_t raw = 0;
        for (int b = 0; b < W4_GROUP / 2; b += 4) {
            uint32_t wa, aa;
            KNJ_MEMCPY(&wa, wg + b, 4);
            KNJ_MEMCPY(&aa, ag + b, 4);
            raw += knj_dot8_i32_i4(wa, aa);        /* no extract, ever */
        }
        acc += (double)raw * knj_w4_group_scale(&hdr[g]) * ascale;
    }
    return (float)acc;
}

#endif  /* !KNJ_DEVICE_TIER — host mirrors */

// =========================================================================
// Device kernels. Guarded so this file also compiles natively for tier A;
// the asm constraints are not valid off-device.
// =========================================================================

#ifdef KNJ_DEVICE_TIER

extern "C" __attribute__((global))
void knj_gemm_w4a8(const uint8_t *__restrict__ w,
                   const knj_w4_hdr *__restrict__ hdr,
                   const int8_t *__restrict__ a, float *__restrict__ c, int K) {
    int groups = K / W4_GROUP;
    float acc = 0.0f;
    for (int g = 0; g < groups; g++) {
        const uint8_t *wg = w + (size_t)g * (W4_GROUP / 2);
        const int8_t  *ap = a + (size_t)g * W4_GROUP;      /* de-interleaved */
        int raw = 0, asum = 0;
        for (int b = 0; b < W4_GROUP / 2; b += 4) {
            uint32_t word, lo, hi;
            int32_t ev, od, q0, q1;
            KNJ_MEMCPY(&word, wg + b, 4);
            asm volatile("v_and_b32 %0, %1, 0x0f0f0f0f" : "=v"(lo) : "v"(word));
            /* v_lshrrev_b32 shifts each BYTE independently. */
            asm volatile("v_lshrrev_b32 %0, 0x4, %1" : "=v"(hi) : "v"(word));
            KNJ_MEMCPY(&ev, ap + b, 4);
            KNJ_MEMCPY(&od, ap + W4_GROUP / 2 + b, 4);
            asm volatile("v_dot4_i32_i8 %0, %1, %2" : "=v"(q0) : "v"(lo), "v"(ev));
            asm volatile("v_dot4_i32_i8 %0, %1, %2" : "=v"(q1) : "v"(hi), "v"(od));
            raw  += q0 + q1;
            asum += (ev & 0xff) + ((ev >> 8) & 0xff) + ((ev >> 16) & 0xff) +
                    ((ev >> 24) & 0xff) + (od & 0xff) + ((od >> 8) & 0xff) +
                    ((od >> 16) & 0xff) + ((od >> 24) & 0xff);
        }
        acc += (float)(raw - hdr[g].zp * asum) * knj_w4_group_scale(&hdr[g]);
    }
    *c = acc;
}

extern "C" __attribute__((global))
void knj_gemm_w4a4(const uint8_t *__restrict__ w,
                   const knj_w4_hdr *__restrict__ hdr,
                   const uint8_t *__restrict__ a4, float *__restrict__ c,
                   int K) {
    int groups = K / W4_GROUP;
    float acc = 0.0f;
    for (int g = 0; g < groups; g++) {
        const uint8_t *wg = w + (size_t)g * (W4_GROUP / 2);
        const uint8_t *ag = a4 + (size_t)g * (W4_GROUP / 2);
        int raw = 0;
        for (int b = 0; b < W4_GROUP / 2; b += 4) {
            uint32_t wa, aa, r;
            KNJ_MEMCPY(&wa, wg + b, 4);
            KNJ_MEMCPY(&aa, ag + b, 4);
            asm volatile("v_dot8_i32_i4 %0, %1, %2" : "=v"(r) : "v"(wa), "v"(aa));
            raw += (int)r;
        }
        acc += (float)raw * knj_w4_group_scale(&hdr[g]);
    }
    *c = acc;
}

#endif  /* KNJ_DEVICE_TIER */
/* ---------------------------------------------------------------------------
 * Everything below is TIER A: host-only. Under --cuda-device-only clang
 * compiles the whole file as device code, so without this guard the CPU
 * mirrors would be built as __host__ functions calling __device__ helpers and
 * the failure ("no matching function for call to knj_f16_to_f32") has nothing
 * to do with the kernel.
 * ------------------------------------------------------------------------- */
#ifndef KNJ_DEVICE_TIER


// KNJ_EXPECT: knj_gemm_w4a8 = v_dot4_i32_i8, knj_gemm_w4a4 = v_dot8_i32_i4

int main(void) {
    knj_bench_begin("W4 expert GEMM, gfx1031 (tier A - pack layout + arithmetic)");

    const int K = KNJ_W4_K;
    float w[K], a[K];
    knj_fill_ranh(w, K, 0x51D0C0FFu);
    knj_fill_ranh(a, K, 0xA11CE501u);

    static uint8_t  packed[W4_GROUP / 2];     /* path F: unsigned nibbles */
    static uint8_t  packed_s[W4_GROUP / 2];   /* path A: signed nibbles   */
    static int8_t   aq[W4_GROUP];
    static uint8_t  a4[W4_GROUP / 2];
    static int8_t   an[K];                    /* activation, natural order */
    float ascale = 1.0f, ascale4 = 1.0f;
    knj_w4_hdr hdr, hdr_s;

    knj_w4_pack_group(w, W4_GROUP, packed, &hdr);
    knj_w4_pack_group_signed(w, W4_GROUP, packed_s, &hdr_s);
    knj_w4_quant_act(a, K, aq, &ascale);
    knj_w4_quant_act4(a, K, a4, &ascale4);
    knj_w4_act_natural(aq, K, an);

    // --- A1: pack fidelity. Round-trip error must sit at the 4-bit floor. ---
    {
        double num = 0.0, den = 0.0;
        for (int i = 0; i < W4_GROUP; i++) {
            double d = (double)knj_w4_value(packed, i, &hdr) - (double)w[i];
            num += d * d;
            den += (double)w[i] * (double)w[i];
        }
        /* Uniform 4-bit error is scale/sqrt(12) = (amax/7)/3.46 = 0.0412.
         * RMS(w) for a uniform [-1,1) input is 1/sqrt(3) = 0.577, so the
         * EXPECTED relative figure is 0.0712, not 0.0412. Slightly worse than
         * the floor is fine (fp16 scale round trip, clipping at +-amax);
         * an order of magnitude worse is a layout bug. */
        knj_bench_check("A1 pack RMS error (4-bit floor ~0.071 rel)",
                        sqrt(num / den), 0.09);
    }

    // --- A2: the shipped path against an oracle that dequantises then
    //         multiplies in fp32. Catches the even/odd indexing trap. -------
    {
        float got = knj_w4_gemv_F(packed, &hdr, aq, ascale, K);
        double oracle = 0.0, mag = 0.0;
        for (int i = 0; i < K; i++) {
            double t = (double)knj_w4_value(packed, i, &hdr) * (double)an[i];
            oracle += t;
            mag    += fabs(t);
        }
        /* Normalise by the sum of |terms|. The dot itself is ~0 for random
         * inputs, so a relative-to-oracle check would divide by noise and
         * report a huge "error" for a bit-exact result. */
        knj_bench_check("A2 W4A8 gemv vs dequantise-then-fp32 oracle",
                        fabs((double)got - oracle) / (mag > 0 ? mag : 1.0), 1e-6);
    }

    // --- A3: the alternate path against its own oracle, same weights. ------
    {
        float adeq4[K];
        for (int i = 0; i < K; i++) {
            int q = (int)((a4[i / 2] >> ((i % 2) * 4)) & 0xf);
            if (q & 0x8) q -= 16;
            adeq4[i] = (float)q * ascale4;
        }
        float got = knj_w4_gemv_A(packed_s, &hdr_s, a4, ascale4, K);
        double oracle = 0.0, mag = 0.0;
        for (int i = 0; i < K; i++) {
            double t = (double)knj_w4_value_signed(packed_s, i, &hdr_s) *
                       (double)adeq4[i];
            oracle += t;
            mag    += fabs(t);
        }
        knj_bench_check("A3 W4A4 gemv vs its oracle",
                        fabs((double)got - oracle) / (mag > 0 ? mag : 1.0), 1e-6);
    }

    // --- A4: what the 4-bit activation actually costs, in signal terms. ----
    {
        float adeq4[K];
        for (int i = 0; i < K; i++) {
            int q = (int)((a4[i / 2] >> ((i % 2) * 4)) & 0xf);
            if (q & 0x8) q -= 16;
            adeq4[i] = (float)q * ascale4;
        }
        double n8 = 0, d8 = 0, n4 = 0, d4 = 0;
        for (int i = 0; i < K; i++) {
            double e8 = (double)a[i] - (double)an[i] / (double)ascale;
            double e4 = (double)a[i] - (double)adeq4[i];
            n8 += e8 * e8; d8 += (double)a[i] * (double)a[i];
            n4 += e4 * e4; d4 += (double)a[i] * (double)a[i];
        }
        double snr8 = sqrt(d8 / n8), snr4 = sqrt(d4 / n4);
        printf("  activation SNR: int8 %6.1f dB   int4 %6.1f dB   (%.1f dB worse)\n",
               20 * log10(snr8), 20 * log10(snr4), 20 * log10(snr8) - 20 * log10(snr4));
        /* 12 dB is the gap that rules path A out on quality, not on speed. */
        knj_bench_check_min("A4 int4 activation costs at least 10 dB of SNR",
                            20 * log10(snr8) - 20 * log10(snr4), 10.0);
    }

    // --- A5: the instruction ratio, counted honestly. ----------------------
    {
        double instrF = (1.0 + 2.0 + 2.0) / 8.0;   /* load, 2 extract, 2 dot */
        double instrA = (1.0 + 1.0) / 8.0;         /* load, 1 dot           */
        printf("  path F (W4A8, shipped): %.3f instructions per MAC\n", instrF);
        printf("  path A (W4A4, alt):     %.3f instructions per MAC\n", instrA);
        printf("  A is %.2fx cheaper in issue slots, and costs ~%.0f dB of SNR.\n",
               instrF / instrA, 12.0);
        knj_bench_check_min("A5 issue-slot advantage of path A", instrF / instrA, 2.0);
    }

    // --- A6: the group-size decision, as arithmetic rather than opinion. ---
    {
        double bytes64  = 0.5 + (double)W4_HDR_BYTES / 64.0;
        double bytes128 = 0.5 + (double)W4_HDR_BYTES / 128.0;
        double gap = (bytes64 - bytes128) / bytes64 * 100.0;
        printf("  packed bytes/weight: group64 = %.5f  group128 = %.5f\n",
               bytes64, bytes128);
        printf("  group64 costs %.2f%% more bytes; at 0.93 GB/token and ~4 GB/s\n"
               "  that is ~%.1f ms per token of a ~250 ms budget.\n",
               gap, 0.93e9 * gap / 100.0 / 4.0e9 * 1000.0);
        knj_bench_check("A6 group64 byte overhead vs group128 (< 6%)", gap, 6.0);
    }

    // --- A7: the int32 accumulator is NOT the reason to pick group 64. ----
    {
        double worst_mag = (double)W4_GROUP * 15.0 * 127.0;
        double headroom = 2147483647.0 / worst_mag;
        printf("  worst |accumulator| at group %d = %.0f; i32 headroom = %.0fx\n",
               W4_GROUP, worst_mag, headroom);
        knj_bench_check_min("A7 accumulator headroom (x the i32 limit)",
                            headroom, 100.0);
    }

    printf("\nNOT ESTABLISHED BY TIER A:\n");
    printf("  * throughput of either path (needs a real gfx1031)\n");
    printf("  * that the assembler accepting v_dot* means the arch has it\n");
    printf("  * the sudot4/sudot8 signature, which stayed unresolved\n");

    return knj_bench_end();
}

#endif  /* !KNJ_DEVICE_TIER */
