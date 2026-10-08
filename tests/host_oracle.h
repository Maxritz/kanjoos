// tests/host_oracle.h -- the host reference. Portable C++17, no HIP, no GPU.
//
// ---------------------------------------------------------------------------
// WHY THE ORACLE IS DELIBERATELY "DUMB"
// ---------------------------------------------------------------------------
// docs/03-kernels.md section 3.2 requires one reference implementation in
// portable C++ that every device kernel is validated against. The obvious way
// to write this oracle is to mirror the kernel's data path: unpack the nibbles
// the same way, call the same dot, apply the same de-interleave. That is a
// mistake, and an expensive one.
//
// If the oracle mirrors the kernel, then a bug in the SHARED assumption -- the
// nibble order, the even/odd pairing, the bias, the group stride -- cancels
// exactly and the comparison passes while both sides are wrong. Every silent
// defect listed in docs/00-verified-facts.md section 7.5 is of that shape.
//
// So this oracle is written as DEQUANTISE-THEN-MULTIPLY, in the most obvious
// way possible, with no packing structure, no dot product, and no de-interleave:
//
//     C[m][n] = sum_g  scale_g * sum_{k in g} (u_k - 8) * a_k
//
// It walks the payload byte by byte in natural k order. It shares exactly one
// thing with the kernel: the meaning of the bytes, which is the thing under test.
// If the kernel's unpack, its de-interleave, or its bias is wrong, this does not
// reproduce it.

#ifndef KNJ_HOST_ORACLE_H
#define KNJ_HOST_ORACLE_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

namespace knj_ref {

// --- fp16 -> fp32 ---------------------------------------------------------
// Independent of the device implementation on purpose.
inline float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (uint32_t)(h >> 10) & 0x1fu;
    const uint32_t man  = (uint32_t)(h & 0x3ffu);
    uint32_t bits;
    if (exp == 0u) {
        if (man == 0u) {
            bits = sign;
        } else {
            uint32_t e = 0u, m = man;
            while ((m & 0x400u) == 0u) { m <<= 1; ++e; }
            m &= 0x3ffu;
            bits = sign | ((113u - e) << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// --- layout constants, mirroring kernels/knj_pack.h ----------------------
// These are the ONLY things shared with the device path, and they are shared by
// value rather than by code, so that a change to the header cannot silently
// change the oracle.
static const int      GROUP      = 128;
static const int      PAYLOAD    = 64;   // 128 nibbles
static const int      STRIDE     = 68;   // payload + scale(2) + zp(1) + pad(1)
static const int      NIB_BIAS   = 8;

// --- the reference --------------------------------------------------------
//
// C is [M][N] float, row-major. Activations are in NATURAL k order (the device
// path uses a de-interleaved form; this one does not, and that difference is the
// point).
inline void gemm_w4_reference(const uint8_t *Bq, const int8_t *Act,
                              float *C, int M, int N, int K) {
    const int groups = K / GROUP;
    for (int m = 0; m < M; ++m) {
        const int8_t *a = Act + (size_t)m * K;
        for (int n = 0; n < N; ++n) {
            const uint8_t *col = Bq + (size_t)n * groups * STRIDE;
            double acc = 0.0;
            for (int g = 0; g < groups; ++g) {
                const uint8_t *grp = col + (size_t)g * STRIDE;
                const float scale = h2f((uint16_t)(grp[64] | (uint16_t)(grp[65] << 8)));
                const int kbase = g * GROUP;
                // Walk the payload in natural k order, one nibble at a time.
                double inner = 0.0;
                for (int j = 0; j < PAYLOAD; ++j) {
                    const uint8_t byte = grp[j];
                    const int lo = byte & 0x0f;
                    const int hi = (byte >> 4) & 0x0f;
                    inner += (double)(lo - NIB_BIAS) * (double)a[kbase + 2 * j];
                    inner += (double)(hi - NIB_BIAS) * (double)a[kbase + 2 * j + 1];
                }
                acc += (double)scale * inner;
            }
            C[(size_t)m * N + n] = (float)acc;
        }
    }
}

// --- the shared pre-processing, host side --------------------------------
//
// De-interleave natural-order int8 activations into the [M][K/8][2] uint32
// form the device expects. Kept here rather than in the driver so the driver
// contains no layout logic of its own.
inline void deinterleave_activations(const int8_t *Act, uint32_t *Out, int M, int K) {
    const int words_per_row = (K / 8) * 2;
    for (int m = 0; m < M; ++m) {
        const int8_t *a = Act + (size_t)m * K;
        uint32_t *o = Out + (size_t)m * words_per_row;
        for (int j = 0; j < K / 8; ++j) {
            // The accumulator MUST be 32-bit. This was originally uint8_t, and
            // `e |= (uint8_t)x << (8*t)` silently discards every term with t>0,
            // leaving only the first activation byte in each word. It compiled
            // cleanly, produced plausible-looking 32-bit words, and made the
            // device-side group sums come back as large negative garbage -- the
            // whole class of defect docs/00-verified-facts.md section 7.5 lists.
            uint32_t e = 0, oo = 0;
            for (int t = 0; t < 4; ++t) {
                e  |= (uint32_t)(uint8_t)a[j * 8 + 2 * t]     << (8 * t);
                oo |= (uint32_t)(uint8_t)a[j * 8 + 2 * t + 1] << (8 * t);
            }
            o[j * 2 + 0] = e;
            o[j * 2 + 1] = oo;
        }
    }
}

// --- the error metric ----------------------------------------------------
//
// Reported the way docs/00-verified-facts.md section 8.0 reports it: relative
// RMSE over EVERY output, max absolute error, the fraction of the output RMS
// that max error represents, and the count of exactly-zero outputs.
//
// The zero count matters more than it looks. A strided element sample, a passing
// exit code and a plausible number are all consistent with total nonsense
// (section 8.6 lists four defects that each produced a plausible number). The
// count of unwritten outputs is what catches "every output stayed zero".
struct ErrorStats {
    double rel_rmse;
    double max_abs;
    double out_rms;
    int    max_idx;
    int    nonzero;
    int    total;
};

inline ErrorStats compare(const float *got, const float *want, int M, int N) {
    ErrorStats s;
    s.total = M * N;
    s.max_abs = 0.0;
    s.max_idx = -1;
    s.nonzero = 0;
    double se = 0.0, sw = 0.0;
    for (int i = 0; i < s.total; ++i) {
        if (want[i] != 0.0f) {
            se += (double)(got[i] - want[i]) * (double)(got[i] - want[i]);
            sw += (double)want[i] * (double)want[i];
        }
        const double d = fabs((double)got[i] - (double)want[i]);
        if (d > s.max_abs) { s.max_abs = d; s.max_idx = i; }
        if (got[i] != 0.0f) { ++s.nonzero; }
    }
    s.rel_rmse = (sw > 0.0) ? sqrt(se / sw) : 0.0;
    s.out_rms  = (sw > 0.0) ? sqrt(sw / (double)s.total) : 0.0;
    return s;
}

}  // namespace knj_ref

#endif  /* KNJ_HOST_ORACLE_H */