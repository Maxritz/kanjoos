// Kanjoos C16 attention - prefill and decode kernels, tier C.
//
// WHAT THIS IS
// ------------
// Two DISTINCT kernels, per docs/02-components.md C16 and ai-coder/c16:
//   attn_prefill - tiled Q-block x KV-block, online (streaming) softmax.
//                  One workgroup owns QT=8 query rows; K/V stream through
//                  LDS in small tiles (KT=16), so each kv-head's K/V reach DRAM
//                  once and are re-served from the 64 MB Infinity Cache
//                  to the (S/QT) workgroups that share them.
//   attn_decode  - single Q row per (batch, head), full KV stream, no
//                  Q tiling: the decode shape (one token, everything the
//                  tiering work exists for).
// Not one kernel with a parameter: the two loops have different shapes,
// different LDS footprints and different grids, which is the point C16
// makes about "decode is not a special case of prefill".
//
// GQA/MQA: query head hq uses kv head hq/G (G=1 is MHA, G=H is MQA).
// Validation runs G=2 (GQA) and G=4 (MQA) at S=512; timing runs G=4.
//
// THE SCORE ENGINE (the layout lessons this file is built on)
// -----------------------------------------------------------
// A wave computes ONE score per round: each lane owns 4 k elements
// (k = lane*4..+3, dword-aligned, so k-adjacent lanes read ADJACENT LDS
// addresses - no bank conflict). The 32 partials combine with a 5-step
// __shfl_xor butterfly: NO prt[] LDS round trip, NO __syncthreads in the
// score loop (the old design paid 2 barriers/score in prefill and a
// 32-slot LDS scan per lane; the 1481-cycle trace named both as top
// costs). Butterfly pairing is identical for every lane, so all 32
// lanes get the SAME bits back: the per-lane online softmax (m, l) is
// bit-uniform across lanes and there is NO cross-lane reduce at the end
// - each lane normalizes its own 4 output columns. Causal masking runs
// after the reduce and is wave-uniform (rg is per-wave), so nothing in
// divergent control flow ever touched a barrier. Q loads are hoisted
// out of the j-loop (invariant per rstep) and each round's V loads are
// ISSUED FIRST so their LDS latency rides under reduce+exp.
//
// STAGING uses the decoupled load-then-store batch pattern measured in
// gemm_tiled.hip (load->store in one loop serializes a wait_loadcnt per
// element; batches of 16 dwords pay the latency once per batch).
//
// BUILD (the only line that links on this machine), guard via ENV:
//   hipcc -nogpulib -O2 --offload-arch=gfx1201 \
//         -LG:/ROCM10RT-gfx1201/lib -lamdhip64 attn_c16.hip -o at.exe
//   KNJ_BUILD_ARCH=gfx1201 ./at.exe
//
// Constraints: no gridDim (lowers through the device runtime we do not
// link), no device libm - __expf is the native v_exp_f32 lowering
// (docs/00 section 9, measured by the I7 harness), fp16 in / fp32
// accumulate / fp16 out.
//
// EXIT CODES: 0 pass, 1 correctness/HIP failure, 6 arch mismatch (skip),
// 7 KNJ_BUILD_ARCH unset (refused).

#include <hip/hip_runtime.h>
#include <chrono>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HD 128     /* head dim                                  */
#define KT 16      /* prefill KV tile rows: 8 KB KL+VL + 2 KB QL
                     * ~ 10 KB, so 6 workgroups/CU - residency is what
                     * hides the serial score chain. Measured sweep:
                     * 64=40.5 ms, 32=15.3, 16=9.5, 8=10.3 (staging
                     * cadence loses at 8).                          */
#define KTD 32     /* decode KV tile rows: decode is 1 wave/block so
                     * bigger tiles win on staging cadence (16 was
                     * worse than 32: 208.7 vs 194.3 us/step).      */
#define QT 8       /* query rows per prefill workgroup          */
#define BLKP 64    /* prefill threads (2 waves -> 2 q rows)     */
#define BLKD 32    /* decode threads (1 wave)                   */

#define F2H(f) (__builtin_bit_cast(uint16_t, (_Float16)(f)))
#define H2F(h) ((float)__builtin_bit_cast(_Float16, (uint16_t)(h)))

static double now_us(void) {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch())
        .count();
}

#define HIPCHECK(x, what)                                                      \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            printf("%s FAILED: %s\n", (what), hipGetErrorString(e_));          \
            printf("RESULT: FAIL\n");                                          \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// Decoupled staging: ALL 16 loads into registers, then ALL 16 stores.
// Guarded on both sides: m_ >= npt_ must never touch DST (an OOB LDS
// store would silently corrupt the score state).
#define KNJ_STAGE(DST, SRC, NDW, TID, BLK)                                     \
    do {                                                                       \
        const int npt_ = (NDW) / (BLK);                                        \
        for (int b_ = 0; b_ < npt_; b_ += 16) {                                \
            uint32_t buf_[16];                                                 \
            for (int t_ = 0; t_ < 16; t_++) {                                  \
                const int m_ = b_ + t_;                                        \
                const int d_ = (TID) + m_ * (BLK);                             \
                if (m_ < npt_)                                                 \
                    memcpy(&buf_[t_], &(SRC)[(size_t)d_ * 2], 4);              \
                else                                                           \
                    buf_[t_] = 0u;                                             \
            }                                                                  \
            for (int t_ = 0; t_ < 16; t_++) {                                  \
                const int m_ = b_ + t_;                                        \
                const int d_ = (TID) + m_ * (BLK);                             \
                if (m_ < npt_) memcpy(&(DST)[d_ * 2], &buf_[t_], 4);           \
            }                                                                  \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// PREFILL: grid (H, S/QT). One workgroup per (head, q-tile).
// Layouts: Q,O [H][S][HD]; K,V [H/G][S][HD]; fp16 halves.
// ---------------------------------------------------------------------------
__global__ void attn_prefill(const uint16_t *__restrict__ Q,
                             const uint16_t *__restrict__ K,
                             const uint16_t *__restrict__ V,
                             uint16_t *__restrict__ O, int S, int G) {
    __shared__ uint16_t QL[QT * HD];
    __shared__ uint16_t KL[KT * HD];
    __shared__ uint16_t VL[KT * HD];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int w = tid >> 5;
    const int hq = blockIdx.x;
    const int kvh = hq / G;
    const int r0 = blockIdx.y * QT;

    float acc[QT][4];
    float m[QT], l[QT];
    for (int r = 0; r < QT; r++) {
        m[r] = -1e30f;
        l[r] = 0.0f;
        for (int c = 0; c < 4; c++) acc[r][c] = 0.0f;
    }

    /* Q tile staged once per workgroup (rows r0..r0+QT-1) */
    KNJ_STAGE(QL, Q + (size_t)(hq * S + r0) * HD, QT * HD / 2, tid, BLKP);
    __syncthreads();

    for (int j0 = 0; j0 < S; j0 += KT) {
        KNJ_STAGE(KL, K + (size_t)(kvh * S + j0) * HD, KT * HD / 2, tid, BLKP);
        KNJ_STAGE(VL, V + (size_t)(kvh * S + j0) * HD, KT * HD / 2, tid, BLKP);
        __syncthreads();        for (int rstep = 0; rstep < QT; rstep += 2) {
            const int rloc = rstep + w;       /* wave w owns this q row  */
            const int rg = r0 + rloc;         /* global row, for causal  */
            /* Q is invariant over j: load once per rstep, not per j.  */
            const int k0 = lane * 4;
            uint32_t q0, q1;
            memcpy(&q0, &QL[rloc * HD + k0], 4);
            memcpy(&q1, &QL[rloc * HD + k0 + 2], 4);
            /* unroll: stage j+1's K loads under j's reduce+exp chain -
             * the unroll lesson from gemm_tiled.hip says census after.
             * 4 is the sweet spot: 8 bought 0.7% for +12 VGPR. */
#pragma unroll 4
            for (int j = 0; j < KT; j++) {
                /* Issue this round's V loads FIRST: their LDS latency
                 * rides under the reduce + exp chain (decoupled staging
                 * lesson from gemm_tiled.hip). */
                uint32_t v0, v1;
                memcpy(&v0, &VL[j * HD + lane * 4], 4);
                memcpy(&v1, &VL[j * HD + lane * 4 + 2], 4);
                uint32_t z0, z1;
                memcpy(&z0, &KL[j * HD + k0], 4);
                memcpy(&z1, &KL[j * HD + k0 + 2], 4);
                float part =
                    H2F((uint16_t)(q0 & 0xffffu)) *
                        H2F((uint16_t)(z0 & 0xffffu)) +
                    H2F((uint16_t)(q0 >> 16)) * H2F((uint16_t)(z0 >> 16)) +
                    H2F((uint16_t)(q1 & 0xffffu)) *
                        H2F((uint16_t)(z1 & 0xffffu)) +
                    H2F((uint16_t)(q1 >> 16)) * H2F((uint16_t)(z1 >> 16));
                /* 5-step butterfly: replaces prt[] + 2 __syncthreads +
                 * the 32-slot LDS scan. Identical tree pairing for every
                 * lane => identical bits => wave-uniform m/l. */
                for (int o = 1; o < 32; o <<= 1)
                    part += __shfl_xor(part, o);
                float s = part;
                /* causal: wave-uniform (rg is per-wave), no barrier */
                if (j0 + j > rg) s = -1e30f;

                const float mm = s > m[rloc] ? s : m[rloc];
                const float alpha = __expf(m[rloc] - mm);
                const float p = __expf(s - mm);
                l[rloc] = l[rloc] * alpha + p;

                acc[rloc][0] =
                    acc[rloc][0] * alpha + p * H2F((uint16_t)(v0 & 0xffffu));
                acc[rloc][1] =
                    acc[rloc][1] * alpha + p * H2F((uint16_t)(v0 >> 16));
                acc[rloc][2] =
                    acc[rloc][2] * alpha + p * H2F((uint16_t)(v1 & 0xffffu));
                acc[rloc][3] =
                    acc[rloc][3] * alpha + p * H2F((uint16_t)(v1 >> 16));
                m[rloc] = mm;
            }
        }
    }

    for (int rstep = 0; rstep < QT; rstep += 2) {
        const int rloc = rstep + w;
        const float inv = 1.0f / l[rloc];
        uint16_t *orow = O + (size_t)(hq * S + r0 + rloc) * HD;
        const uint32_t o0 = (uint32_t)F2H(acc[rloc][0] * inv) |
                            ((uint32_t)F2H(acc[rloc][1] * inv) << 16);
        const uint32_t o1 = (uint32_t)F2H(acc[rloc][2] * inv) |
                            ((uint32_t)F2H(acc[rloc][3] * inv) << 16);
        memcpy(orow + lane * 4, &o0, 4);
        memcpy(orow + lane * 4 + 2, &o1, 4);
    }
}

// ---------------------------------------------------------------------------
// DECODE: grid (B*H,). One wave, one query row, full KV (equivalent to the
// causal mask at the last position). Layouts: Q,O [B*H][HD];
// K,V [B*(H/G)][S][HD].
// ---------------------------------------------------------------------------
__global__ void attn_decode(const uint16_t *__restrict__ Q,
                            const uint16_t *__restrict__ K,
                            const uint16_t *__restrict__ V,
                            uint16_t *__restrict__ O, int S, int H, int G) {
    __shared__ uint16_t QL[HD];
    __shared__ uint16_t KL[KTD * HD];
    __shared__ uint16_t VL[KTD * HD];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int ob = blockIdx.x;
    const int hq = ob % H;
    const int b = ob / H;
    const int kvh = hq / G;
    const int kbase = (b * (H / G) + kvh) * S;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m = -1e30f, l = 0.0f;

    KNJ_STAGE(QL, Q + (size_t)ob * HD, HD / 2, tid, BLKD);
    __syncthreads();

    const int k0 = lane * 4;
    uint32_t q0, q1;
    memcpy(&q0, &QL[k0], 4);
    memcpy(&q1, &QL[k0 + 2], 4);

    for (int j0 = 0; j0 < S; j0 += KTD) {
        KNJ_STAGE(KL, K + (size_t)(kbase + j0) * HD, KTD * HD / 2, tid, BLKD);
        KNJ_STAGE(VL, V + (size_t)(kbase + j0) * HD, KTD * HD / 2, tid, BLKD);
        __syncthreads();

        for (int j = 0; j < KTD; j++) {
            /* V loads first - latency hides under reduce + exp. */
            uint32_t v0, v1;
            memcpy(&v0, &VL[j * HD + lane * 4], 4);
            memcpy(&v1, &VL[j * HD + lane * 4 + 2], 4);
            uint32_t z0, z1;
            memcpy(&z0, &KL[j * HD + k0], 4);
            memcpy(&z1, &KL[j * HD + k0 + 2], 4);
            float part =
                H2F((uint16_t)(q0 & 0xffffu)) *
                    H2F((uint16_t)(z0 & 0xffffu)) +
                H2F((uint16_t)(q0 >> 16)) * H2F((uint16_t)(z0 >> 16)) +
                H2F((uint16_t)(q1 & 0xffffu)) *
                    H2F((uint16_t)(z1 & 0xffffu)) +
                H2F((uint16_t)(q1 >> 16)) * H2F((uint16_t)(z1 >> 16));
            /* butterfly reduce: no prt[], no barrier (the old single
             * barrier here was also a latent store-before-read race
             * across lanes of the wave - now structurally impossible) */
            for (int o = 1; o < 32; o <<= 1)
                part += __shfl_xor(part, o);
            const float s = part;

            const float mm = s > m ? s : m;
            const float alpha = __expf(m - mm);
            const float p = __expf(s - mm);
            l = l * alpha + p;

            acc[0] = acc[0] * alpha + p * H2F((uint16_t)(v0 & 0xffffu));
            acc[1] = acc[1] * alpha + p * H2F((uint16_t)(v0 >> 16));
            acc[2] = acc[2] * alpha + p * H2F((uint16_t)(v1 & 0xffffu));
            acc[3] = acc[3] * alpha + p * H2F((uint16_t)(v1 >> 16));
            m = mm;
        }
    }

    const float inv = 1.0f / l;
    uint16_t *orow = O + (size_t)ob * HD;
    const uint32_t o0 = (uint32_t)F2H(acc[0] * inv) |
                        ((uint32_t)F2H(acc[1] * inv) << 16);
    const uint32_t o1 = (uint32_t)F2H(acc[2] * inv) |
                        ((uint32_t)F2H(acc[3] * inv) << 16);
    memcpy(orow + lane * 4, &o0, 4);
    memcpy(orow + lane * 4 + 2, &o1, 4);
}

// ---------------------------------------------------------------------------
// VRAM streaming read - the roofline both kernels are judged against.
// ---------------------------------------------------------------------------
__global__ void bw_read(const uint32_t *__restrict__ p, uint32_t n,
                        uint32_t stride, float *__restrict__ out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    uint32_t k = i;
    for (; k + 3u * stride < n; k += 4u * stride) {
        s0 += p[k]; s1 += p[k + stride];
        s2 += p[k + 2u * stride]; s3 += p[k + 3u * stride];
    }
    for (; k < n; k += stride) s0 += p[k];
    if (s0 + s1 + s2 + s3 == 0xffffffffu) out[i] = (float)(s0 + s1 + s2 + s3);
}

// ---------------------------------------------------------------------------
// CPU references, double accumulate. rmax/omax bound the region computed so
// the timing configuration gets a spot oracle without a 10 s full sweep.
// ---------------------------------------------------------------------------
static void ref_prefill(const uint16_t *Q, const uint16_t *K,
                        const uint16_t *V, float *Out, int S, int H, int G,
                        int rmax) {
    double *sc = (double *)malloc((size_t)S * sizeof(double));
    for (int hq = 0; hq < H; hq++) {
        const int kvh = hq / G;
        for (int r = 0; r < rmax; r++) {
            double mx = -1e300;
            for (int j = 0; j <= r; j++) {
                double s = 0.0;
                for (int k = 0; k < HD; k++)
                    s += (double)H2F(Q[((size_t)hq * S + r) * HD + k]) *
                         (double)H2F(K[((size_t)kvh * S + j) * HD + k]);
                sc[j] = s;
                if (s > mx) mx = s;
            }
            double z = 0.0;
            for (int j = 0; j <= r; j++) z += exp(sc[j] - mx);
            for (int c = 0; c < HD; c++) {
                double num = 0.0;
                for (int j = 0; j <= r; j++)
                    num += exp(sc[j] - mx) *
                           (double)H2F(V[((size_t)kvh * S + j) * HD + c]);
                Out[((size_t)hq * S + r) * HD + c] = (float)(num / z);
            }
        }
    }
    free(sc);
}

static void ref_decode(const uint16_t *Q, const uint16_t *K,
                       const uint16_t *V, float *Out, int S, int H, int G,
                       int B, int omax) {
    double *sc = (double *)malloc((size_t)S * sizeof(double));
    for (int ob = 0; ob < omax; ob++) {
        const int hq = ob % H;
        const int b = ob / H;
        const int kvh = hq / G;
        const int kbase = (b * (H / G) + kvh) * S;
        for (int j = 0; j < S; j++) {
            double s = 0.0;
            for (int k = 0; k < HD; k++)
                s += (double)H2F(Q[(size_t)ob * HD + k]) *
                     (double)H2F(K[((size_t)kbase + j) * HD + k]);
            sc[j] = s;
        }
        double mx = -1e300;
        for (int j = 0; j < S; j++)
            if (sc[j] > mx) mx = sc[j];
        double z = 0.0;
        for (int j = 0; j < S; j++) z += exp(sc[j] - mx);
        for (int c = 0; c < HD; c++) {
            double num = 0.0;
            for (int j = 0; j < S; j++)
                num += exp(sc[j] - mx) *
                       (double)H2F(V[((size_t)kbase + j) * HD + c]);
            Out[(size_t)ob * HD + c] = (float)(num / z);
        }
    }
    free(sc);
}

static double rel_rmse(const float *want, const float *got, size_t n) {
    double sse = 0.0, mag = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)got[i] - (double)want[i];
        sse += d * d;
        mag += (double)want[i] * (double)want[i];
    }
    return sqrt(sse / (mag > 0.0 ? mag : 1.0));
}

static void fill_rand(uint16_t *p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = F2H(((float)((i * 1103515245u >> 16) & 0xffff) / 32768.0f) -
                   1.0f);
}

/* Outputs are fp16; refs are fp32. The first draft compared the raw
 * fp16 BITS against the float oracle - every element mismatched. Convert
 * explicitly, always. */
static void f16_to_f32(const uint16_t *src, float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = H2F(src[i]);
}

/* Prefill outputs are laid out [H][S][HD] but the spot oracle fills only
 * r < rmax rows PER HEAD: comparing the first n elements would read the
 * stale gaps between heads. Compare exactly the rows that were computed. */
static double rel_rmse_prefill_rows(const float *want, const float *got,
                                    int S, int H, int rmax) {
    double sse = 0.0, mag = 0.0;
    for (int hq = 0; hq < H; hq++)
        for (int r = 0; r < rmax; r++)
            for (int c = 0; c < HD; c++) {
                const size_t i = ((size_t)hq * S + r) * HD + c;
                const double d = (double)got[i] - (double)want[i];
                sse += d * d;
                mag += (double)want[i] * (double)want[i];
            }
    return sqrt(sse / (mag > 0.0 ? mag : 1.0));
}

// ---------------------------------------------------------------------------
int main(void) {
    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, 0);

    const char *built = getenv("KNJ_BUILD_ARCH");
    if (!built || !*built) {
        printf("REFUSED: KNJ_BUILD_ARCH not set. An unguarded run is not a "
               "measurement.\n");
        return 7;
    }
    if (strncmp(p.gcnArchName, built, strlen(built)) != 0) {
        printf("DECLINED: built for %s, device is %s. Skip, not a result.\n",
               built, p.gcnArchName);
        return 6;
    }

    double clk_ghz = p.clockRate / 1.0e6;
    double peak_pk = p.multiProcessorCount * 128.0 * 2.0 * 2.0 * clk_ghz / 1.0e3;
    printf("device %s  arch %s  %d CU  %d MHz\n", p.name, p.gcnArchName,
           p.multiProcessorCount, p.clockRate / 1000);
    printf("packed-f16 peak, derived: %.1f TFLOP/s\n\n", peak_pk);

    /* --- roofline, measured fresh ------------------------------------ */
    uint32_t *dBW;
    float *dTmp;
    HIPCHECK(hipMalloc(&dBW, 256u << 20), "bw alloc");
    HIPCHECK(hipMalloc(&dTmp, 1024 * sizeof(float)), "bw out");
    HIPCHECK(hipMemset(dBW, 1, 256u << 20), "bw fill");
    double bw_vram = 0;
    {
        const int BLK = 256;
        const int bb = p.multiProcessorCount * 8;
        const uint32_t stride = (uint32_t)bb * BLK;
        const uint32_t n_big = (256u << 20) / 4;
        bw_read<<<bb, BLK>>>((const uint32_t *)dBW, n_big, stride, dTmp);
        HIPCHECK(hipDeviceSynchronize(), "bw warm");
        double t = now_us();
        for (int i = 0; i < 20; i++)
            bw_read<<<bb, BLK>>>((const uint32_t *)dBW, n_big, stride, dTmp);
        HIPCHECK(hipDeviceSynchronize(), "bw run");
        bw_vram = (double)n_big * 4 / ((now_us() - t) / 20 * 1e-6) / 1e9;
    }
    printf("measured VRAM roofline: %.1f GB/s (streaming 256 MB)\n\n", bw_vram);

    /* --- buffers, sized for the TIMING config (the largest) ---------- */
    const int S_T = 2048, H_T = 8, G_T = 4, B_T = 8;   /* GQA 4          */
    const size_t KVH_T = (size_t)H_T / G_T;
    const size_t qpre = (size_t)H_T * S_T * HD;         /* prefill Q/O    */
    const size_t kv_t = (size_t)B_T * KVH_T * S_T * HD; /* K/V all b      */
    const size_t qdec = (size_t)B_T * H_T * HD;

    uint16_t *hQ = (uint16_t *)malloc(qpre * sizeof(uint16_t));
    uint16_t *hK = (uint16_t *)malloc(kv_t * sizeof(uint16_t));
    uint16_t *hV = (uint16_t *)malloc(kv_t * sizeof(uint16_t));
    float *hRef = (float *)malloc(qpre * sizeof(float));
    float *hGot = (float *)malloc(qpre * sizeof(float));
    uint16_t *hGot16 = (uint16_t *)malloc(qpre * sizeof(uint16_t));
    if (!hQ || !hK || !hV || !hRef || !hGot || !hGot16) {
        printf("host alloc failed\n");
        return 1;
    }

    uint16_t *dQ, *dK, *dV, *dO;
    HIPCHECK(hipMalloc(&dQ, qpre * sizeof(uint16_t)), "malloc Q");
    HIPCHECK(hipMalloc(&dK, kv_t * sizeof(uint16_t)), "malloc K");
    HIPCHECK(hipMalloc(&dV, kv_t * sizeof(uint16_t)), "malloc V");
    HIPCHECK(hipMalloc(&dO, qpre * sizeof(uint16_t)), "malloc O");

    int fails = 0;

    /* =================== VALIDATION (oracle-checked) ================= */
    const int valcfg[2][3] = { { 512, 4, 2 }, { 512, 4, 4 } }; /* S,H,G  */
    for (int v = 0; v < 2; v++) {
        const int S = valcfg[v][0], H = valcfg[v][1], G = valcfg[v][2];
        const int B = 2;
        const size_t kvh_n = (size_t)H / G;
        const size_t qn = (size_t)H * S * HD;
        const size_t kn = (size_t)B * kvh_n * S * HD;
        const size_t qdn = (size_t)B * H * HD;

        fill_rand(hQ, qn);
        fill_rand(hK, kn);
        fill_rand(hV, kn);
        HIPCHECK(hipMemcpy(dQ, hQ, qn * sizeof(uint16_t),
                           hipMemcpyHostToDevice), "up Q");
        HIPCHECK(hipMemcpy(dK, hK, kn * sizeof(uint16_t),
                           hipMemcpyHostToDevice), "up K");
        HIPCHECK(hipMemcpy(dV, hV, kn * sizeof(uint16_t),
                           hipMemcpyHostToDevice), "up V");

        /* prefill: full oracle */
        dim3 gp(H, S / QT);
        HIPCHECK(hipMemset(dO, 0, qn * sizeof(uint16_t)), "clr O");
        attn_prefill<<<gp, BLKP>>>(dQ, dK, dV, dO, S, G);
        HIPCHECK(hipDeviceSynchronize(), "prefill");
        HIPCHECK(hipMemcpy(hGot16, dO, qn * sizeof(uint16_t),
                           hipMemcpyDeviceToHost), "rd O");
        f16_to_f32(hGot16, hGot, qn);
        ref_prefill(hQ, hK, hV, hRef, S, H, G, S);
        double rp = rel_rmse(hRef, hGot, qn);
        int okp = rp < 5e-3;

        /* decode: full oracle. Decode Q/O live in [B*H][HD]; the first
         * qdn halves of hQ/dQ serve as that buffer for kernel AND ref. */
        HIPCHECK(hipMemset(dO, 0, qdn * sizeof(uint16_t)), "clr Od");
        attn_decode<<<B * H, BLKD>>>(dQ, dK, dV, dO, S, H, G);
        HIPCHECK(hipDeviceSynchronize(), "decode");
        HIPCHECK(hipMemcpy(hGot16, dO, qdn * sizeof(uint16_t),
                           hipMemcpyDeviceToHost), "rd Od");
        f16_to_f32(hGot16, hGot, qdn);
        ref_decode(hQ, hK, hV, hRef, S, H, G, B, B * H);
        double rd = rel_rmse(hRef, hGot, qdn);
        int okd = rd < 5e-3;

        printf("validate S=%d H=%d G=%d%s  prefill relRMSE %.2e %s   "
               "decode relRMSE %.2e %s\n",
               S, H, G, G == H ? " (MQA)" : " (GQA)", rp, okp ? "PASS" : "FAIL",
               rd, okd ? "PASS" : "FAIL");
        if (!okp || !okd) fails++;
    }

    /* =================== TIMING (spot-oracled) ======================= */
    fill_rand(hQ, qpre);
    fill_rand(hK, kv_t);
    fill_rand(hV, kv_t);
    HIPCHECK(hipMemcpy(dQ, hQ, qpre * sizeof(uint16_t),
                       hipMemcpyHostToDevice), "up Qt");
    HIPCHECK(hipMemcpy(dK, hK, kv_t * sizeof(uint16_t),
                       hipMemcpyHostToDevice), "up Kt");
    HIPCHECK(hipMemcpy(dV, hV, kv_t * sizeof(uint16_t),
                       hipMemcpyHostToDevice), "up Vt");

    /* spot oracle: 64 q rows per head on prefill, 8 obs on decode */
    {
        dim3 gp(H_T, S_T / QT);
        attn_prefill<<<gp, BLKP>>>(dQ, dK, dV, dO, S_T, G_T);
        HIPCHECK(hipDeviceSynchronize(), "spot prefill");
        HIPCHECK(hipMemcpy(hGot16, dO, qpre * sizeof(uint16_t),
                           hipMemcpyDeviceToHost), "rd spot");
        f16_to_f32(hGot16, hGot, qpre);
        ref_prefill(hQ, hK, hV, hRef, S_T, H_T, G_T, 64);
        double rp = rel_rmse_prefill_rows(hRef, hGot, S_T, H_T, 64);

        attn_decode<<<B_T * H_T, BLKD>>>(dQ, dK, dV, dO, S_T, H_T, G_T);
        HIPCHECK(hipDeviceSynchronize(), "spot decode");
        HIPCHECK(hipMemcpy(hGot16, dO, qdec * sizeof(uint16_t),
                           hipMemcpyDeviceToHost), "rd spot d");
        f16_to_f32(hGot16, hGot, qdec);
        ref_decode(hQ, hK, hV, hRef, S_T, H_T, G_T, B_T, 8);
        double rd = rel_rmse(hRef, hGot, 8u * HD);
        printf("spot oracle at timing config: prefill %.2e %s, "
               "decode %.2e %s\n\n",
               rp, rp < 5e-3 ? "PASS" : "FAIL", rd,
               rd < 5e-3 ? "PASS" : "FAIL");
        if (rp >= 5e-3 || rd >= 5e-3) fails++;
    }

    /* prefill timing: best of 3 x 2 */
    dim3 gpre(H_T, S_T / QT);
    double pre_us = 1e18;
    attn_prefill<<<gpre, BLKP>>>(dQ, dK, dV, dO, S_T, G_T);
    HIPCHECK(hipDeviceSynchronize(), "pre warm");
    for (int r = 0; r < 3; r++) {
        double t = now_us();
        for (int i = 0; i < 2; i++)
            attn_prefill<<<gpre, BLKP>>>(dQ, dK, dV, dO, S_T, G_T);
        HIPCHECK(hipDeviceSynchronize(), "pre timed");
        double per = (now_us() - t) / 2.0;
        if (per < pre_us) pre_us = per;
    }
    /* decode timing: best of 3 x 10 */
    double dec_us = 1e18;
    attn_decode<<<B_T * H_T, BLKD>>>(dQ, dK, dV, dO, S_T, H_T, G_T);
    HIPCHECK(hipDeviceSynchronize(), "dec warm");
    for (int r = 0; r < 3; r++) {
        double t = now_us();
        for (int i = 0; i < 10; i++)
            attn_decode<<<B_T * H_T, BLKD>>>(dQ, dK, dV, dO, S_T, H_T, G_T);
        HIPCHECK(hipDeviceSynchronize(), "dec timed");
        double per = (now_us() - t) / 10.0;
        if (per < dec_us) dec_us = per;
    }

    /* FLOP/byte models.
     * prefill: causal pairs x (QK 2*HD + PV 2*HD) FLOP; DRAM minimum is
     * Q + O read/written once plus each kv-head's K/V once (tiles are
     * L2-resident across the S/QT workgroups that share them).
     * decode:  B*H*S*(2*HD + 2*HD); DRAM minimum is B*KVH K/V (the G
     * query heads sharing a kv head hit L2 after the first). */
    const double pre_pairs = (double)H_T * S_T * (S_T + 1) / 2.0;
    const double pre_flop = pre_pairs * HD * 4.0;
    const double pre_bytes =
        (double)H_T * S_T * HD * 2.0 /* Q */ +
        (double)H_T * S_T * HD * 2.0 /* O */ +
        (double)KVH_T * S_T * HD * 4.0; /* K+V */
    const double dec_flop = (double)B_T * H_T * S_T * HD * 4.0;
    const double dec_bytes = (double)B_T * KVH_T * S_T * HD * 4.0;

    const double pre_tf = pre_flop / (pre_us * 1e-6) / 1e12;
    const double pre_gbs = pre_bytes / (pre_us * 1e-6) / 1e9;
    const double dec_gbs = dec_bytes / (dec_us * 1e-6) / 1e9;
    const double dec_tf = dec_flop / (dec_us * 1e-6) / 1e12;

    printf("   kernel                     us   TFLOP/s  %%peak     GB/s  "
           "%%roof\n");
    printf("   prefill 2048x8 GQA4     %7.1f   %7.3f   %5.1f%%  %7.1f   "
           "%5.1f%%\n",
           pre_us, pre_tf, pre_tf / peak_pk * 100.0, pre_gbs,
           pre_gbs / bw_vram * 100.0);
    printf("   decode  step x8x8        %7.1f   %7.3f   %5.1f%%  %7.1f   "
           "%5.1f%%\n",
           dec_us, dec_tf, dec_tf / peak_pk * 100.0, dec_gbs,
           dec_gbs / bw_vram * 100.0);
    /* Amortized cycles per score round, CU-spread (throughput view):
     * total round count divided over CUs. Same formula before/after the
     * score-round rework, so it is the before/after number. */
    const double pre_rounds =
        (double)H_T * (S_T / QT) * (QT / 2) * (S_T / KT) * KT;
    const double dec_rounds = (double)B_T * H_T * S_T;
    const double cyc_per_us = clk_ghz * 1e3;
    const double pre_cyc =
        pre_us * cyc_per_us * p.multiProcessorCount / pre_rounds;
    const double dec_cyc =
        dec_us * cyc_per_us * p.multiProcessorCount / dec_rounds;
    printf("\n   cycles/score-round (CU-amortized): prefill %.0f, decode %.0f\n",
           pre_cyc, dec_cyc);

    printf("\n   decode is memory-bound by construction (docs/00 8.3): "
           "%.1f MB of K+V per step - %%roof carries it.\n"
           "   prefill is NEITHER DRAM-bound (%.1f%% roof: KV re-reads are "
           "L2-resident) NOR near\n"
           "   compute peak: %%peak is the headroom the score round still "
           "has. Its levers are\n"
           "   residency (tile size) and the serial chain - both applied "
           "in this file.\n",
           dec_bytes / 1e6, pre_gbs / bw_vram * 100.0);

    hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO);
    hipFree(dBW); hipFree(dTmp);
    free(hQ); free(hK); free(hV); free(hRef); free(hGot); free(hGot16);

    if (fails) {
        printf("RESULT: FAIL (%d)\n", fails);
        return 1;
    }
    printf("RESULT: PASS\n");
    return 0;
}
