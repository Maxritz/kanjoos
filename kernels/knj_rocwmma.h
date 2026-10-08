// kernels/knj_rocwmma.h -- the rocWMMA tile contract for gfx1201 model ops.
//
// WHY THIS FILE EXISTS (docs/00-verified-facts.md section 7.4a is now answered)
// ---------------------------------------------------------------------------
// The repo spent a long time treating "the `_gfx12` A/B operand layout" as a
// blocking unknown: the raw builtins `__builtin_amdgcn_wmma_*_gfx12` take A and
// B as *4 VGPRs each* (8 x f16) with a per-lane interleave that is NOT 8
// consecutive f16, and a tiled GEMM written on the naive assumption reads the
// wrong elements silently.
//
// That question is CLOSED. rocWMMA 2.2.1 ships in G:/ROCM10RT-gfx1201, its
// `internal/wmma_impl.hpp` instantiates the gfx12 path for
// AMDGCN_ARCH_ID_GFX1200/GFX1201, and it calls exactly the builtins above. The
// per-lane register mapping lives inside `load_matrix_sync` / `store_matrix_sync`
// and never has to be re-derived by hand. A 16x16x16 f16->f32 probe was built
// and run on the RX 9070 XT (gfx1201) and matched a host oracle bit-exactly,
// 0/256 mismatches.
//
// So the rule is: **do not hand-roll the gfx12 operand layout, and do not write
// inline `__builtin_amdgcn_wmma_*_gfx12`.** Express the model operations as
// rocWMMA fragments and let the library own the mapping.
//
// BUILD FACTS (both required, both non-obvious -- verified, rc=0):
//   * `--rocm-path` must be the LLVM tree, `.../lib/llvm`, NOT the ROCm root.
//     Pointing it at the ROCm root gives "cannot find ROCm device library".
//   * `-std=c++17` is mandatory. rocWMMA headers use `std::apply`; under c++14
//     they fail with `no member named 'apply' in namespace 'std'` plus a cascade
//     of constexpr-init errors in io_bearer_base.hpp / register_layout_traits_impl.hpp.
//
// BUILD LINE (the one that works):
//   hipcc --rocm-path="G:/ROCM10RT-gfx1201/lib/llvm" -std=c++17 -O2 \
//         --offload-arch=gfx1201 -Xclang -target-feature \
//         -Xclang +wavefrontsize32 <driver>.hip -o <driver>.exe
//
// The wave32 flag is inert on RDNA4 (the hardware is natively wave32; the flag
// warns "not a recognized feature (ignoring)"), but it is kept because
// every WMMA builtin is wave32-only and the intent should be explicit.

#ifndef KNJ_ROCWMMA_H
#define KNJ_ROCWMMA_H

#include <rocwmma/rocwmma.hpp>
#include <hip/hip_runtime.h>

#include "knj_pack.h"

namespace knjrw {

// The only tile shape the gfx12 WMMA builtins support: 16 x 16 x 16, wave32.
// There is no 32-wide or K=32 variant to reach for.
constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

// Expert weights use group 128 (kernels/knj_pack.h KNJ_GROUP_EXPERT) and the
// 3 B/group metadata rule. A group of 128 at 4 bits is 64 payload bytes; the
// packer pads by 1 to keep the next column 4-byte aligned, so the on-disk
// stride per (row, group) is 68 bytes. That stride comes from knj_pack.h so the
// kernel and the packer cannot drift -- an earlier revision of this repo had
// the same pack quoted with two different byte costs.
constexpr int GROUP = KNJ_GROUP_EXPERT;                        // 128
constexpr int GROUP_STRIDE = KNJ_GROUP_STRIDE_BYTES(4, GROUP);  // 68
static_assert(GROUP_STRIDE == 68, "W4 g128 group stride is 68 bytes");

// Offset of one (row, group) group inside a packed expert weight blob laid out
// as [expert][output_row][group] with a fixed per-row stride.
__host__ __device__ __forceinline__
uint64_t group_offset(uint64_t row, int g, int groups_per_row) {
    return row * (uint64_t)groups_per_row * (uint64_t)GROUP_STRIDE
         + (uint64_t)g * (uint64_t)GROUP_STRIDE;
}

// Dequantise one 4-bit code to f16.
//
// The repo's codes are UNSIGNED: `u = q + zero_point`, and the int8 dot path
// recovers the signed weight by folding `(zp - 8)` into a per-group correction
// term instead of correcting per element (kernels/knj_pack.h).
//
// The WMMA path does not need that trick. It materialises the tile into f16
// before `load_matrix_sync` anyway, so the zero point is applied once, right
// here, and there is no correction term to carry. This is a real structural
// advantage of the f16-fragment path over the int8-dot path: no de-interleaved
// activation layout, no group correction, no accumulator headroom question.
__host__ __device__ __forceinline__
__half code_to_half(uint32_t nibble, __half scale, uint32_t zero_point) {
    return __float2half((float)((int)nibble - (int)zero_point) * __half2float(scale));
}

// Extract nibble `k` from a packed W4 row payload (low nibble first, ascending k).
__host__ __device__ __forceinline__
uint32_t nibble_at(const uint8_t *payload, int k) {
    const uint8_t byte = payload[k >> 1];
    return (k & 1) ? (uint32_t)(byte >> 4) : (uint32_t)(byte & 0x0f);
}

// The scale/zero-point pair for group g lives immediately after the group's
// payload: [payload: 64 B][scale: fp16][zero_point: u8][pad: 1 B].
__host__ __host__ __device__ __forceinline__
void group_meta(const uint8_t *blob, int g, __half *scale, uint32_t *zp) {
    const uint8_t *meta = blob + (uint64_t)g * GROUP_STRIDE + (GROUP * 4 / 8);
    __half s;
    __builtin_memcpy(&s, meta, sizeof(__half));
    *scale = s;
    *zp = meta[2];
}

}  // namespace knjrw

#endif  // KNJ_ROCWMMA_H
