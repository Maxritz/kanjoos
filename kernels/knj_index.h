// kernels/knj_index.h -- thread and block index, obtainable without HIP headers.
//
// ---------------------------------------------------------------------------
// WHY THIS FILE EXISTS (measured, not stylistic)
// ---------------------------------------------------------------------------
// The obvious spelling is `threadIdx.x` / `blockIdx.x`. Under a full hipcc
// build those come from <hip/hip_runtime.h>, which defines them on top of
// compiler builtins. Compile the same file with `-nogpuinc` -- which is how
// tools/isa_probe/run_isa_probe.sh and the compile-only capability checks run,
// and which is what keeps those checks immune to a broken ROCm install -- and
// `threadIdx` / `blockIdx` are UNDECLARED:
//
//     error: use of undeclared identifier 'blockIdx'
//     error: use of undeclared identifier 'threadIdx'
//
// So a kernel written against threadIdx cannot be compile-checked on a machine
// whose HIP headers are missing, which is precisely the machine where you most
// want a cheap check. The fix is to use the compiler builtins DIRECTLY:
//
//     __builtin_amdgcn_workitem_id_x()     thread index within its wave
//     __builtin_amdgcn_workgroup_id_x()     work-group (block) index
//
// MEASURED: both compile and emit a kernel on gfx1031 AND gfx1201 with
// `-nogpuinc -nogpulib --cuda-device-only` and no HIP headers on the include
// path. They are also perfectly valid in a full hipcc build, because they are
// language builtins rather than library declarations. One spelling, both worlds.
//
// ---------------------------------------------------------------------------
// WHAT IS DELIBERATELY NOT USED HERE: gridDim
// ---------------------------------------------------------------------------
// `gridDim` is not usable in this tree. MEASURED (docs/00-verified-facts.md
// section 8.5): under the link line this project is forced to use, gfx1201
// materialises gridDim through `__ockl_get_num_groups`, whose inline definition
// lives in the device runtime that `-nogpulib` excludes from the link. The link
// then fails with `undefined hidden symbol`.
//
// Consequences, all of them load-bearing:
//   * every launch configuration is passed to the kernel as an argument;
//   * every loop is an explicit grid-stride loop over that argument;
//   * the launch helper (src/device/) is the single place that decides how many
//     blocks and threads, so this policy is enforced in one file.
//
// The kernels here are launched 1-D. `knj_bid()` is therefore the linear block
// index. If a 2-D or 3-D launch is ever needed, the launch helper must be
// changed first -- silently flattening a 2-D grid in a helper is how a kernel
// ends up reading out of bounds.

#ifndef KNJ_INDEX_H
#define KNJ_INDEX_H

#ifndef __device__
#define __device__ __attribute__((device))
#endif
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif

// Thread index within the work-group. On a 32-wide wave this is 0..63 for a
// 2-wave block, which is the block shape every kernel in this tree launches
// with. If a block is not a multiple of the wave size, the tail threads of the
// last wave are still addressable here and MUST be masked by the launch helper
// or by the kernel's own bounds check -- the hardware does not do it for us.
__device__ __forceinline__ unsigned knj_tid(void) {
    return __builtin_amdgcn_workitem_id_x();
}

// Linear block index for a 1-D launch.
__device__ __forceinline__ unsigned knj_bid(void) {
    return __builtin_amdgcn_workgroup_id_x();
}

// Linear thread index across the whole grid, given the launch configuration.
// Both terms are passed in because gridDim cannot be read from inside a kernel.
__device__ __forceinline__
unsigned knj_gid(unsigned threads_per_block, unsigned blocks) {
    return knj_bid() * threads_per_block + knj_tid();
}

#endif  /* KNJ_INDEX_H */