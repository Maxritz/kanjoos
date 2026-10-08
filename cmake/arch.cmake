# cmake/arch.cmake -- the SINGLE source of truth for per-target flags.
#
# Every arch difference in the build lives here and nowhere else. Nothing above
# this file, cmake/rocm.cmake, src/platform/ and kernels/ contains an arch
# conditional (docs/07-build-platforms.md section 2, the rule that keeps this
# tree from rotting).
#
# Each flag below carries the MEASUREMENT that justifies it. A flag with no
# measured reason does not belong in this file; if you cannot say why a flag is
# here, delete it and see whether the build still works.

set(KNJ_SUPPORTED_ARCHS gfx1031 gfx1201)

# ---------------------------------------------------------------------------
# Per-arch flags.
# ---------------------------------------------------------------------------
function(knj_arch_flags arch out_var)
  set(_flags "")
  if(NOT arch IN_LIST KNJ_SUPPORTED_ARCHS)
    message(FATAL_ERROR "knj: unsupported arch '${arch}'. Supported: ${KNJ_SUPPORTED_ARCHS}")
  endif()

  list(APPEND _flags "--offload-arch=${arch}")

  # RDNA2 and RDNA4 targets are both natively wave32. All WMMA builtins are
  # wave32-only (rocWMMA states this for both generations), and every wave
  # reduction must be unrolled to exactly FIVE shuffle steps, never six.
  list(APPEND _flags "-mno-wavefrontsize64")

  # RDNA4 only: the `_gfx12` WMMA builtins are selected by this target feature.
  #
  # MEASURED (docs/00-verified-facts.md sections 1.1 and 7.0): the working
  # invocation is
  #     __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(...)
  # compiled with
  #     -Xclang -target-feature -Xclang +wavefrontsize32
  # emitting exactly
  #     v_wmma_f32_16x16x16_f16 v[9:16], v[0:3], v[4:7], v[9:16]
  #
  # The `wmma-256b-insts` feature this plan used to hunt for CANNOT be enabled
  # in this build and is not needed. Do not go looking for it again.
  #
  # NOTE on the host code: this flag is accepted as a *device* target feature.
  # It is NOT valid in the host pass. cmake/rocm.cmake splits the flag lists so
  # the host pass never sees it. That split is not a convenience: a host-pass
  # `-DKNJ_BUILD_ARCH="gfx1201"` does not survive hipcc's command-string
  # re-spawn (docs/00-verified-facts.md section 8.7), which is why the arch
  # reaches tier C drivers through the ENVIRONMENT instead.
  if(arch STREQUAL "gfx1201")
    list(APPEND _flags "-Xclang" "-target-feature" "-Xclang" "+wavefrontsize32")
  endif()

  # Kernels select their arch variant on this macro. It is a DEVICE-pass macro:
  # defining it on the host pass is harmless but useless, because host code has
  # no __gfxNNNN__ semantics.
  list(APPEND _flags "-D__${arch}__=1")
  list(APPEND _flags "-DKNJ_ARCH_NAME=\"${arch}\"")

  set(${out_var} "${_flags}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Link flags.
# ---------------------------------------------------------------------------
#
# MEASURED (docs/00-verified-facts.md section 7.0): on this machine hipcc cannot
# resolve the device bitcode directory through EITHER --rocm-path or
# --rocm-device-lib-path, in both the ROCm 7 layout (<rocm>/amdgcn/bitcode) and
# the ROCm 10 layout (<rocm>/lib/llvm/amdgcn/bitcode). The device library files
# are present; hipcc simply will not look there. The working link line passes
# -nogpulib and names amdhip64 by hand.
#
# -nogpulib has two real consequences, both already hit and both respected by
# the kernels in this tree (docs/00-verified-facts.md section 8.5):
#   * NO DEVICE-SIDE LIBM. A kernel using sqrt/exp/anything from libm will not
#     build. `__expf` is fine -- it is the native v_exp_f32 instruction.
#   * NO gridDim IN ANY KERNEL. gfx1201 materialises gridDim through
#     __ockl_get_num_groups, whose inline definition lives in the device runtime
#     we are not linking, so the link fails with `undefined hidden symbol`.
#     Every grid-stride loop in this tree therefore takes its stride and its
#     element count as KERNEL ARGUMENTS. See kernels/knj_kernels.hip.
function(knj_link_flags out_var)
  set(${out_var} "-nogpulib" PARENT_SCOPE)
endfunction()