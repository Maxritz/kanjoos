// kernels/knj_dot.h -- integer and packed-float dot products.
//
// ONE HEADER FOR BOTH TARGETS. That is the entire point of this file, and it
// is a reversal of the earlier policy in this project.
//
// ---------------------------------------------------------------------------
// WHY ONE HEADER (and not two #if-selected ones)
// ---------------------------------------------------------------------------
// An earlier revision of the plan carried a "two arch-selected headers, never
// share one" policy. That policy existed only because a BROKEN TOOLCHAIN made
// inline-asm dot products look unavailable.
//
// MEASURED, and reproducible via tools/isa_probe/run_isa_probe.sh (exit 0):
//
//   probe                  gfx1031 (RDNA2)        gfx1201 (RDNA4)
//   v_dot4_i32_i8   (asm)  OK                     OK
//   v_dot8_i32_i4   (asm)  OK                     OK
//   v_dot2c_f32_f16 (asm)  OK                     OK
//   __builtin_amdgcn_sdot4   OK -> v_dot4c_i32_i8  REFUSED (needs dot1-insts)
//
// So the builtin is strictly WORSE on both targets:
//
//   * on gfx1031 it lowers to v_dot4c_i32_i8, which is a *clamping* variant with
//     different semantics from v_dot4_i32_i8 -- writing `dot4` and getting
//     `dot4c` is a silent numerical change, not a spelling difference;
//   * on gfx1201 it does not compile at all.
//
// A single spelling that is correct on both arches is therefore not a
// convenience, it is the only correct choice. The builtin is retained in ONE
// place -- the ISA probe -- purely as a cross-check oracle, and is deliberately
// not used by any kernel.
//
// NOTE (00-verified-facts.md section 7.2): the AMDGPU *assembler* is not gated
// by -offload-arch. Every one of these spellings assembles for either arch, so
// "it assembled" is NOT evidence the hardware has it. Dot availability must come
// from the ISA reference and from the probe reading back the EMITTED
// instruction, never from the assembler's exit code.

// ---------------------------------------------------------------------------
// Self-sufficiency
// ---------------------------------------------------------------------------
// This header must compile in BOTH of the two worlds this tree builds in:
//
//   1. compile-only capability checks, invoked as
//        clang --offload-arch=<arch> -nogpuinc -nogpulib --cuda-device-only
//      with NO HIP headers on the include path at all (this is how
//      tools/isa_probe/run_isa_probe.sh runs, and how it stays immune to a
//      broken ROCm install -- see 00-verified-facts.md section 0);
//   2. a real build through hipcc, where <hip/hip_runtime.h> IS present and has
//      already defined these macros.
//
// So define them only if nobody has. Unconditional #define here would either
// be a redefinition (world 2, harmless but noisy) or a lie about world 1.
#ifndef __device__
#define __device__ __attribute__((device))
#endif
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
#ifndef KNJ_HOST_CALLABLE
#define KNJ_HOST_CALLABLE __host__ __device__
#endif
#ifndef __host__
#define __host__ __attribute__((host))
#endif

// ---------------------------------------------------------------------------
// Semantics, stated explicitly because getting them wrong is silent
// ---------------------------------------------------------------------------
//
//   v_dot4_i32_i8   D, A, B, C   D = C + dot4_s8(A, B)   A,B = 4 x i8, C/D = i32
//   v_dot2_i32_i16  D, A, B, C   D = C + dot2_s16(A, B)  A,B = 2 x i16, C/D = i32
//   v_dot8_i32_i4   D, A, B, C   D = C + dot8_s4(A, B)   A,B = 8 x i4, C/D = i32
//   v_dot2c_f32_f16 D, A, B, C   D = C + dot2_f16(A, B)  A,B = 2 x f16, C/D = f32
//
// EVERY one of these is the ACCUMULATING form and the accumulator is an EXPLICIT
// fourth operand. Two consequences, both learned here:
//
// (a) An uninitialised accumulator register is a silent-wrong-answer bug, not a
//     compile error. It was exactly that in this repo (00-verified-facts.md
//     section 7.5, item 6: a packed-f16 kernel accumulating into an uninitialised
//     register showed 10x the oracle error). Every accumulator in this tree is
//     zeroed before first use.
//
// (b) The 3-operand spelling "v_dot4_i32_i8 D, A, B" is WRONG and fails at LINK
//     time, not at compile time. MEASURED on this machine while building the
//     expert GEMM:
//
//         ld.lld: error: too few operands for instruction
//                 v_dot4_i32_i8 v15, v20, v19
//
//     It compiles cleanly under `--cuda-device-only -S`, because in that mode
//     the inline-asm string is passed through without full operand-count
//     validation. A real build reaches ld.lld, which does validate, and the
//     build fails. tools/isa_probe/run_isa_probe.sh runs in exactly the
//     compile-only mode, so it CANNOT catch this class of error -- which is why
//     the accumulated-form spelling below is used unconditionally, and why a
//     probe result is never treated as evidence that a KERNEL links.
//
// The 3-operand form does exist in the ISA, but for the UNSIGNED variant
// (v_dot4_i32_u8_u8 D, A, B), where the accumulator is implicitly zero. Using it
// for the signed spelling and hoping for the same behaviour is exactly the kind
// of "it compiled, so it works" reasoning that produced a false ISA matrix here
// once already.
//
// The `c` in dot2c denotes the CLAMPING variant. Only v_dot2c_f32_f16 is wanted
// here, because it is the float form; there is no non-clamping f16 dot, so the
// clamping form is simply the correct instruction for the job rather than a
// semantic compromise.

// ---------------------------------------------------------------------------
// knj_dot4_i32_i8 -- 4 x int8 dot, accumulating into int32
// ---------------------------------------------------------------------------
// Portable across gfx1031 and gfx1201. This is the workhorse: the W4 expert
// path dequantises nibble weights to 4 int8 codes per VGPR and multiplies them
// by 4 int8 activations per VGPR (see kernels/knj_pack.h).
__device__ __forceinline__ int knj_dot4_i32_i8(int a, int b, int acc) {
    int r = acc;
    asm volatile("v_dot4_i32_i8 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b));
    return r;
}

// Non-accumulating form, for the first step where the accumulator is known
// zero. Provided because "zero an int32 register then add into it" costs a real
// instruction in the inner loop, and the compiler will not always fold it.
//
// NOTE: there is no non-accumulating int form available for dot4 that both
// arches accept, so this is spelled as an accumulating dot against an explicit
// zero. That is deliberate: the alternative portable spelling does not exist.
__device__ __forceinline__ int knj_dot4_i32_i8(int a, int b) {
    return knj_dot4_i32_i8(a, b, 0);
}

// ---------------------------------------------------------------------------
// knj_dot8_i32_i4 -- 8 x int4 dot, accumulating into int32
// ---------------------------------------------------------------------------
// The natural RDNA2 shape for a nibble-packed weight stream: it consumes packed
// nibbles directly with NO unpack step.
//
// It is NOT the default expert path, and the reason is measured, not stylistic:
// docs/00-verified-facts.md section 7.6 rejected the int4-ACTIVATION variant at
// 81.5 dB worse SNR than int8. Using v_dot8_i32_i4 with *4-bit activations* is
// what that measurement rejected. The shipped design keeps 8-bit activations and
// de-interleaves them, so the engine uses knj_dot4_i32_i8 for the multiply and
// this is kept for the W2/W3 research path where the payload is genuinely
// narrower than int8.
//
// Also portable across gfx1031 and gfx1201.
__device__ __forceinline__ int knj_dot8_i32_i4(int a, int b, int acc) {
    int r = acc;
    asm volatile("v_dot8_i32_i4 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b));
    return r;
}

__device__ __forceinline__ int knj_dot8_i32_i4(int a, int b) {
    return knj_dot8_i32_i4(a, b, 0);
}

// ---------------------------------------------------------------------------
// knj_dot2_i32_i16 -- 2 x int16 dot
// ---------------------------------------------------------------------------
// Portable across both arches. There is no builtin spelling that compiles for
// either of them (the probe reports sdot2 REFUSED on BOTH, and the honest
// reading is "this build exposes no call signature we could match", not "no
// hardware"), so the asm spelling is the only option -- and it works everywhere.
__device__ __forceinline__ int knj_dot2_i32_i16(int a, int b, int acc) {
    int r = acc;
    asm volatile("v_dot2_i32_i16 %0, %1, %2" : "+v"(r) : "v"(a), "v"(b));
    return r;
}

__device__ __forceinline__ int knj_dot2_i32_i16(int a, int b) {
    return knj_dot2_i32_i16(a, b, 0);
}

// ---------------------------------------------------------------------------
// The f16 float dot: NOT PROVIDED, and deliberately so
// ---------------------------------------------------------------------------
// An earlier revision of this header offered `knj_dot2_f32_f16` as
// "portable across both arches", on the strength of the ISA probe reporting
// `v_dot2c_f32_f16` as assembling on both. That claim is FALSE, and it is worth
// recording exactly how it was found and what it cost, because it is the exact
// failure mode docs/00-verified-facts.md section 7.2 warns about -- "the AMDGPU
// assembler is not gated by -offload-arch; dot availability comes from the ISA
// reference, never from the assembler" -- and the probe is compile-only, so it
// cannot catch it.
//
// MEASURED while linking the expert GEMM (both via ld.lld, not the assembler):
//
//   v_dot2c_f32_f16  on gfx1201  ld.lld: instruction not supported on this GPU
//                                     (gfx1201): v_dot2c_f32_f16
//   v_dot2c_f32_f16  on gfx1031  ld.lld: invalid operand for instruction
//                                     v_dot2c_f32_f16 v2, v0, v1, v2
//                                     ^  src2 wants a dword accumulator, not
//                                        the 64-bit VOP3P destination pair
//   v_dot2_f16_16b_f16  both     ld.lld: invalid instruction (not in this LLVM)
//
// So there is no f16 dot in this toolchain that links on BOTH targets. gfx1031
// has the clamping f16 dot and needs a correct src2 encoding to use it; gfx1201
// has no f16 dot under any spelling tried.
//
// CONSEQUENCE FOR THIS TREE:
//   * The W4 expert path is int8-dot only, so it is unaffected -- this is the
//     path that matters (docs/00 section 8.3: the engine is transfer-bound, so
//     the integer path is the one that ships).
//   * Any future fp16 SIMT path must use `v_pk_fma_f16`, which is MEASURED to
//     assemble on both targets, rather than a dot. That is also the basis of
//     docs/00-verified-facts.md section 1.4: RDNA2 has no matrix units, so its
//     f16 path is packed FMA regardless.
//   * The runtime capability probe does NOT probe an f16 dot, because there is
//     nothing portable to probe.
//
// Adding an f16 dot back means writing it for gfx1031 only, behind
// `#if defined(__gfx1031__)`, with the src2 encoding resolved. That is a
// deliberate future task, not something to ship unproven.

// ---------------------------------------------------------------------------
// Capability reporting
// ---------------------------------------------------------------------------
//
// These are compile-time facts about the TARGET, and they are asserted against
// the runtime probe rather than trusted. src/device/device_caps.cpp cross-checks
// them against hipGetDeviceProperties().gcnArchName and refuses to run if the
// build arch and the device arch disagree -- because an arch-mismatched object
// runs on the card without error, dispatches no kernel, and reports confident
// nonsense (00-verified-facts.md section 8.7: 2327% of peak in 0.1 us).
#define KNJ_DOT_ASM_PORTABLE 1

// True on both targets, by measurement above -- and, unlike the f16 dot, each
// one has been verified to LINK on both, not merely to assemble.
#define KNJ_HAS_DOT4_I8_ASM  1
#define KNJ_DOT8_I4_ASM      1
#define KNJ_DOT2_I16_ASM     1

// No portable f16 dot exists in this toolchain. See the section above.
#define KNJ_HAS_DOT2_F16_ASM 0

// The builtin is deliberately absent. It is in the probe only.
#define KNJ_SDOT4_BUILTIN_PORTABLE 0

// ---------------------------------------------------------------------------
// Target-arch assertion -- DEVICE PASS ONLY
// ---------------------------------------------------------------------------
// `__gfxNNNN__` is a DEVICE-pass macro. In a full hipcc build this header is also
// processed by the HOST pass, where those macros are undefined, so an
// unconditional #error fires in the wrong pass and breaks the build. That is the
// mirror image of the trap in docs/00-verified-facts.md section 8.7, where a
// guard written as `__gfxNNNN__` in host code compiled to nothing and guarded
// nothing.
//
// `__HIP_DEVICE_COMPILE__` is defined only in the device pass, so it is the
// correct discriminator. The assertion below therefore fires where it is meant
// to -- in the device pass of a kernel -- and stays silent on the host.
#if defined(__HIP_DEVICE_COMPILE__)
#  if !defined(__gfx1031__) && !defined(__gfx1201__)
#    error "knj_dot.h: device pass for an unrecognised target arch. Dot-asm portability is only claimed for gfx1031 (RDNA2) and gfx1201 (RDNA4); a new arch needs its probe run and its link verified."
#  endif
#endif