// Kanjoos ISA probe — compile-only capability matrix for one target arch.
//
// Purpose: pin, per target (gfx1031 RDNA2, gfx1201 RDNA4), exactly which
// arithmetic primitives the installed device compiler will accept. A
// capability that fails to compile here must not appear in any kernel.
//
//   clang --offload-arch=<arch> -nogpuinc -nogpulib --cuda-device-only \
//         -x hip -std=c++17 -DKNJ_PROBE=<id> -c isa_probe.hip
// tools/isa_probe/run_isa_probe.sh automates the matrix and prints a table.
//
// WHY THIS FILE INCLUDES NO HIP HEADERS
// -------------------------------------
// An earlier revision included <hip/hip_runtime.h> and was compiled through
// hipcc. That made the probe a hostage to the ROCm *install*, not the ISA:
// a missing device-lib directory or a stale hipcc path made every probe
// "fail", and the matrix printed a uniform wall of REFUSED that looked like
// a hardware verdict. A capability probe must not depend on the toolchain it
// is testing. Everything below needs only the AMDGPU frontend + backend, so
// the file is self-contained and the results cannot be falsified by a broken
// install. The runner additionally preflights a trivial kernel and refuses
// to report a matrix if the compiler itself is unhealthy.
//
// MEASURED on AMD clang 23.0.0git (ROCm 10.x, G:\ROCM10RT-gfx1031):
//
//   probe                       gfx1031   gfx1201   note
//   sdot4  (builtin)            OK        REFUSED   A
//   sdot2  (builtin)            REFUSED   REFUSED   B
//   v_pk_add_f16    (asm)       OK        OK
//   v_pk_fma_f16    (asm)       OK        OK
//   v_pk_mad_f32    (asm)       OK        OK
//   v_dot2_i32_i16  (asm)       OK        OK        C
//   v_dot2c_i32_i16 (asm)       OK        OK        C
//   v_dot4_i32_i8   (asm)       OK        OK        C
//   v_dot8_i32_i4   (asm)       OK        OK        C
//   v_dot2c_f32_f16 (asm)       OK        OK        C
//   wmma f16 _w32 (gfx11 sig)   REFUSED   REFUSED   D
//   wmma f16 _gfx12             REFUSED   OK        E
//   wmma bf16 _w32 (gfx11 sig)  REFUSED   REFUSED   D
//   wmma bf16 _gfx12            REFUSED   OK        E
//   dot2_i16  (asm)             OK        OK        B
//   dot2c_i16 (asm)             OK        OK        B
//
// The "OK <insn>" cell the runner prints is read back out of the emitted .s,
// so each OK means the instruction is genuinely present, not merely that the
// compiler returned 0.
//
// A — the sdot4 builtin is arch-specific: it passes the feature gate on
//   gfx1031 and lowers to v_dot4c_i32_i8, a NATIVE RDNA2 dot-4 instruction
//   (not a shift/add expansion), and is REFUSED on gfx1201 with the verbatim
//   diagnostic
//       '__builtin_amdgcn_sdot4' needs target feature dot1-insts
//   That is a gate on the BUILTIN SPELLING, not a statement about the
//   silicon: the same dot, written as inline asm `v_dot4_i32_i8`, assembles
//   fine on gfx1201 (row below). Never let a builtin refusal stand in for a
//   capability claim — check the asm spelling before concluding anything.
// B — sdot2's builtin is REFUSED on both arches, and here the honest reading
//   is "this compiler exposes no call signature that matches the builtin",
//   NOT "no hardware". Two independent facts support that: the builtin wants
//   arg0 of type `short vector(2)` (its first parameter is an accumulator-like
//   operand, not an int predicate, which is what a caller naturally writes),
//   and the inline-asm row `v_dot2_i32_i16` below assembles on BOTH arches.
//   So int16 dot products are portable; only the builtin spelling is unusable.
// C — dot products as inline asm assemble on BOTH arches, in all four widths
//   (i16 x2, i8 x4, i4 x8, f16 x2 -> f32). An earlier revision
//   of this file claimed the assembler rejected them ("too few operands for
//   instruction"); that was an artifact of a broken toolchain, not an
//   assembler limitation. Consequence: write dots as inline asm ONCE and share
//   the source across arches. The builtin form is strictly worse — on gfx1031
//   it lowers to v_dot4c_i32_i8, a *clamping* variant with different semantics,
//   and on gfx1201 it does not compile at all.
// D — RDNA2 has no matrix units, so *every* WMMA spelling is genuinely
//   REFUSED on gfx1031. This is hardware, not a flag.
// E — gfx1201 DOES have WMMA and it needs no exotic flag. The `_gfx12`
//   builtin, compiled with -Xclang -target-feature -Xclang +wavefrontsize32
//   (gfx1201's native wave size), emits a real
//       v_wmma_f32_16x16x16_f16 v[9:16], v[0:3], v[4:7], v[9:16]
//   The gfx11-style `_w32` signature is REFUSED even with
//   +wmma-256b-insts,+wavefrontsize32: this build cannot enable that feature,
//   and its A/B operands (16x f16 each) do not match the wave32 layout. Use
//   the `_gfx12` spelling everywhere on RDNA4.
//   Operand widths are asymmetric: `_gfx12` takes A = 8 x f16, B = 8 x f16,
//   C/D = 8 x f32 — 8 VGPRs per operand, which is where the 256-bit in
//   "wmma-256b-insts" comes from.
//
// ---------------------------------------------------------------------------
// WHAT AN "OK" CELL HERE DOES AND DOES NOT PROVE  (measured 2026-10-06)
// ---------------------------------------------------------------------------
// This probe is COMPILE-ONLY. It runs as
//     clang --offload-arch=<arch> -nogpuinc -nogpulib --cuda-device-only -S
// and an OK cell proves exactly one thing: the instruction is present in the
// emitted ISA for that target. It does NOT prove the instruction can be used in
// a KERNEL that LINKS. Two measured counter-examples, both found while building
// the expert GEMM:
//
//   1. THE 3-OPERAND DOT SPELLING COMPILES AND DOES NOT LINK.
//         "v_dot4_i32_i8 %0, %1, %2"        -> OK here
//         ld.lld: error: too few operands for instruction
//                 v_dot4_i32_i8 v15, v20, v19
//      The accumulating form requires the accumulator as an explicit fourth
//      operand. Probes 5 and 6 above were corrected for this reason; they now
//      emit the same instructions they always did, in the only form that links.
//
//   2. THERE IS NO f16 DOT IN THIS TOOLCHAIN THAT LINKS ON BOTH TARGETS.
//         v_dot2c_f32_f16       gfx1031: invalid operand (src2 wants a dword)
//                               gfx1201: instruction not supported on this GPU
//         v_dot2_f16_16b_f16   both:     invalid instruction
//      Probe row 7 assembles on both arches and is listed as OK, and that is
//      true and useless: nothing built from it will link.
//
// The general rule this adds to the ones already in this file: **an assembler
// acceptance is not a link, and a compile-only probe cannot stand in for one.**
// A capability that a kernel depends on has to be verified by building and
// running the kernel, not by this table. The table answers "is this instruction
// in the ISA for this arch"; only a passing tier C run answers "can I use it".

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

typedef float    __attribute__((ext_vector_type(4)))  knj_v4f;
typedef float    __attribute__((ext_vector_type(8)))  knj_v8f;
typedef float    __attribute__((ext_vector_type(2)))  knj_v2f;
typedef _Float16 __attribute__((ext_vector_type(2)))  knj_v2h;
typedef _Float16 __attribute__((ext_vector_type(8)))  knj_v8h;
typedef _Float16 __attribute__((ext_vector_type(16))) knj_v16h;
typedef short    __attribute__((ext_vector_type(8)))  knj_v8b;
typedef short    __attribute__((ext_vector_type(16))) knj_v16b;
typedef short    __attribute__((ext_vector_type(2)))  knj_v2s;
typedef int      __attribute__((ext_vector_type(2)))  knj_v2i;

#if KNJ_PROBE == 0   // int8 dot4 builtin, 4-scalar form
extern "C" __global__ void knj_probe(knj_v4f *out) { out[0] = (float)__builtin_amdgcn_sdot4(3, 5, 0, 0); }
#elif KNJ_PROBE == 1 // int16 dot2 builtin, vector form
extern "C" __global__ void knj_probe(knj_v4f *out) {
    knj_v2s a = {3, 5}, b = {7, 11}, c = {0, 0};
    c = __builtin_amdgcn_sdot2(0, a, b, c);
    out[0] = (float)c.x;
}
#elif KNJ_PROBE == 2 // packed fp16 add
extern "C" __global__ void knj_probe(knj_v4f *out) {
    knj_v2h a, b; a.x = 1.0f; a.y = 2.0f; b.x = 3.0f; b.y = 4.0f;
    knj_v2h r;
    asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    knj_v4f o; o.x = (float)r.x; o.y = (float)r.y; o.z = 0; o.w = 0; out[0] = o;
}
#elif KNJ_PROBE == 3 // packed fp16 FMA
extern "C" __global__ void knj_probe(knj_v4f *out) {
    knj_v2h a, b; a.x = 1.0f; a.y = 2.0f; b.x = 3.0f; b.y = 4.0f;
    knj_v2h r = a;
    asm volatile("v_pk_fma_f16 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b));
    knj_v4f o; o.x = (float)r.x; o.y = (float)r.y; o.z = 0; o.w = 0; out[0] = o;
}
#elif KNJ_PROBE == 4 // packed fp16 -> f32 mixed FMA, register-held scale
extern "C" __global__ void knj_probe(knj_v4f *out) {
    knj_v2h a, b; a.x = 1.0f; a.y = 2.0f; b.x = 3.0f; b.y = 4.0f;
    knj_v2f one; one.x = 1.0f; one.y = 1.0f;
    knj_v2f r;
    asm volatile("v_pk_mad_f32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(one));
    out[0] = r.x;
}
#elif KNJ_PROBE == 5 // int8 dot4, inline asm audit
// FOUR operands, not three. See the note below: the 3-operand spelling
// assembles under --cuda-device-only and is REJECTED by ld.lld at link time.
extern "C" __global__ void knj_probe(knj_v4f *out) {
    int a = 3, b = 5, r = 0;
    asm volatile("v_dot4_i32_i8 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b));
    out[0] = (float)r;
}
#elif KNJ_PROBE == 6 // int4 dot8, inline asm audit
extern "C" __global__ void knj_probe(knj_v4f *out) {
    int a = 3, b = 5, r = 0;
    asm volatile("v_dot8_i32_i4 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b));
    out[0] = (float)r;
}
#elif KNJ_PROBE == 7 // packed-fp16 dot2 accumulating, inline asm audit
extern "C" __global__ void knj_probe(knj_v4f *out) {
    knj_v2h a, b; a.x = 1.0f; a.y = 2.0f; b.x = 3.0f; b.y = 4.0f;
    knj_v2f r;
    asm volatile("v_dot2c_f32_f16 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(knj_v2f{}));
    out[0] = r.x;
}
#elif KNJ_PROBE == 8 // WMMA f16, gfx11-style wave32 signature (A/B = 16 x f16)
extern "C" __global__ void knj_probe(knj_v8f *out, knj_v16h a, knj_v16h b, knj_v8f c) {
    out[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
#elif KNJ_PROBE == 9 // WMMA f16, gfx12 signature (A/B = 8 x f16) -- the one that works
extern "C" __global__ void knj_probe(knj_v8f *out, knj_v8h a, knj_v8h b, knj_v8f c) {
    out[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}
#elif KNJ_PROBE == 10 // WMMA bf16, gfx11-style wave32 signature
extern "C" __global__ void knj_probe(knj_v8f *out, knj_v16b a, knj_v16b b, knj_v8f c) {
    out[0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}
#elif KNJ_PROBE == 11 // WMMA bf16, gfx12 signature -- the one that works
extern "C" __global__ void knj_probe(knj_v8f *out, knj_v8b a, knj_v8b b, knj_v8f c) {
    out[0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
#elif KNJ_PROBE == 12 // int16 dot2 accumulating, inline asm audit
// This is the ASM spelling of what probe 1's builtin could not express. It
// is the cell that licenses the int16 dot in any kernel: it assembles on
// BOTH arches, so an int16 dot path needs no per-arch #if.
extern "C" __global__ void knj_probe(knj_v4f *out) {
    int a = 3, b = 5, r = 0;
    asm volatile("v_dot2_i32_i16 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b));
    out[0] = (float)r;
}
#elif KNJ_PROBE == 13 // int16 dot2 non-accumulating (clamping) variant
extern "C" __global__ void knj_probe(knj_v4f *out) {
    int a = 3, b = 5, r = 0;
    asm volatile("v_dot2c_i32_i16 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    out[0] = (float)r;
}
#elif KNJ_PROBE == 90 // preflight: trivial kernel, no ISA features at all
extern "C" __global__ void knj_probe(float *out) { out[0] = 1.0f; }
#else
#error "unknown KNJ_PROBE id"
#endif
