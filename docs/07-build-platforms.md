# 07 — Build, platforms, targets

Windows 11 and Linux are both first-class. ROCm 10.1 is the baseline. Two GPU
targets: RDNA4 `gfx1201` and RDNA2 `gfx1031`.

---

## 1. Toolchain facts on this machine

| path | what it is |
|---|---|
| `G:\ROCM10RT-gfx1201` | full RDNA4 runtime: hipcc, hipBLAS **and hipBLASLt**, rocBLAS, rocSOLUTION, rocFFT, rocRAND, MIOpen, hipTensor, hipDNN, hiprtc |
| `G:\ROCM10RT-gfx1031` | RDNA2 runtime: hipcc, hipBLAS, rocBLAS, hiprtc, kpack {blas, fft, rand} |
| `G:\rocm-10` | ROCm 10.x (TheRock) source tree — `SUPPORTED_GPUS.md` marks gfx1030–gfx1036 build-passing, sanity-tested **and release-ready** on Linux |
| `C:\…\kpack\*.kpack` | `KPAK` v1 containers for gfx1031: blas (44.5 MB), fft, rand, rccl, torch, torchvision |

Consequences that shape the build:

* **gfx1031 is a supported target, not a hack.** It is release-ready in the Linux
  channel; the plan treats it as a first-class backend.
* **hipBLASLt exists for gfx1201 and not for gfx1031.** No RDNA2 design in this
  plan may depend on hipBLASLt.
* **The gfx1031 blas kpack carries prebuilt Tensile kernels** (44.5 MB) — the
  P0 microbenchmark baseline and the correctness oracle for hand-written RDNA2
  GEMM.
* The `torch`, `torchvision` and `rccl` kpacks are irrelevant to this engine.
  They are recorded here so nobody spends time trying to link them.
* ROCm's own docs note that release numbering has drifted between the Windows
  and Linux channels, so **nothing tests for a version string**. The engine
  probes `hipRuntimeGetVersion()` for logging only, and selects kernels from the
  `__gfx*__` macros of the compile pass. That existing decision is correct and
  stays.

---

## 2. Build layout

```
kanjoos/
  CMakeLists.txt
  cmake/
    arch.cmake            # per-target flags, ONE place
    rocm.cmake            # locate hipcc, set offload-archs
    sanitize.cmake        # ASAN/UBSAN build for the eviction stress tests
  src/
    core/      runtime, streams, events, budget        C1, C22
    platform/  windows/, linux/, rebar, pinned         C24
    device/    backend_gfx1201, backend_gfx1031        C2
    model/     gguf, loader, arch specs                 C4
    compiler/  expert quantise/pack/manifest            C5
    store/     expert directory, nvme, journal          C6
    host/      warm pool, pinned pool, staging          C7
    gpu/       slot pool, expert gemm                   C8, C17
    residency/ manager, predictor, eviction, admission  C9, C10, C12
    transfer/  io_uring, directstorage, iocp, engine    C11
    kv/        radix index, page manager, codecs        C13, C14, C15
    attn/      flash prefill/decode, paged, out-of-core C16
    spec/      mtp, dflash, verify                      C20
    profile/   scopes, ring, report                     C21
    server/    scheduler, http, sessions                C23
  kernels/               # .hip, one file per kernel, arch-#if-selected
  tools/isa_probe/       # capability probe, already present
  tests/
    unit/ integration/ bench/ stress/
  docs/
```

**The rule that keeps this from rotting:** every arch difference lives in
`src/device/`, `kernels/`, `src/platform/`, or `cmake/arch.cmake`. Nothing above
those four directories contains a `#ifdef __gfx*__` or `#ifdef _WIN32`.

---

## 3. Per-target flags

```cmake
# cmake/arch.cmake — the single source of truth
set(KNJ_ARCH_GFX1201_FLAGS
    --offload-arch=gfx1201
    -mno-wavefrontsize64              # WMMA is wave32-only
    -D__gfx1201__)                    # kernels select on this
set(KNJ_ARCH_GFX1031_FLAGS
    --offload-arch=gfx1031
    -mno-wavefrontsize64              # RDNA2 is wave32
    -D__gfx1031__)
```

Two flags above are load-bearing and both are verified:

* `-mno-wavefrontsize64`: **all WMMA builtins are wave32-only** (rocWMMA states
  this for both RDNA3 and RDNA4), and every wave reduction must be unrolled to
  exactly five shuffle steps, never six.
* `HSA_OVERRIDE_GFX_VERSION` is **not** set. Both targets are natively supported;
  forcing a version is how you get a card that enumerates and then miscomputes.

### 3.1 The WMMA gate: solved, and it was not the flag it claimed

The flag was never the problem. Measured:

| spelling | gfx1031 | gfx1201 |
|---|---|---|
| `..._f16_w32` (gfx11, A/B = 16×f16) | REFUSED | REFUSED, and **crashes the backend** if forced past the frontend |
| `..._f16_w32_gfx12` (A/B = 8×f16) | REFUSED | **OK** → `v_wmma_f32_16x16x16_f16` |

So gfx1201 builds its matrix path with:

```
-Xclang -target-feature -Xclang +wavefrontsize32
```

and no `wmma-256b-insts` anywhere. That feature cannot be enabled in this
build; stop looking for it.

The gfx11 `_w32` builtin is a trap worth writing down: with its 16×f16 operands
it passes the frontend check and then **segfaults in Branch relaxation**. It is
a compiler bug, not a capability signal, and it is the reason the probe runner
distinguishes `CRASH` from `REFUSED`.

`03-kernels.md` §1.1 has the exact source shape. The per-arch build flags that
result:

| target | extra flags |
|---|---|
| gfx1201 | `-Xclang -target-feature -Xclang +wavefrontsize32` |
| gfx1031 | none; it has no matrix units and never will |

### 3.2 Toolchain state on this machine

Do not build against paths that no longer exist:

| path | state |
|---|---|
| `G:\ROCM10RT-gfx1031` | present, `clang.exe` works, **no `amdgcn/bitcode`**, `bin/hipcc.exe` broken |
| `G:\ROCM10RT-gfx1201` | **deleted** |
| `G:\rocm-10` | **deleted** |

Consequences for the build script:

* Invoke `lib/llvm/bin/clang.exe` directly. `hipcc.exe` here is unusable — it
  hardcodes the clang path inside the deleted `ROCM10RT-gfx1201` tree and exits
  with an empty error, which is indistinguishable from a compiler error.
* Compile-only capability checks use `-nogpuinc -nogpulib --cuda-device-only`
  and therefore work today despite the missing device libs.
* A **real link needs a complete install**: restore `amdgcn/bitcode` or reinstall
  the gfx1201 runtime. This blocks Phase 1 execution testing, not Phase 1
  development.
* Only `gfx1031` kernel packs exist (`C:\Users\rr\OneDrive\Desktop\kpack`); there
  are no gfx1201 `.kpack` files on this machine.

---

## 4. Windows 11

* hipcc.bat must be on PATH; `hipcc.exe` is used directly by the build scripts
  on this machine.
* **DirectStorage preferred, IOCP / overlapped-I/O fallback.** DirectStorage is
  built for many small reads at low CPU cost and exposes CPU/GPU decompression
  options — but it does not make the SSD a VRAM extension. Staging into
  GPU-addressable memory and the Vulkan/ROCm side remain our job.
* `VirtualLock` on a pre-committed reserve for the pinned pool; the pool is
  bounded (C7).
* `FILE_FLAG_SEQUENTIAL_SCAN` and
  `SetFileInformationByHandle(FileInformationIfHint)` for read-ahead;
  `FSCTL_MANAGE_BANDWIDTH_THROTTLING`/`SetFileIoOverlappedRange` if profiling
  shows Windows read-ahead fighting our own.
* **gfx1031 on Windows is best-effort.** The Linux channel is release-ready;
  Windows ships the runtime and the packs but no hipBLASLt. The engine detects
  the platform+arch combination, prints it in `kanjoos doctor`, and does not
  pretend the two are equivalent.

## 5. Linux

* `io_uring` with `O_DIRECT` where it wins, `RWF_HIPRI` for demand reads,
  `fallocate`d cold-store files, pre-registered buffers, many requests in flight.
* `posix_fadvise(WILLNEED)` / `madvise(MADV_WILLNEED | MADV_HUGEPAGE)` at load
  and on eviction.
* `mlock` bounded by RLIMIT_MEMLOCK, checked at init so a missing capability
  fails loudly instead of mid-transfer.
* ReBAR via `/sys/class/drm/card*/device/resource`.
* The headless RDNA2 box (`ssh rr@10.0.0.12`, RX 6700 XT, 46 GB) is the
  reference target for the gfx1031 backend.

---

## 6. Kernel packs

```
G:\ROCM10RT-gfx1031\lib\kpack\   blas_lib_gfx1031.kpack  fft_lib_gfx1031.kpack  rand_lib_gfx1031.kpack
G:\ROCM10RT-gfx1201\lib\kpack\   blas_lib_gfx1201.kpack  fft_lib_gfx1201.kpack  rand_lib_gfx1201.kpack
```

* The **blas** pack is what makes `rocm-bench` and the Tensile kernel set
  available for the P0 baseline on gfx1031.
* The **fft** and **rand** packs are not used by the inference path. They are
  installed for completeness and are documented as unused, so nobody wastes a
  day wiring them in.
* `tools/kpack_report.md` (maintained with the repo) records what each pack
  provides and what the engine deliberately does not use.

---

## 7. CI

| job | matrix | gate |
|---|---|---|
| build | Windows × {gfx1201, gfx1031}, Linux × {gfx1201, gfx1031} | compiles, warnings-as-errors |
| isa-probe | all | matrix matches the pinned expectation in `00-verified-facts.md` |
| unit | all | host-reference equivalence, per format tolerance |
| integration | all | forward pass vs CPU reference; KV reload **bit-identical** |
| bench | all | no regression > 10% against the pinned baseline |
| stress | Linux, gfx1031 + gfx1201 | eviction stress under ASAN/UBSAN, no use-after-free |

The ISA-probe job is the unusual one and it earns its place: when a ROCm upgrade
changes the WMMA feature gate or the dot builtins, that job fails and says
exactly which capability moved, instead of the engine silently losing 3× on
expert GEMM after a driver update.

---

## 8. `kanjoos doctor`

One command, printed at the top of every profile. The block below is the
**format contract** — placeholder values, never produced by a run (the
attached 9070 XT is 32 CU, and `dot=builtin(sdot4)` is refused on gfx1201 —
the measured gfx1201 line is `dot=asm(v_dot4_i32_i8)`, see
`00-verified-facts.md` section 1):

```
device        AMD Radeon RX 9070 XT  gfx1201  32 CU  wave32
capabilities  wmma=on  fp8=on  dot=asm(v_dot4_i32_i8)  smi=lane-split  v_pk_fma_f16=yes
rebar         full aperture (16.0 GiB mapped)
vram          total 16.00 GiB   usable 15.31 GiB (W4 resident set)   ceiling 14.00 GiB   operating point: W3 g128 (11.79 GiB resident, 3.46 GiB KV budget)
ram           total 96.0 GiB    warm target 62.4 GiB   pinned pool 6.0 GiB (locked 5.8)
nvme          quota 1.4 TiB   used 812 GiB   measured seq read 4.1 GB/s   extent 4 MiB
queues        compute 1  copy 1  async 1   overlap enabled
backend       gfx1201 simt            (or: gfx1201 wmma f16/f8/i8, simt fallback)
tuning        cache hit, 37 entries, arch-keyed, invalidated on binary change
```

When the engine prints this, every line must be a measured fact, not a
summary. If `rebar` says `256 MiB aperture`
the run is going to be bad and the engine will say so before the first token
rather than after the first minute.