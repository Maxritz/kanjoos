# Kernel packs — what is present and what the engine uses

ROCm ships precompiled device kernels in `.kpack` containers (`KPAK` v1 magic,
version field, then a payload). They are installed per-target and are what make
`hipblas`/`rocblas` device kernels and PyTorch binaries work without building
ROCm from source.

This file records what each pack provides and what Kanjoos deliberately does not
use. It exists so nobody spends a day trying to link the wrong thing.

## Present on this machine

> `G:\ROCM10RT-gfx1201` has been **deleted**, so there are no gfx1201 packs
> anywhere on this machine. The gfx1031 rows below are the only ones that
> exist; the gfx1201 column in the earlier revision of this file was
> aspirational, not observed. On RDNA4 the oracle must come from a CPU
> reference until a gfx1201 install is restored.

| `C:\…\kpack\blas_lib_gfx1031.kpack` | 44.5 MB | copy of the above, loose | staging copy |
| `C:\…\kpack\fft_lib_gfx1031.kpack` | 2.9 KB | copy | no |
| `C:\…\kpack\rand_lib_gfx1031.kpack` | 3.1 MB | copy | no |
| `C:\…\kpack\rccl_lib_gfx1031.kpack` | 17.3 MB | RCCL collectives | **no** — single device |
| `C:\…\kpack\torch_gfx1031.kpack` | 44.6 MB | PyTorch device kernels | **no** — this is a C++/HIP engine, no PyTorch |
| `C:\…\kpack\torchvision_gfx1031.kpack` | 118 KB | torchvision kernels | **no** |

## Why blas is used and the rest is not

The **blas** pack is the one that matters, for a specific reason: it is the
measured reference implementation of what a tuned GEMM looks like on each target.
Kanjoos's own GEMMs must beat it on the profile it is tuned for (fused dequant +
grouped expert execution, which Tensile has no kernel for), and must match it
where the profile is the same. That makes the pack a correctness oracle as well
as a performance floor.

**fft** and **rand** have no consumer in an inference engine: there are no FFTs
and no random number generation in a forward pass. The sampler is a
multinomial over the vocabulary from a deterministic PRNG that lives in the
runtime, not in a device RNG library.

**rccl** is collective communication across multiple GPUs. This engine is
single-device by design; multi-GPU is a non-goal recorded in
`docs/01-architecture.md` §8.

**torch** and **torchvision** are for the PyTorch binary distribution. Kanjoos
does not embed or link PyTorch — it is a standalone C++/HIP engine, and
depending on the PyTorch runtime would drag in a dependency that ships its own
kernels for a graph API the engine does not use.

## What this implies for the build

* `lib/kpack` must be on the runtime search path (`ROCM_PATH`), otherwise
  `hipblas`/`rocblas` device kernels fail to load at runtime rather than at link
  time — a failure mode worth knowing about before debugging.
* `kanjoos doctor` should report pack presence alongside the capability table, so
  a missing pack is visible immediately rather than as a load failure in the
  middle of a benchmark.
* The presence of a blas pack does **not** imply hipBLASLt. gfx1201 ships
  hipBLASLt as a library; gfx1031 does not, pack or no pack. That asymmetry is
  recorded in `docs/00-verified-facts.md` §2 and drives the kernel plan.