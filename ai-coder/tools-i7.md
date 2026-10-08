# Worksheet — tools/i7 (existing, bit-identity contract and GPU proof)

**What it is:** `tools/i7/i7_bit_identity.py` (the host reference — the contract) + `tools/i7/i7_hip.hip` (the
device proof) + `tools/i7/run_i7.sh` (the runner). Written before any codec exists, which is the only time it is
worth writing.

**The contract (`00` §9.7, `09` §3, I7):** three paths must agree **bitwise**: resident (A), lossless-codec round
trip through host RAM (B: movement codec, D2H then H2D over PCIe), and a hot/cold tier split (C: some pages in VRAM,
some streamed from pinned host memory). Three negative controls must **fail**: the flat single-accumulator form, the
cold-first (descending) merge, and a lossy codec.

**Two findings that changed the design (verbatim, `00` §9.7):**
1. **I7 as originally worded is unsatisfiable.** "Any codec must reproduce the same bits" is false of every quantising
   codec, by construction. Re-scoped: **tier movement** must be bit-identical and its storage codec must be lossless; a
   codec that changes values is a **declared precision reduction**, priced and recorded per page, and never conflated with
   a move.
2. **Bit-identity requires a canonical reduction order.** FP addition is not associative, so per-page partials merged
   left-to-right and a single running accumulator produce different bits from identical inputs. The contract **is** the page
   decomposition: ascending pages, ascending keys within a page, one left-to-right merge, global fp32 max first.

**What the negative controls caught (verbatim, `00` §9.7):** one of the negative controls fired on **this repo's own
tier-split implementation** — the hot path indexed the full V array with a page-local offset, producing plausible, wrong
output that no tolerance check would have flagged. That is the argument for writing the test first.

**GPU proof — run this turn, exit 0, on the real gfx1201 (`00` §9.7.1, and run this turn):**
* device: gfx1201, wave32, BLK=64; payload bfloat16 (the model's `torch_dtype`), widened to fp32 by a 16-bit shift;
  softmax `__expf` = native `v_exp_f32`, no device libm (`-nogpulib`).
* **B vs A movement codec, D2H then H2D over PCIe — bitwise identical.**
* **C vs A tier split, 8 hot pages in VRAM, 8 cold pages streamed from pinned RAM — bitwise identical.**
* negative control: descending (cold-first) merge — **DETECTED**, 1 ulp apart (`0xbc2b785d` vs `0xbc2b785c`).
* path A vs double-precision host oracle — worst relative error **1.1e-05**.
* arch guard: mismatched arch (gfx1031) → exit 6; unguarded → exit 7; real dispatch (gfx1201) → exit 0.

**Two results worth stating precisely (verbatim, `00` §9.7.1), because conflating them is how bit-identity tests get faked:**
1. **The identity claim is GPU-vs-GPU**, between three paths on one device. It is structural: all three paths call the same
   `reduce_pages()` routine and the device executes identical instructions on identical bits.
2. **The correctness claim is GPU-vs-CPU**, and it is *within tolerance*, not bitwise. `__expf` is the native quarter-rate
   instruction and is not required to be correctly rounded, so cross-platform bit-identity against libm `exp` is neither
   asserted nor achievable. Asserting it anyway would have "passed" only by coincidence.

**The 1-ulp negative-control line is the most useful in the output (verbatim):** the wrong merge order is *almost* right,
and no tolerance check would ever have flagged it.

**Three bugs caught by writing this, all silent-wrong-answer class (verbatim, `00` §9.7.1):**
* the staging buffer holds one page but the kernel indexed it with the **global** key index — an out-of-bounds device read
  of up to 8× past a 16 KB buffer, and the same page-local-vs-global mistake the host reference had already caught in this
  repo's own tier-split path;
* `(float)(uint32_t)x` is a *conversion*, not a bit reinterpretation, so bf16 widening returned `0xBE450000` as the integer
  3,192,193,024 instead of the float −0.0332 it encodes. The host oracle and the GPU **agreed on the nonsense** — the
  dangerous part: a single-sided bug would have looked like a hardware problem;
* `hipMemcpy(&pointer, ...)` writes 4 bytes *into the pointer*, and the following dereference faults.

**What a coder must not fake or weaken:**
* Do not collapse the three positive paths into "the kernel produces the right answer within tolerance". The contract is
  **bitwise between the three paths on device** (B vs A, C vs A), plus a separate **within-tolerance GPU-vs-CPU correctness**
  claim. Those are different claims and must be reported separately.
* Do not weaken the canonical reduction order. Ascending pages, ascending keys within a page, one left-to-right merge, global
  fp32 max first — that is the contract. A single running accumulator, a different page order, or a different key order produces
  different bits (the descending-merge control proved it at 1 ulp). Any "optimization" that reorders the reduction must be
  justified against the contract, not against a tolerance check.
* Do not drop the negative controls. The descending-merge control fired on this repo's own tier-split implementation and would
  have been invisible to a tolerance check. A bit-identity test without a negative control that is known to fail is not a test.
* Do not conflate "bitwise identical between paths on device" with "matches the CPU oracle to 1e-5". The oracle agreement is
  within tolerance because `__expf` is not correctly rounded; the identity claim is bitwise because the three paths call the same
  `reduce_pages()`. These are different things and the 1-ulp control is the reason the distinction matters.
* Do not drop the page-local-vs-global discipline. The staging buffer holds one page; the kernel must index it with the page-local
  key index, not the global key index. The global-index bug was an out-of-bounds device read of up to 8× past a 16 KB buffer and
  the same mistake the host reference had already caught. This is the second time this repo has hit the page-local-vs-global trap
  (the host reference caught it first in the tier-split path, then the device test caught it again).

**Files a coder should read before touching this:**
* `tools/i7/i7_bit_identity.py` (the contract and the three positive paths + three negative controls).
* `tools/i7/i7_hip.hip` (the device proof — the `reduce_pages()` routine, the staging buffer discipline, the arch guard, the
  negative controls).
* `tools/i7/run_i7.sh` (the runner — the arch-guard proof step, the exit-code semantics).
* `docs/00-verified-facts.md` §9.7 and §9.7.1 (the contract, the two findings, the GPU proof, the three bugs).
* `docs/09-kv-engine-architecture.md` §5 (the I7 contract restated in the KV engine doc — tier movement bit-identical, storage
  codec lossless, a codec that changes values is a declared precision reduction).
