# Worksheet — C7 Host tier: warm pool + bounded pinned DMA staging

**What it owns:** RAM. This is the component the performance argument rests on, and it is also the one that can take the machine down.

**Why it exists (from `02-components.md` — C7 section; `04` §3):** the warm pool is pageable, huge-page-hinted, 2 MiB aligned; the pinned pool is bounded,
page-locked, DMA-only, with chunks of 2–32 MiB. The whole machine is never pinned. The pinned pool caps are 2 GiB (24/32 GiB hosts), 3 GiB (48), 6 GiB (96).
Windows: `VirtualLock` on a pre-committed reserve. Linux: `mlock` + `MAP_POPULATE`, with an RLIMIT check up front so a missing capability fails loudly at init,
not mid-transfer. Chunk size is tuned per host (P0-1 measures the host copy path; `01` §3.2 records ~19 GB/s cached file reads).

**Spec lines that must hold:**
* `02-components.md` — C7 section (verbatim: WarmPool / PinnedPool API, the rules, the failure mode, done-when).
* `04-memory-tiering.md` §3 (RAM profiles: WARM_KV_LIMIT, PINNED_POOL caps, the table, dynamic shrink, "the host RAM tier is not a cache for this model on this card — it is
  where the other weights have to live"; pinning stays bounded).
* `01-architecture.md` §3.2 (RAM budget arithmetic: WARM_KV_LIMIT = min(0.65 × phys_ram, phys_ram − 8 GiB); PINNED_POOL caps; dynamic shrink order:
  PINNED > active request's KV > active request's experts > WARM > COLD).
* `01-architecture.md` §7 — gfx1031 tier emphasis: RAM-heavy, cold-tolerant; gfx1201: VRAM-hot-heavy, aggressive prefetch.

**Interface (verbatim `02-components.md` C7):**
```cpp
class WarmPool {                 // pageable, huge-page-hinted, 2 MiB aligned
    void*  allocate(size_t bytes, PoolTag tag);
    void   release(void* p);
    size_t used() const; size_t available() const;
    void   shrink_to(size_t bytes);         // under memory pressure
};
class PinnedPool {               // bounded, page-locked, DMA-only
    Chunk  acquire(size_t bytes);           // 2–32 MiB chunks
    void   release(Chunk);
    size_t capacity_bytes() const;
};
```

**Rules (verbatim):**
* **The whole machine is never pinned.** Pinned pool caps are 2 GiB (24/32 GiB hosts), 3 GiB (48), 6 GiB (96). Everything else is pageable and copied through a pinned bounce buffer
  in large aligned chunks.
* Chunk size is tuned per host (P0-1 measures the host copy path; the workspace note already records ~19 GB/s cached file reads).
* Windows: `VirtualLock` on a pre-committed reserve. Linux: `mlock` + `MAP_POPULATE`, with an RLIMIT check up front so a missing capability fails loudly at init rather than mid-transfer.

**Fails by (verbatim):** shrinking under pressure in the order given in §3.2 of `01-architecture.md`.

**Acceptance test (verbatim "done when"):**
* RSS stays inside the declared budget under a 96 GiB-profile synthetic load.
* A transfer loop pinned-bursts without ever exceeding the pinned cap.

**Gates:**
* P0-1 (platform census — `00` §5, ASSUMPTION) measures the host copy path. Until it is measured on the real box, the "~19 GB/s cached file reads" and the chunk-size defaults
  (2–32 MiB, tuned per host) are starting points, not measured facts. C11 (transfer engine) depends on the chunk size; C7 must be able to report its chunk size and the copy bandwidth it
  achieves, so P0-1 can tune it.
* Depends on C11 (transfer engine — the pinned bounce buffer is where C11's async H2D reads from/writes to; the warm pool is where C11's NVMe reads land before H2D), C9 (residency —
  the warm pool is where NVMe-resident and RAM-resident objects live), C4 (loader — the warm pool is where the trunk and the warm experts live), C22 (budget — the pinned pool cap and
  WARM_KV_LIMIT come from C22's profile).
* The "whole machine is never pinned" rule is a hard bound, not a guideline. A coder must enforce the pinned-pool cap and must not pin more than the cap under any load. The acceptance test
  ("pinned-bursts without ever exceeding the pinned cap") is the proof.

**Coder notes / pitfalls:**
* The pinned pool is bounded and DMA-only. It is not a general-purpose pinned allocation. A coder must not let the engine pin arbitrary buffers — only the DMA bounce buffers that C11 consumes.
  The cap (2/3/6 GiB by host RAM) is a hard bound; exceeding it fails loudly at init (RLIMIT check / `VirtualLock` failure), not mid-transfer.
* The warm pool is pageable and huge-page-hinted (2 MiB aligned). It is where the trunk, the warm experts, and the warm KV live. It can be shrunk under pressure (in the §3.2 order), and it
  is the tier that the OS can swap if the engine does not shrink it first. So the engine must poll platform-available memory each scheduling tick and shrink WARM before the OS swaps — dynamic
  shrink is mandatory, not optional (`04` §3: "Dynamic shrink is not optional.").
* The "~19 GB/s cached file reads" note (`01` §3.2) is a recorded observation, not a guarantee. A coder must not hardcode it as the copy bandwidth. C7 must report the copy bandwidth it achieves
  (via C11/C21), so P0-1 can tune the chunk size and the engine can know its real host-copy ceiling.
* Huge pages: `madvise(MADV_HUGEPAGE)` on Linux for the warm pool — expert pages are read once, so THP is nearly free. On Windows, the equivalent is `VirtualAlloc` with `MEM_LARGE_PAGES`
  where available. A coder must not assume huge pages are available and must degrade gracefully (fall back to 2 MiB aligned pageable) if they are not.
* The warm pool is not a cache for this model on this card at 16 GiB (`04` §3: "The host RAM tier is therefore not a *cache* for this model on this card. It is where the other weights have
  to live."). At 16 GiB with W4, the expert bank is 14.13 GiB and does not fit in VRAM, so the other experts live in RAM (warm) and NVMe (cold). At 96 GiB host, the whole W4 bank (14.13 GiB)
  fits warm, and so does W6 (20.7 GiB) with room to spare — that is the configuration where the engine is genuinely excellent (`04` §3). So the warm pool's role differs by host RAM and by pack:
  at 96 GiB it is a full resident tier; at 16 GiB it is the overflow tier for the experts that do not fit in VRAM. A coder must not implement the warm pool as a small "cache" — its size is set by
  C22's profile (WARM_KV_LIMIT, warm experts typical), and at 96 GiB it is ~70 GiB.

**Worked micro-example (host, sanity-check before device):**
Implement a fake WarmPool (pageable, 2 MiB aligned, huge-page-hintable) and a fake PinnedPool (bounded, page-locked, DMA-only, with a cap). Then: (1) allocate across both pools under a
  synthetic 96 GiB-profile load and assert RSS stays inside the declared budget (WARM_KV_LIMIT + warm experts + pinned cap) — the "RSS inside declared budget" gate; (2) run a transfer loop that
  pinned-bursts (acquire many 2–32 MiB chunks, use them for DMA bounce, release) and assert the pinned pool never exceeds its cap — the "pinned-bursts without exceeding cap" gate; (3) simulate
  memory pressure (reduce platform-available memory) and assert the warm pool shrinks in the §3.2 order (PINNED > active KV > active experts > WARM > COLD) — the dynamic-shrink gate; (4) assert
  that a request for pinned memory above the cap fails loudly at init (RLIMIT / `VirtualLock` failure), not mid-transfer; (5) report the copy bandwidth the transfer loop achieves (C11/C21 consume
  this) so P0-1 can tune the chunk size — and assert it is reported, not assumed.

**Files a coder should read before starting:**
* `docs/02-components.md` — C7 section.
* `docs/04-memory-tiering.md` §3 (RAM profiles, pinned pool caps, dynamic shrink, "not a cache" note).
* `docs/01-architecture.md` §3.2 (RAM budget arithmetic, shrink order), §7 (tier emphasis per arch).
* `docs/00-verified-facts.md` §5 (P0-1, ASSUMPTION — host copy path not yet measured on the real box).
* `ai-coder/c11-transfer.md` (the transfer engine that consumes the pinned bounce buffer and lands NVMe reads into the warm pool).
* `ai-coder/c22-budget.md` (the profile that sets WARM_KV_LIMIT, warm experts, pinned pool cap — C7's sizes come from here).
* `ai-coder/c24-platform.md` (the platform layer — Windows `VirtualLock` / `FILE_FLAG_SEQUENTIAL_SCAN`, Linux `mlock`/`MAP_POPULATE`/`MADV_HUGEPAGE`, the RLIMIT check).
