# Worksheet — C4 Model loader

**What it owns:** turning a GGUF on disk into resident buffers and a cold index.

**Why it exists (from `02-components.md` — C4 section):** the loader must not read expert weights at load. It reads the
trunk and the *directory*. The expert weights stay on NVMe until the residency manager (C9) and transfer engine (C11) move
them, in kernel-native packed form (C5), into VRAM slots (C8) or the warm pool (C7).

**Spec lines that must hold:**
* `02-components.md` — C4 section (verbatim: mmap, honour `general.alignment` 32/64/128, sequential-read hints per platform,
  build a cold index at load: tensor directory, expert offsets grouped by layer into contiguous extents (C6), KV geometry;
  never read expert weights at load; done-when).
* `01-architecture.md` §3.1, §3.3 (VRAM ceiling, the allocation split, NVMe quota — the loader must produce a cold index that
  respects these, not one that assumes a different geometry).
* `04-memory-tiering.md` §4.1 (logical/physical split: residency decides per expert; I/O moves multi-expert extents. The loader's
  cold index is the "logical" side — it must be able to produce multi-expert extents for I/O, not one-extent-per-expert).

**Interface (from `02-components.md` C4, implied by the rest):**
* Input: a GGUF path (and the model fingerprint, arch, tokenizer, attention-format, RoPE config, KV codec, impl-version that
  become part of the cache key — I5). Output: resident trunk buffers (attention, router, head, embeddings) + a cold index
  (tensor directory + expert extents + KV geometry) that C5/C6/C9/C14 consume.
* The loader must honour `general.alignment` — assuming 32 silently misreads every offset (`02-components.md` C4 bullet, recorded
  in the project's RESEARCH.md). This is not a "should"; it is the reason the loader can read the file at all.

**Acceptance test (verbatim "done when"):**
* Load time for a 30B-class MoE is bounded by the trunk size, not the model size.
* A cold-start on a 4 GB/s disk stays under a stated number for a 40 GiB model.

**Gates:**
* The "stated number for a 40 GiB model on a 4 GB/s disk" is a future acceptance test, not yet measured. Until it is, the loader's
  load-time claim is a target, not a measured fact.
* The cold index's extent shape (multi-expert, layer-contiguous) must match C6's object store and C11's I/O shape (4 KiB aligned,
  sequential, coalesced, 64 KiB–8 MiB extents). A loader that produces one-extent-per-expert would force the 4 KiB random-read failure
  mode that the whole plan avoids.
* Depends on C5 (the packed expert store — the loader's cold index points into it), C6 (the object store — the cold index is the directory
  of it), C14 (KV geometry — the loader computes it from the config, never hard-codes it), C22 (the VRAM/RAM/NVMe budget the resident
  buffers must fit within).

**Coder notes / pitfalls:**
* Do not read expert weights at load. The loader reads the trunk (attention, router, head, embeddings) resident, and the *directory*
  (tensor offsets, expert extents, KV geometry) for the experts. The expert weights themselves are read later, by C11 from the C5/C6 store,
  asynchronously, into C8/C7.
* The alignment field matters. GGUF `general.alignment` can be 32/64/128; assuming 32 silently misreads every offset. Read the field and honour it.
* The cold index is the contract between load and runtime. If the loader's cold index is wrong (wrong offsets, wrong extent grouping, wrong KV geometry),
  the runtime will read wrong bytes — plausibly, with no error. The cold index must be checksummed/hash-verified against the source, because a wrong cold index
  is the same class of defect as the packer's byte-offset-as-weight-index bug (`00` §7.5, item 1).
* KV geometry is computed from the loaded config, never hard-coded (`09` §9.1: `bytes/token = 2 * layers * kv_heads * head_dim * sizeof(codec)`). The loader
  is where that computation lives for load time; C14/C15 use it at runtime. Do not hard-code 48/4/128 — read them from the config and refuse (exit 2/3/4) when
  missing, same as `kv_roofline.py`.
* The "cold-start on a 4 GB/s disk stays under a stated number for a 40 GiB model" gate means the loader must be able to report its load time and the bytes it read,
  so the number is measurable. Do not hide load time behind "it's fast enough". A 40 GiB model on a 4 GB/s disk is a 10-second floor on the raw read alone; the loader's
  job is to keep the *effective* cold-start cost to the trunk read + directory build, with the expert bytes not read at all. That is the claim to measure.

**Worked micro-example (host, sanity-check before device):**
Take a small GGUF-like file (or a fake one with a known tensor directory and a few expert sections), mmap it, honour the alignment field, build a cold index
(tensor directory + expert extents grouped by layer + KV geometry from a config), and assert: (1) the cold index's tensor offsets match the file's actual tensor locations
(modulo alignment); (2) the expert extents are multi-expert and layer-contiguous (not one-extent-per-expert); (3) the KV geometry matches the config's layers/kv_heads/head_dim
(so a 48/4/128 config gives the same bytes/token as `kv_roofline.py` section B); (4) no expert weight bytes were read into a resident buffer (only the directory was read);
(5) a wrong alignment assumption would produce a wrong offset (so the loader must read and honour the field, not assume 32).

**Files a coder should read before starting:**
* `docs/02-components.md` — C4 section.
* `docs/01-architecture.md` §3.1, §3.3.
* `docs/04-memory-tiering.md` §4.1 (logical/physical split).
* `docs/09-kv-engine-architecture.md` §9.1 (KV geometry from config — the loader computes it at load time).
* `tools/kvroof/kv_roofline.py` section B (the KV bytes/token numbers the loader's KV geometry must reproduce).
