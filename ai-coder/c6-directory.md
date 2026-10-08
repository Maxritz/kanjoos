# Worksheet — C6 Expert directory and NVMe object store

**What it owns:** the addressable unit that the residency manager moves.

**Why it exists (from `02-components.md` — C6 section):** the expert object is the logical unit (one expert); the physical I/O unit is a
coalesced multi-expert extent (a whole layer slab where the packing is contiguous). Residency decides per expert; I/O moves 2–32 MiB extents.
This is the split that avoids the 4 KiB random-read failure mode. The store is content-addressed, immutable, KVP-style, with checksums, an atomic
publish, and a crash-safe journal + manifest.

**Spec lines that must hold:**
* `02-components.md` — C6 section (verbatim: ExpertObject, the logical/physical split, content-addressed immutable KVP container, checksum on read,
  corrupt-object handling, done-when).
* `04-memory-tiering.md` §4.1 (logical/physical split: residency decides per expert; I/O moves multi-expert extents; fine-grained decisions, coarse-grained
  transfer; never a 4 KiB application-level random read of a payload).
* `08-roadmap.md` — Acceptance criteria: "a crash during an NVMe write leaves committed objects intact"; "a cache page from a different model/config/codec is rejected, not misread."

**Interface (verbatim `02-components.md` C6):**
```cpp
struct ExpertObject {
    uint32_t layer, expert;
    uint8_t  precision, format;
    uint16_t group_size;
    uint64_t nvme_offset, nvme_size;
    uint64_t packed_hash;
    float    quality_loss, route_loss;
};
```

**The logical/physical split (the one thing not to get wrong, verbatim `04` §4.1):** **residency decides per expert, I/O moves multi-expert extents.**
Fine-grained decisions, coarse-grained transfer. A 4 KiB random read of expert payload is the failure mode this avoids.

**Container (verbatim `02-components.md` C6):** content-addressed, immutable, `KVP`-style container: header, model fingerprint, quant descriptor, token/layer metadata,
checksum, payload index, payload. Aligned offsets, versioned format, atomic publish, crash-safe journal + manifest.

**Acceptance test (verbatim "done when"):**
* A crash during a write leaves committed objects intact and readable.
* A corrupt object is detected by checksum and replaced, with a telemetry counter that fires.

**Gates:**
* Depends on C5 (the packed store — C6 is the NVMe-resident manifestation of it; the manifest C5 writes is what C6 publishes), C11 (the transfer engine — I/O moves
  coalesced extents, not one-extent-per-expert), C9 (residency — per-expert decisions query C6 by expert id; C11 moves coalesced extents).
* The "crash during a write leaves committed objects intact" gate means the publish is atomic (journal + manifest, versioned format). A coder must implement the crash-safety, not
  assume the SSD's atomicity. The acceptance test is: simulate a crash mid-write (kill the process after the write starts but before the manifest is updated) and assert committed objects
  are intact and readable, and the partial object is not readable as a committed object.
* The "corrupt object detected by checksum and replaced" gate means every read verifies the checksum, and a mismatch triggers a re-read from the canonical checkpoint (C4/C5) and a
  quarantine, with a telemetry counter. A coder must not silently return a corrupt payload. The telemetry counter "fires" means it is instrumented and reachable in the profiler/telemetry
  (C21) — a corrupt-object event must be a visible counter, not a silent retry.

**Coder notes / pitfalls:**
* Do not implement "one extent per expert" on NVMe. That is the 4 KiB random-read failure mode. The physical I/O unit is a coalesced multi-expert extent (a whole layer slab where
  the packing is contiguous). The logical unit is one expert (C9 queries by expert id); the physical unit is the extent (C11 reads the extent that contains the expert). The mapping
  from expert → extent is what C6's directory provides.
* The container is content-addressed and immutable. A new pack (C5) produces new objects with new hashes; it does not overwrite old ones in place. "Atomic publish" means the new
  manifest is written and then made visible as a unit (journal + manifest update), so a crash mid-publish leaves the old manifest intact. A coder must not do an in-place overwrite of a
  committed object.
* Checksum on read, every read. Not on write only. A corrupt object on disk (bit rot, partial write) must be caught on read, quarantined, and re-read from the canonical checkpoint (C4/C5).
  Silent corruption is the worst failure here (the engine would compute wrong answers, plausibly, with no error — same class as the page-local-vs-global and byte-offset-as-weight-index bugs).
* The model fingerprint, quant descriptor, and format are part of the object's identity (I5: a cache page from a different build is not readable). So C6's object key must include enough to
  reject a wrong object — model fingerprint + quant descriptor + precision/format/group_size + packed_hash. A wrong object is not "a cache miss"; it is "this object is not the one for this
  build" and must be rejected, not misread.
* The `nvme_offset`/`nvme_size` in ExpertObject is the logical object's location in the store; the physical extent is the coalesced unit C11 reads. Do not confuse the two. The coder must
  maintain both: the per-expert logical location (for residency decisions) and the coalesced physical extent (for I/O).

**Worked micro-example (host, sanity-check before device):**
Build a fake NVMe store on disk (a file, or a directory of files, with a journal and a manifest). Write a few ExpertObjects (pack them into coalesced multi-expert extents, record the
logical per-expert locations and the physical extents). Then: (1) read back each object by expert id, verify the checksum matches, and assert the payload matches what was written; (2) simulate a
  crash mid-write of a new object (write part of the payload, then kill before the manifest/journal update) — assert the old manifest is intact, the partial object is not readable as committed, and
  committed objects are unchanged; (3) corrupt a byte in a committed object on disk, read it back, assert the checksum mismatch triggers a quarantine + telemetry counter and a re-read from the
  canonical source; (4) write an object with a different model fingerprint/quant descriptor and assert it is rejected as "not the one for this build" rather than misread; (5) confirm the physical
  I/O unit is a coalesced multi-expert extent (one read brings in several experts), not one read per expert.

**Files a coder should read before starting:**
* `docs/02-components.md` — C6 section.
* `docs/04-memory-tiering.md` §4.1 (logical/physical split).
* `docs/08-roadmap.md` — Acceptance criteria (crash-safe write, reject-wrong-page).
* `docs/09-kv-engine-architecture.md` §5 (I5 — cache identity includes model fingerprint, arch, dtype, layout, RoPE, KV codec, impl-version).
* `ai-coder/c5-compiler.md` (the manifest C5 writes — C6 publishes it).
* `ai-coder/c11-transfer.md` (the I/O that moves coalesced extents — C6's directory maps expert → extent).
* `ai-coder/c9-residency.md` (the per-expert residency decisions that query C6 by expert id).
