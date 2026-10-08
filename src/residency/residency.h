// src/residency/residency.h -- C9 Residency manager: where every expert and KV page is.
//
// This is the component the project exists for. Everything else -- the kernels, the
// loader, the tokenizer -- is a subset of llama.cpp until an expert can live on NVMe
// and still be executed. C9 is the table that makes that possible, and its whole
// design is one idea:
//
//   A kernel never asks where something is. It asks `ready()`.
//
// Three rules (docs/02-components.md C9; docs/01-architecture.md section 6) are
// enforced mechanically here rather than by convention:
//
//  1. `ready()` is the only question a kernel may ask. Kernels are handed a
//     `KernelView` whose only methods are `ready()` and `ready_kv()`. It has no
//     `locate()`, no `request()` and no clock, so a kernel cannot learn where an
//     object is or when it will arrive -- which is what makes it impossible to wait
//     on a slow tier even by accident. The ordering that matters (demand enqueued ->
//     object ready, or scheduler fails over) happens before the kernel runs.
//  2. Nothing on the demand path blocks. `request()` only *enqueues* work through C11
//     and records intent; `demand()` reports whether the object is usable now; and
//     `demand_or_fallback()` is the scheduler's move when it is not -- hand the
//     contribution to C19. A miss is never waited for, and a read from RAM is never
//     allowed to turn into a stall for NVMe: it escalates and still returns not
//     ready. Issued is not ready; completed is ready, and C8's load event is the
//     proof (a load with no event behind it is a guess).
//  3. Every transition is observable. Each one emits a C21 event carrying from, to,
//     cause, tick and timestamp. Those events are how C21 computes transfer time,
//     idle time and stall time, and how the acceptance test proves I4 ("zero
//     synchronous NVMe reads in the attention path") from a trace rather than from a
//     promise.
//
// Expert residency and KV residency are separate sub-tables of the same state
// machine. They are separate because their economics are opposite (AGENTS.md
// section 1): expert bytes are a fixed bank with exploitable locality, KV is
// unbounded and misses every token forever. They differ in behaviour in exactly one
// place -- KV bytes are mutable, so demoting a KV page off RAM has to write it back,
// while demoting an expert copy just drops staging, because the file is still
// authoritative. That is `mutable_bytes`, and it is the only asymmetry.
//
// The state table
// ---------------
// The manager is a cache table in front of a backing store. `register_expert()`
// records the object's extent in that store, and its initial state is
// NVME_RESIDENT by construction, because the bytes are in the model file. ABSENT
// therefore means "this manager holds no record of this id" -- what `locate()`
// returns for an unregistered id -- and INVALID means "a record whose cached bytes
// are suspect", which is what a failed read produces and a re-read clears.
// docs/02-components.md lists the eleven states but does not say which of ABSENT /
// NVME_RESIDENT is the pre-load state; the mapping above is the one that keeps I1
// ("every expert is recoverable from RAM or NVMe") literally true, and it is
// recorded in docs/CODING-LOG.md as an interpretation rather than a citation.
//
// Loading runs in two hops, because C11's NVMe reads land in *host* memory and the
// GPU copy is a second transfer:
//
//   NVME_RESIDENT -> LOADING_NVME -> RAM_RESIDENT -> LOADING_RAM -> VRAM_RESIDENT
//
// The names say where the bytes are coming from, so LOADING_RAM is the H2D hop. A
// demand that finds the object only on NVMe takes the whole path; a demand that
// finds it in RAM takes the second hop. Either way the first call returns
// "not ready", which is the entire point.
//
// HOST C++17. No HIP header and no device code: the gates C9 drives -- C8 slots,
// C11 transfers, C19 fallback, C21 telemetry -- are injected interfaces, so C9 runs
// and is graded against fakes on the host today and against the real HIP
// implementations later without a line changing here.

#ifndef KNJ_RESIDENCY_H
#define KNJ_RESIDENCY_H

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace knj {

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

struct ExpertId {
  uint32_t layer = 0;
  uint32_t index = 0;  // expert ordinal within the layer
  uint64_t key() const { return ((uint64_t)layer << 32) | (uint64_t)index; }
  bool operator==(const ExpertId& o) const { return key() == o.key(); }
};

struct KvPageId {
  uint32_t layer = 0;
  uint32_t page = 0;
  uint64_t key() const { return ((uint64_t)layer << 32) | (uint64_t)page; }
  bool operator==(const KvPageId& o) const { return key() == o.key(); }
};

// Where the object's bytes live when they are not cached: an extent in the model
// file (expert) or in the KV spill file (page). C11 never sees a smaller unit than
// this -- the residency decision is fine-grained, the I/O decision is not
// (AGENTS.md rule 13).
struct BackingExtent {
  uint64_t offset = 0;  // bytes from the start of the backing store
  uint64_t bytes = 0;
};

// ---------------------------------------------------------------------------
// The eleven states (docs/02-components.md, verbatim order)
// ---------------------------------------------------------------------------

enum class Residency : uint8_t {
  ABSENT = 0,     // no record for this id
  NVME_RESIDENT,  // bytes are in the backing store only
  LOADING_NVME,   // an NVMe read into host staging is in flight
  RAM_RESIDENT,   // a host staging copy exists
  LOADING_RAM,    // an H2D from staging into a slot is in flight
  VRAM_RESIDENT,  // demand copy in a slot
  PREFETCHED,     // speculative copy in a slot, not yet consumed
  EVICTING_VRAM,  // slot retirement in flight
  EVICTING_RAM,   // staging drop (KV: write-back) in flight
  PINNED,         // held in a slot, never an eviction candidate
  INVALID         // cached bytes suspect; must be re-read
};

const char* to_string(Residency r);
// Reachable in VRAM right now: VRAM_RESIDENT, PREFETCHED or PINNED.
bool is_vram_state(Residency r);

enum class KVTier : uint8_t { NVME = 0, RAM = 1, VRAM = 2 };
const char* to_string(KVTier t);
// The tier one step up, or `t` when already at the top. `promote()` moves by one.
KVTier next_tier_above(KVTier t);

enum class PrefetchPriority : uint8_t { BACKGROUND = 0, NORMAL = 1, URGENT = 2 };

// ---------------------------------------------------------------------------
// The injected gates
// ---------------------------------------------------------------------------

// C8 -- slot allocator. C9 sizes a slot and passes the handle around; the slot's
// lifetime is bounded by a HIP event rather than by wall clock, which is why
// "ready" is an event query and not a timestamp comparison.
using SlotHandle = int32_t;
constexpr SlotHandle kNoSlot = -1;

class SlotAllocator {
 public:
  virtual ~SlotAllocator() = default;

  // Reserve a slot for a resident object. `kNoSlot` means "no room right now" --
  // C9 must then leave the bytes where they are and report not-ready, never block
  // and never evict something in use. Slot count is C22's decision, not C9's.
  virtual SlotHandle acquire(uint32_t layer, uint32_t index, uint64_t byte_size) = 0;
  // Start retiring the slot. The release completes only once every transfer that
  // referenced it has finished -- C8 owns that bookkeeping, C9 only observes it.
  virtual void release(SlotHandle slot) = 0;
  virtual bool released(SlotHandle slot) const = 0;
  virtual void pin(SlotHandle slot) = 0;
  virtual void unpin(SlotHandle slot) = 0;
  // C8's load event (a HIP event) has fired: the copy in the slot is complete.
  // This is what keeps `ready()` honest, and it is why `ready()` is a query to C8
  // rather than a flag C9 keeps for itself.
  virtual bool load_event_fired(SlotHandle slot) const = 0;
  // Device pointer for the slot, or null. The tests read it to prove that a reload
  // reproduces the same bytes.
  virtual void* vram_ptr(SlotHandle slot) const = 0;
  virtual uint64_t byte_size(SlotHandle slot) const = 0;
};

// C11 -- transfer engine. Every call is asynchronous and returns a handle. There is
// deliberately no blocking variant in this interface: an interface with a blocking
// read in it gets a blocking read added to it.
using TransferId = uint64_t;
constexpr TransferId kNoTransfer = 0;

enum class TransferState : uint8_t { PENDING = 0, DONE, FAILED, CANCELLED };

// Demand traffic has a deadline (the step needs it now); prefetch traffic does not,
// and must never starve demand (C11 acceptance).
enum class IoClass : uint8_t { DEMAND = 0, PREFETCH = 1 };

class TransferEngine {
 public:
  virtual ~TransferEngine() = default;

  // Reads beat speculative writes, always (C11): the write submissions exist for
  // KV spill, which is capacity rather than scheduling.
  virtual TransferId submit_nvme_read(void* dst_host, uint64_t src_offset, uint64_t bytes,
                                      IoClass io) = 0;
  virtual TransferId submit_nvme_write(const void* src_host, uint64_t dst_offset,
                                       uint64_t bytes, IoClass io) = 0;
  virtual TransferId submit_h2d(void* dst_device, const void* src_host, uint64_t bytes,
                                IoClass io) = 0;
  virtual TransferId submit_d2h(void* dst_host, const void* src_device, uint64_t bytes,
                                IoClass io) = 0;
  virtual TransferState poll(TransferId id) const = 0;
  virtual bool cancel(TransferId id) = 0;
};

// The RAM tier's staging extents. In the real build these are pre-registered pinned
// buffers (C11 needs them registrable, not freshly malloc'd per read); the interface
// exists so that C9 does not decide the allocation policy and so the fake can hand
// the test pointers it can inspect.
class HostArena {
 public:
  virtual ~HostArena() = default;
  virtual void* alloc(uint64_t bytes) = 0;
  virtual void free(void* p) = 0;
};

// C19 -- the escape hatch that replaces a stall. Not optional and not a skip: I1
// says a routed contribution still has to be computed, just not here.
class FallbackSink {
 public:
  virtual ~FallbackSink() = default;
  virtual void expert_unavailable(ExpertId id, Residency observed) = 0;
  virtual void kv_unavailable(KvPageId id, Residency observed) = 0;
};

// C21 -- telemetry. Every transition, with no exceptions and no sampling.
struct TransitionEvent {
  bool is_kv = false;
  ExpertId expert;
  KvPageId kv;
  Residency from = Residency::ABSENT;
  Residency to = Residency::ABSENT;
  // A short stable token, not a sentence: "demand_miss", "reread", "prefetch_miss",
  // "nvme_read_done", "h2d_issued", "event_fired", "h2d_failed", "read_failed",
  // "evict", "evict_done", "writeback_done", "writeback_failed", "pin", "unpin",
  // "consumed", "prefetch_landed", "prefetch_cancelled", "reset",
  // "deadline_missed".
  // "deadline_missed" is the one token that is *not* a transition: it is the C21
  // miss record, and it arrives through on_deadline_missed with from == to.
  const char* cause = "";
  uint32_t layer = 0;
  uint64_t tick = 0;
  uint64_t now_ns = 0;
};

class TelemetrySink {
 public:
  virtual ~TelemetrySink() = default;
  virtual void on_transition(const TransitionEvent& ev) = 0;
  // A demand whose deadline passed without the object becoming ready. C9 has not
  // blocked -- the scheduler is already taking the C19 path for this object -- but
  // C21 still needs the miss on the record, because it is the numerator of
  // KV_MISS_STALL_TIME / TOTAL_DECODE_TIME. A deadline miss is telemetry, not a
  // stall, and that distinction is the whole acceptance criterion.
  virtual void on_deadline_missed(const TransitionEvent& ev, uint64_t ticks_late) = 0;
};

// ---------------------------------------------------------------------------
// C9
// ---------------------------------------------------------------------------

class ResidencyManager {
 public:
  struct Config {
    SlotAllocator* slots = nullptr;      // C8
    TransferEngine* transfer = nullptr;  // C11
    HostArena* arena = nullptr;          // RAM staging
    FallbackSink* fallback = nullptr;    // C19
    TelemetrySink* telemetry = nullptr;  // C21
    // Wall clock, for C21 timestamps only. Nothing in C9's control flow reads it:
    // deadlines are counted in ticks, so a slow or fake clock cannot change an
    // outcome.
    uint64_t (*now_ns)() = nullptr;
    // How many ticks a demand may go unserved before C21 records a miss. It is not
    // a timeout: nothing is cancelled and nothing is waited on, and the answer is
    // still "not ready" after the miss has been recorded.
    int64_t demand_deadline_ticks = 1;
    // How many ticks an unconsumed PREFETCHED copy stays a candidate before C12 is
    // invited to demote it.
    int64_t prefetch_ttl_ticks = 4;
  };

  explicit ResidencyManager(const Config& cfg);

  // -- registration: where the bytes are when they are not cached -------------
  void register_expert(ExpertId id, const BackingExtent& extent);
  void register_kv_page(KvPageId id, const BackingExtent& extent);
  std::size_t n_experts() const { return experts_.size(); }
  std::size_t n_kv_pages() const { return kv_.size(); }
  bool known(ExpertId id) const { return experts_.count(id.key()) != 0; }
  bool known_kv(KvPageId id) const { return kv_.count(id.key()) != 0; }

  // -- the C9 API, verbatim (docs/02-components.md) --------------------------
  Residency locate(ExpertId id) const;
  void request(ExpertId id, KVTier tier);  // demand
  void prefetch(ExpertId id, KVTier tier, PrefetchPriority prio);
  void promote(ExpertId id);
  void demote(ExpertId id);
  void tick();  // called once per layer
  bool ready(ExpertId id) const;
  // Extension to the verbatim seven: PINNED is one of the eleven states, so there
  // has to be a way into it. Pinning refuses eviction instead of ignoring it.
  void pin(ExpertId id);
  void unpin(ExpertId id);

  // -- the KV sub-table: same machine, separate table -----------------------
  Residency locate_kv(KvPageId id) const;
  void request_kv(KvPageId id, KVTier tier);
  void prefetch_kv(KvPageId id, KVTier tier, PrefetchPriority prio);
  void promote_kv(KvPageId id);
  void demote_kv(KvPageId id);
  bool ready_kv(KvPageId id) const;

  // -- the per-layer advance ------------------------------------------------
  // tick() is the synchronization point between the step loop and the tiers: the
  // prefetch horizon, the eviction scoring and the completion of every in-flight
  // transfer are advanced there, once per layer, and not by a background thread
  // that would race the step. begin_layer() records which layer that tick belongs
  // to and must be called before it.
  void begin_layer(uint32_t layer);
  uint32_t layer() const { return layer_; }
  uint64_t tick_index() const { return tick_; }

  // -- the scheduler's side -------------------------------------------------
  // Demand for a routed contribution. Never blocks: it enqueues what it can and
  // reports whether the result is usable *now*.
  bool demand(ExpertId id, KVTier tier);
  // The only call shape that satisfies I1 without a stall. When the object is not
  // ready it hands the contribution to C19 and reports false, because the kernel is
  // about to run and the contribution still has to be computed somewhere.
  bool demand_or_fallback(ExpertId id);
  bool demand_kv(KvPageId id, KVTier tier);
  bool demand_kv_or_fallback(KvPageId id);

  // -- C12's input ----------------------------------------------------------
  // tick() scores the candidates; C12 decides, and C12's lever is `demote()`.
  const std::vector<ExpertId>& eviction_candidates() const { return evict_candidates_; }
  const std::vector<KvPageId>& kv_eviction_candidates() const { return kv_evict_candidates_; }

  // A fresh run: every object returns to NVME_RESIDENT and every cached copy is
  // dropped. In-flight work is cancelled first -- nothing may outlive the run.
  void reset();

  // -- counters for the acceptance test -------------------------------------
  uint64_t transitions() const { return transitions_; }
  uint64_t deadline_misses() const { return deadline_misses_; }
  uint64_t preempted_prefetches() const { return preempted_; }
  uint64_t refused_demotions() const { return refused_demotions_; }
  // The tick() work probe (Phase 36 trap): entry_visits is the sum over ticks of
  // entries actually advanced; the naive loop visits 2 * N * ticks. A dirty-set
  // advance must show visits << that while every observable outcome — state
  // machine transitions, misses, preemptions, eviction candidates — is unchanged.
  uint64_t tick_calls() const { return tick_calls_; }
  uint64_t entry_visits() const { return entry_visits_; }
  uint64_t tick_ns() const { return tick_ns_; }
  // Invariant I4 stated as a fact rather than as a counter. There is no blocking
  // read in `TransferEngine` to call, so there is nothing to count, and a counter
  // would only prove that whoever added the blocking path also remembered to
  // increment. The probe checks the absence structurally instead.
  static constexpr uint64_t blocking_transfer_calls() { return 0; }

  // -- what a kernel is allowed to see --------------------------------------
  // A kernel is handed one of these and nothing else. `ready()` is the only
  // question a kernel may ask (docs/01-architecture.md section 6), so it cannot
  // learn where an object is or when it will arrive -- and therefore cannot wait on
  // a slow tier, even by accident. The absence of `locate()` and `request()` here is
  // enforced at compile time by the C9 probe.
  class KernelView {
   public:
    explicit KernelView(const ResidencyManager* m) : m_(m) {}
    bool ready(ExpertId id) const { return m_->ready(id); }
    bool ready_kv(KvPageId id) const { return m_->ready_kv(id); }

   private:
    const ResidencyManager* m_;
  };
  KernelView kernel_view() const { return KernelView(this); }

 private:
  struct Entry {
    bool is_kv = false;
    bool mutable_bytes = false;  // KV: demoting off RAM must write back
    ExpertId id;
    KvPageId kv;
    uint32_t layer = 0;
    BackingExtent extent;
    Residency state = Residency::NVME_RESIDENT;
    KVTier want = KVTier::NVME;  // highest tier any caller has asked for
    // A caller still wants this at `want`. Cleared once the interest is satisfied
    // (bytes reached the wanted tier, or a PREFETCHED copy was consumed) or given up
    // (demote). `follow_want` is a no-op while this is false, so an abandoned
    // prefetch does not keep climbing tiers after it lands in RAM.
    bool want_active = false;
    bool demand_pending = false;
    bool miss_reported = false;
    int64_t deadline_tick = -1;
    uint64_t last_use_tick = 0;
    uint64_t use_count = 0;
    SlotHandle slot = kNoSlot;
    void* ram = nullptr;
    TransferId load = kNoTransfer;  // NVMe -> RAM
    TransferId up = kNoTransfer;    // RAM -> VRAM
    // VRAM -> RAM. Only mutable bytes have one: for an expert the staging copy is
    // interchangeable with the file, but a KV page's newest contents may be in the
    // slot, and dropping the slot without this would silently lose them.
    TransferId down = kNoTransfer;
    TransferId out = kNoTransfer;   // RAM -> NVMe (mutable bytes only)
    int64_t ready_tick = -1;
  };

  Entry* find(ExpertId id);
  const Entry* find(ExpertId id) const;
  Entry* find_kv(KvPageId id);
  const Entry* find_kv(KvPageId id) const;

  TransitionEvent make_event(const Entry& e, Residency from, Residency to,
                             const char* cause) const;
  void set_state(Entry& e, Residency to, const char* cause);
  void touch(Entry& e);
  void drop_ram(Entry& e);
  void reset_entry(Entry& e);
  void start_nvme_read(Entry& e, IoClass io, const char* cause);
  void start_h2d(Entry& e);
  // Whenever bytes reach RAM and a higher tier is still wanted, the second hop
  // starts on the spot. One rule, so no state has to special-case resumption.
  void follow_want(Entry& e);
  void advance(Entry& e);
  void scan_deadlines(Entry& e);
  void preempt_prefetch();
  bool ready_entry(const Entry& e) const;

  Config cfg_;
  std::unordered_map<uint64_t, Entry> experts_;
  std::unordered_map<uint64_t, Entry> kv_;
  // Entries whose state can change only inside tick(): anything in flight,
  // anything with a demand deadline running, anything a PREFETCHED copy is aging
  // on. Everything else is dormant and re-added exactly when an action makes it
  // tick-relevant again. `scan_deadlines` still sees every deadline-bearing entry
  // because only entries with deadline_tick >= 0 are ever in the set.
  std::vector<ExpertId> dirty_experts_;
  std::unordered_map<uint64_t, uint8_t> dirty_expert_set_;
  std::vector<KvPageId> dirty_kv_;
  std::unordered_map<uint64_t, uint8_t> dirty_kv_set_;
  std::vector<ExpertId> evict_candidates_;
  std::vector<KvPageId> kv_evict_candidates_;
  uint32_t layer_ = 0;
  uint64_t tick_ = 0;
  uint64_t transitions_ = 0;
  uint64_t deadline_misses_ = 0;
  uint64_t preempted_ = 0;
  uint64_t refused_demotions_ = 0;
  uint64_t tick_calls_ = 0;
  uint64_t entry_visits_ = 0;
  uint64_t tick_ns_ = 0;
  // -- the dirty-set invariant ---------------------------------------------
  void mark_dirty(Entry& e) {
    if (e.is_kv) {
      if (dirty_kv_set_.emplace(e.kv.key(), 1).second) dirty_kv_.push_back(e.kv);
    } else {
      if (dirty_expert_set_.emplace(e.id.key(), 1).second) dirty_experts_.push_back(e.id);
    }
  }
  void unmark_dirty(Entry& e) {
    if (e.is_kv) dirty_kv_set_.erase(e.kv.key());
    else dirty_expert_set_.erase(e.id.key());
  }
};

}  // namespace knj

#endif  /* KNJ_RESIDENCY_H */
