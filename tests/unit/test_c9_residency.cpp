// tests/unit/test_c9_residency.cpp -- C9 Residency manager unit tests.
//
// C9 is the component the project exists for (AGENTS.md section 1): the table that
// lets an expert live on NVMe and still be executed. It is pure host C++17 by
// construction -- C8 slots, C11 transfers, C19 fallback and C21 telemetry are
// injected interfaces (src/residency/residency.h) -- so it is graded here against
// fakes, on a machine with no GPU in the loop at all, and the same code runs against
// the real HIP implementations later without a line changing.
//
// The six gates below are the acceptance criteria verbatim from
// ai-coder/c9-residency.md lines 74-80:
//
//   1. a non-resident demand enqueues an async C11 transfer and returns -- never
//      blocks -- and a demand RAM cannot satisfy returns "not ready" so that C19
//      covers the step; a cold demand escalates to NVMe rather than waiting.
//   2. the kernel asks only ready(); if it is false the scheduler has already failed
//      over to C19 for that expert -- never stalls.
//   3. a full-execution trace shows zero synchronous NVMe reads in the attention
//      path (I4).
//   4. every state transition emits a C21 event.
//   5. tick() advances the per-layer state, and a prefetch issued early arrives in
//      VRAM before its layer.
//   6. an evicted-and-reloaded KV page is bit-identical (I7) and an evicted-and-
//      reloaded expert reproduces the same contribution (I1).
//
// What the fakes are careful to model, because a gate that passes for the wrong
// reason is worse than a gate that fails:
//
//   * the world advances in ticks, and a transfer completes at a *stated* tick. The
//     fake performs the byte copy on the first poll that observes completion, so
//     nothing in a slot or in the backing store changes before the C11 state says it
//     has completed. A test that reads bytes therefore reads them only after C9 has
//     polled the completion.
//   * C8's load event is an event, not a timestamp C9 keeps: load_event_fired() is
//     false until the h2d that landed in that slot has completed.
//   * C8's released() is false until every transfer that *referenced* the slot has
//     finished, so a KV write-back that is still in flight really does hold the slot.
//   * the transfer engine counts submissions and polls that happen while the kernel
//     window is open. A synchronous NVMe read cannot exist without a submission, so
//     gate 3's "zero" is a measurement and not a promise.
//
// Exit codes:
//   0  every gate passed
//   1  at least one gate failed
//
// Run:
//   /c/Strawberry/c/bin/g++ -std=c++17 -O2 -Wall -Wextra -I. -I tests
//     src/residency/residency.cpp tests/unit/test_c9_residency.cpp
//     -o build/host/test_c9_residency.exe

#include "src/residency/residency.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

int g_checks = 0;
int g_failed = 0;

void check(bool ok, const char* what) {
  ++g_checks;
  if (!ok) {
    ++g_failed;
    std::printf("      FAIL  %s\n", what);
  }
}

#define CHECK(x) check((x), #x)

// ---------------------------------------------------------------------------
// The world: ticks are the only clock C9's control flow reads
// ---------------------------------------------------------------------------

struct FakeWorld {
  int64_t now = 0;
  int64_t nvme_latency = 2;  // ticks for an NVMe read/write to complete
  int64_t h2d_latency = 1;   // ticks for the RAM -> VRAM hop
  int64_t d2h_latency = 1;   // ticks for the VRAM -> RAM hop (KV write-back)
};

class FakeSlots;  // defined below; the transfer fake reports landings into it

// ---------------------------------------------------------------------------
// C11 fake -- every call is asynchronous and returns a handle. There is no
// blocking variant to call, which is the structural half of gate 3.
// ---------------------------------------------------------------------------

class FakeTransfer : public knj::TransferEngine {
 public:
  enum Kind { kNvmeRead = 0, kNvmeWrite = 1, kH2D = 2, kD2H = 3 };

  struct Rec {
    knj::TransferId id = 0;
    Kind kind = kNvmeRead;
    knj::IoClass io = knj::IoClass::DEMAND;
    uint64_t bytes = 0;
    int64_t complete_tick = 0;
    bool cancelled = false;
    bool failed = false;
    mutable bool copied = false;
    void* host = nullptr;
    void* dev = nullptr;
    uint64_t offset = 0;
  };

  explicit FakeTransfer(FakeWorld* w) : w_(w) {}

  void set_slots(FakeSlots* s) { slots_ = s; }

  void init_store(uint64_t bytes) {
    store_.assign((size_t)bytes, 0);
    for (size_t i = 0; i < store_.size(); ++i) {
      store_[i] = (uint8_t)((i * 131u + 7u) & 0xffu);
    }
  }

  const std::vector<uint8_t>& store() const { return store_; }

  // A read whose class the test cares about: DEMAND reads have a deadline, PREFETCH
  // reads do not and must be cancellable.
  knj::TransferId submit_nvme_read(void* dst_host, uint64_t src_offset, uint64_t bytes,
                                   knj::IoClass io) override {
    return submit(kNvmeRead, io, bytes, /*host=*/dst_host, /*dev=*/nullptr, src_offset);
  }
  knj::TransferId submit_nvme_write(const void* src_host, uint64_t dst_offset,
                                    uint64_t bytes, knj::IoClass io) override {
    return submit(kNvmeWrite, io, bytes, /*host=*/const_cast<void*>(src_host), nullptr,
                  dst_offset);
  }
  // Defined out of line below: it reports the landing into FakeSlots, which cannot
  // be a complete type here (FakeSlots in turn reads this fake's reference table).
  knj::TransferId submit_h2d(void* dst_device, const void* src_host, uint64_t bytes,
                             knj::IoClass io) override;
  knj::TransferId submit_d2h(void* dst_host, const void* src_device, uint64_t bytes,
                             knj::IoClass io) override {
    return submit(kD2H, io, bytes, dst_host, const_cast<void*>(src_device), 0);
  }

  knj::TransferState poll(knj::TransferId id) const override {
    if (kernel_window_) ++kernel_activity_;
    const Rec* r = find(id);
    if (r == nullptr) {
      ++unknown_polls_;
      return knj::TransferState::CANCELLED;
    }
    if (r->cancelled) return knj::TransferState::CANCELLED;
    if (w_->now < r->complete_tick) return knj::TransferState::PENDING;
    if (r->failed) return knj::TransferState::FAILED;
    if (!r->copied) {
      do_copy(*r);
      r->copied = true;
    }
    return knj::TransferState::DONE;
  }

  // A cancel succeeds only while the transfer is still pending -- cancelling work
  // that has already landed would make `preempted_prefetches` meaningless.
  bool cancel(knj::TransferId id) override {
    if (kernel_window_) ++kernel_activity_;
    Rec* r = find(id);
    if (r == nullptr) return false;
    if (r->cancelled) return false;
    if (w_->now >= r->complete_tick) return false;
    r->cancelled = true;
    ++cancels_;
    return true;
  }

  // ---- inspection ----
  void set_kernel_window(bool on) { kernel_window_ = on; }
  bool kernel_window() const { return kernel_window_; }
  uint64_t kernel_activity() const { return kernel_activity_; }
  uint64_t unknown_polls() const { return unknown_polls_; }

  uint64_t count(Kind k) const {
    uint64_t n = 0;
    for (const Rec& r : recs_) if (r.kind == k) ++n;
    return n;
  }
  uint64_t count_io(Kind k, knj::IoClass io) const {
    uint64_t n = 0;
    for (const Rec& r : recs_) if (r.kind == k && r.io == io) ++n;
    return n;
  }
  uint64_t pending_count() const {
    uint64_t n = 0;
    for (const Rec& r : recs_) if (!r.cancelled && !r.failed && w_->now < r.complete_tick) ++n;
    return n;
  }
  // Every NVMe read is the size of the object it fetches: C9 never splits an object
  // into smaller reads (AGENTS.md rule 13 -- the residency decision is fine-grained,
  // the I/O decision is not).
  bool every_read_is_whole(uint64_t extent_bytes) const {
    for (const Rec& r : recs_) {
      if (r.kind == kNvmeRead && r.bytes != extent_bytes) return false;
    }
    return true;
  }
  // The highest tick any transfer referencing `p` is still running, so C8's
  // release() can be made to mean what it says.
  int64_t ref_done_tick(const void* p) const { return refs_.count(p) ? refs_.at(p) : -1; }
  uint64_t cancels() const { return cancels_; }
  uint64_t writes() const { return count(kNvmeWrite); }

 private:
  const Rec& rec(knj::TransferId id) const { return *find(id); }
  const Rec* find(knj::TransferId id) const {
    for (const Rec& r : recs_) if (r.id == id) return &r;
    return nullptr;
  }
  Rec* find(knj::TransferId id) {
    for (Rec& r : recs_) if (r.id == id) return &r;
    return nullptr;
  }

  void do_copy(const Rec& r) const {
    switch (r.kind) {
      case kNvmeRead:
        std::memcpy(r.host, store_.data() + r.offset, (size_t)r.bytes);
        break;
      case kNvmeWrite:
        std::memcpy(store_.data() + r.offset, r.host, (size_t)r.bytes);
        break;
      case kH2D:
        std::memcpy(r.dev, r.host, (size_t)r.bytes);
        break;
      case kD2H:
        std::memcpy(r.host, r.dev, (size_t)r.bytes);
        break;
    }
  }

  knj::TransferId submit(Kind k, knj::IoClass io, uint64_t bytes, void* host, void* dev,
                         uint64_t offset) {
    if (kernel_window_) ++kernel_activity_;
    Rec r;
    r.id = ++next_id_;
    r.kind = k;
    r.io = io;
    r.bytes = bytes;
    r.complete_tick = w_->now + latency_of(k);
    r.host = host;
    r.dev = dev;
    r.offset = offset;
    recs_.push_back(r);
    // Note the reference for C8's release bookkeeping: a slot is held while any
    // transfer that touched it is still running.
    const int64_t done = r.complete_tick;
    if (host != nullptr) note_ref(host, done);
    if (dev != nullptr) note_ref(dev, done);
    return r.id;
  }

  void note_ref(const void* p, int64_t done) {
    auto it = refs_.find(p);
    if (it == refs_.end() || it->second < done) refs_[p] = done;
  }

  int64_t latency_of(Kind k) const {
    switch (k) {
      case kNvmeRead:
      case kNvmeWrite: return w_->nvme_latency;
      case kH2D: return w_->h2d_latency;
      case kD2H: return w_->d2h_latency;
    }
    return 1;
  }

  FakeWorld* w_;
  FakeSlots* slots_ = nullptr;
  mutable std::vector<uint8_t> store_;  // the fake backing store
  std::vector<Rec> recs_;
  std::map<const void*, int64_t> refs_;
  knj::TransferId next_id_ = 0;
  bool kernel_window_ = false;
  // poll() is const because the C11 interface observes transfer state; these counters
  // are observation machinery, not object identity, so they advance under a const poll.
  mutable uint64_t kernel_activity_ = 0;
  mutable uint64_t unknown_polls_ = 0;
  uint64_t cancels_ = 0;
};

// ---------------------------------------------------------------------------
// C8 fake -- a fixed number of fixed slots, one VA reservation, no per-expert
// allocation. The load event is an event, and a release completes only once the
// transfers that referenced the slot have finished.
// ---------------------------------------------------------------------------

class FakeSlots : public knj::SlotAllocator {
 public:
  FakeSlots(FakeWorld* w, int n, uint64_t cap) : w_(w), n_(n), cap_(cap) {
    slots_.resize((size_t)n);
    for (Slot& s : slots_) s.backing.reset(new uint8_t[(size_t)cap]);
  }

  void set_transfer(FakeTransfer* t) { xfer_ = t; }

  knj::SlotHandle acquire(uint32_t layer, uint32_t index, uint64_t bytes) override {
    ++acquire_calls_;
    if (bytes > cap_) {
      ++oversize_;
      return knj::kNoSlot;
    }
    for (int i = 0; i < n_; ++i) {
      Slot& s = slots_[(size_t)i];
      // A slot is handed out only once its previous release has completed: release()
      // alone does not free it, because a transfer that referenced it may still be
      // running (C8 owns that bookkeeping, C9 only observes it).
      if (occupied(s)) continue;
      s.in_use = true;
      s.release_pending = false;
      s.layer = layer;
      s.index = index;
      s.bytes = bytes;
      s.pinned = false;
      s.landing = 0;
      s.landing_done = 0;
      s.landing_cancelled = false;
      last_acquired_ = (knj::SlotHandle)i;
      return (knj::SlotHandle)i;
    }
    ++acquire_refusals_;
    return knj::kNoSlot;
  }

  void release(knj::SlotHandle h) override {
    if (!valid(h)) return;
    slots_[(size_t)h].release_pending = true;
  }

  // True only once every transfer that referenced this slot has finished. This is a
  // query and not a settle: the C8 interface is const here for exactly that reason,
  // C9 observes the release rather than performing it, and asking twice must give the
  // same answer -- otherwise C9's own advance() consumes the release and a second
  // query reports a slot that is in fact free as still held.
  bool released(knj::SlotHandle h) const override {
    if (!valid(h)) return false;
    const Slot& s = slots_[(size_t)h];
    if (!s.release_pending) return false;
    return release_done(s);
  }

  void pin(knj::SlotHandle h) override {
    if (!valid(h)) return;
    slots_[(size_t)h].pinned = true;
    ++pins_;
  }
  void unpin(knj::SlotHandle h) override {
    if (!valid(h)) return;
    slots_[(size_t)h].pinned = false;
    ++unpins_;
  }

  // C8's load event. False until the h2d that landed here has completed, so a slot
  // is never "ready" on the strength of a submission alone.
  bool load_event_fired(knj::SlotHandle h) const override {
    if (!valid(h)) return false;
    const Slot& s = slots_[(size_t)h];
    if (s.landing == 0 || s.landing_cancelled) return false;
    return w_->now >= s.landing_done;
  }

  void* vram_ptr(knj::SlotHandle h) const override {
    return valid(h) ? slots_[(size_t)h].backing.get() : nullptr;
  }
  uint64_t byte_size(knj::SlotHandle h) const override {
    return valid(h) ? slots_[(size_t)h].bytes : 0;
  }

  // ---- what the transfer fake reports back ----
  void note_landing(void* dst, knj::TransferId id, int64_t done_tick) {
    for (Slot& s : slots_) {
      if (s.backing.get() == dst) {
        s.landing = id;
        s.landing_done = done_tick;
        s.landing_cancelled = false;
        return;
      }
    }
  }

  // ---- inspection ----
  knj::SlotHandle last_acquired() const { return last_acquired_; }
  uint64_t acquire_calls() const { return acquire_calls_; }
  uint64_t acquire_refusals() const { return acquire_refusals_; }
  uint64_t oversize() const { return oversize_; }
  bool in_use(knj::SlotHandle h) const { return valid(h) && occupied(slots_[(size_t)h]); }
  int in_use_count() const {
    int n = 0;
    for (const Slot& s : slots_) if (occupied(s)) ++n;
    return n;
  }
  bool pinned(knj::SlotHandle h) const { return valid(h) && slots_[(size_t)h].pinned; }

 private:
  struct Slot {
    std::unique_ptr<uint8_t[]> backing;
    bool in_use = false;
    bool release_pending = false;
    bool pinned = false;
    uint32_t layer = 0;
    uint32_t index = 0;
    uint64_t bytes = 0;
    knj::TransferId landing = 0;
    int64_t landing_done = 0;
    bool landing_cancelled = false;
  };
  bool valid(knj::SlotHandle h) const { return h >= 0 && h < n_; }

  // A release is over once every transfer that referenced the slot has finished.
  bool release_done(const Slot& s) const {
    const int64_t done = (xfer_ != nullptr) ? xfer_->ref_done_tick(s.backing.get()) : -1;
    return done < 0 || w_->now >= done;
  }
  // Occupied = acquired and either never released, or released but still running. The
  // slot's bytes are untouched on acquire until this is false.
  bool occupied(const Slot& s) const {
    if (!s.in_use) return false;
    if (!s.release_pending) return true;
    return !release_done(s);
  }

  FakeWorld* w_;
  int n_;
  uint64_t cap_;
  std::vector<Slot> slots_;
  FakeTransfer* xfer_ = nullptr;
  knj::SlotHandle last_acquired_ = knj::kNoSlot;
  uint64_t acquire_calls_ = 0;
  uint64_t acquire_refusals_ = 0;
  uint64_t oversize_ = 0;
  uint64_t pins_ = 0;
  uint64_t unpins_ = 0;
};

// Defined here, not with the rest of FakeTransfer: FakeSlots must be complete.
knj::TransferId FakeTransfer::submit_h2d(void* dst_device, const void* src_host,
                                         uint64_t bytes, knj::IoClass io) {
  const knj::TransferId id = submit(kH2D, io, bytes,
                                    /*host=*/const_cast<void*>(src_host), dst_device, 0);
  if (slots_ != nullptr) {
    slots_->note_landing(dst_device, id, rec(id).complete_tick);
  }
  return id;
}

// ---------------------------------------------------------------------------
// RAM staging fake -- malloc/free with leak counting, so a dropped staging copy
// can be observed rather than assumed.
// ---------------------------------------------------------------------------

class FakeArena : public knj::HostArena {
 public:
  void* alloc(uint64_t bytes) override {
    void* p = std::malloc((size_t)bytes);
    if (p != nullptr) {
      ++allocs_;
      live_[p] = bytes;
    }
    return p;
  }
  void free(void* p) override {
    if (p == nullptr) return;
    ++frees_;
    live_.erase(p);
    std::free(p);
  }
  std::size_t live() const { return live_.size(); }
  const void* live_of_size(uint64_t bytes) const {
    for (const auto& kv : live_) if (kv.second == bytes) return kv.first;
    return nullptr;
  }
  uint64_t allocs() const { return allocs_; }
  uint64_t frees() const { return frees_; }

 private:
  std::map<void*, uint64_t> live_;
  uint64_t allocs_ = 0;
  uint64_t frees_ = 0;
};

// ---------------------------------------------------------------------------
// C19 fake -- the escape hatch. A missing expert must never silently disappear, so
// every call is recorded with the state C9 observed, and the test asserts the count.
// ---------------------------------------------------------------------------

class FakeFallback : public knj::FallbackSink {
 public:
  struct Rec {
    knj::ExpertId id;
    knj::Residency observed = knj::Residency::ABSENT;
  };
  struct KvRec {
    knj::KvPageId id;
    knj::Residency observed = knj::Residency::ABSENT;
  };

  void expert_unavailable(knj::ExpertId id, knj::Residency observed) override {
    experts.push_back(Rec{id, observed});
  }
  void kv_unavailable(knj::KvPageId id, knj::Residency observed) override {
    kvs.push_back(KvRec{id, observed});
  }
  std::size_t count_for(knj::ExpertId id) const {
    std::size_t n = 0;
    for (const Rec& r : experts) if (r.id == id) ++n;
    return n;
  }
  std::size_t count_for_kv(knj::KvPageId id) const {
    std::size_t n = 0;
    for (const KvRec& r : kvs) if (r.id == id) ++n;
    return n;
  }

  std::vector<Rec> experts;
  std::vector<KvRec> kvs;
};

// ---------------------------------------------------------------------------
// C21 fake -- every transition and every deadline miss, with no sampling.
// ---------------------------------------------------------------------------

class FakeTelemetry : public knj::TelemetrySink {
 public:
  void on_transition(const knj::TransitionEvent& ev) override { events.push_back(ev); }
  void on_deadline_missed(const knj::TransitionEvent& ev, uint64_t late) override {
    misses.push_back(ev);
    late_ticks.push_back(late);
  }
  bool saw(knj::Residency from, knj::Residency to, const char* cause) const {
    for (const knj::TransitionEvent& e : events) {
      if (e.from == from && e.to == to && std::strcmp(e.cause, cause) == 0) return true;
    }
    return false;
  }
  std::size_t count_cause(const char* cause) const {
    std::size_t n = 0;
    for (const knj::TransitionEvent& e : events) if (std::strcmp(e.cause, cause) == 0) ++n;
    return n;
  }
  std::vector<knj::TransitionEvent> events;
  std::vector<knj::TransitionEvent> misses;
  std::vector<uint64_t> late_ticks;
};

// ---------------------------------------------------------------------------
// KernelView: a kernel is handed this and nothing else. These are compile-time
// proofs that a kernel cannot ask where an object is or when it will arrive -- the
// absence of `locate`/`request` is not a convention here, it is a type error.
// ---------------------------------------------------------------------------

template <class T, class = void>
struct has_locate : std::false_type {};
template <class T>
struct has_locate<T, decltype(void(std::declval<const T&>().locate(
                        std::declval<knj::ExpertId>())))> : std::true_type {};

template <class T, class = void>
struct has_request : std::false_type {};
template <class T>
struct has_request<T, decltype(void(std::declval<const T&>().request(
                         std::declval<knj::ExpertId>(), knj::KVTier::VRAM)))>
    : std::true_type {};

template <class T, class = void>
struct has_tick : std::false_type {};
template <class T>
struct has_tick<T, decltype(void(std::declval<const T&>().tick()))> : std::true_type {};

using Kv = knj::ResidencyManager::KernelView;
static_assert(!has_locate<Kv>::value, "a kernel must not be able to ask where an object is");
static_assert(!has_request<Kv>::value, "a kernel must not be able to enqueue a transfer");
static_assert(!has_tick<Kv>::value, "a kernel must not be able to advance the tiers");
static_assert(has_locate<knj::ResidencyManager>::value, "the manager itself owns locate()");

// ---------------------------------------------------------------------------
// The rig: one world, one set of fakes, one manager
// ---------------------------------------------------------------------------

uint64_t fake_now_ns() {
  static uint64_t t = 0;
  return (t += 1000);
}

struct Rig {
  FakeWorld world;
  FakeTransfer xfer;
  FakeSlots slots;
  FakeArena arena;
  FakeFallback fb;
  FakeTelemetry tel;
  std::unique_ptr<knj::ResidencyManager> mgr;

  Rig(int n_slots, uint64_t slot_cap, uint64_t store_bytes, bool want_telemetry = true)
      : xfer(&world), slots(&world, n_slots, slot_cap) {
    xfer.set_slots(&slots);
    slots.set_transfer(&xfer);
    xfer.init_store(store_bytes);
    knj::ResidencyManager::Config cfg;
    cfg.slots = &slots;
    cfg.transfer = &xfer;
    cfg.arena = &arena;
    cfg.fallback = &fb;
    cfg.telemetry = want_telemetry ? &tel : nullptr;
    cfg.now_ns = &fake_now_ns;
    cfg.demand_deadline_ticks = 1;
    cfg.prefetch_ttl_ticks = 4;
    mgr.reset(new knj::ResidencyManager(cfg));
  }

  // One layer boundary: the world moves, then C9 observes. tick() is the only place
  // a completion is ever seen.
  void tick_after(int64_t dt) {
    world.now += dt;
    mgr->tick();
  }

  std::vector<uint8_t> snapshot_slot(knj::SlotHandle h) {
    std::vector<uint8_t> v((size_t)slots.byte_size(h));
    if (!v.empty()) std::memcpy(v.data(), slots.vram_ptr(h), v.size());
    return v;
  }
};

constexpr uint64_t kExpertBytes = 4096;
constexpr uint64_t kKvBytes = 2048;

}  // namespace

// ===========================================================================
// GATE 1 -- a non-resident demand enqueues an async C11 transfer and returns --
// never blocks -- and a demand RAM does not have in time arrives at NVMe and still
// returns "not ready" (never a blocking read).
// ===========================================================================

static bool gate1() {
  std::printf("  GATE 1  demand is async and never blocks\n");
  const int before = g_failed;
  // Exactly one slot on purpose: e0 takes it for the whole gate, so the later VRAM
  // demand for e1 has nowhere to land and the gate measures the refusal path. With
  // two slots the demand would succeed and the gate would be measuring something
  // else while still calling itself "no room, no wait".
  Rig rig(1, 8192, 1 << 16);
  knj::ResidencyManager& m = *rig.mgr;

  const knj::ExpertId e0{0, 0};
  const knj::ExpertId e1{0, 1};
  m.register_expert(e0, knj::BackingExtent{1024, kExpertBytes});
  m.register_expert(e1, knj::BackingExtent{8192, kExpertBytes});

  // --- 1a: cold demand returns immediately with the read already in flight ------
  const int64_t now_at_call = rig.world.now;
  const bool ready0 = m.demand(e0, knj::KVTier::VRAM);
  CHECK(!ready0);                                    // not ready on the spot
  CHECK(m.locate(e0) == knj::Residency::LOADING_NVME);  // escalated to NVMe
  CHECK(rig.xfer.count(FakeTransfer::kNvmeRead) == 1);  // exactly one async read
  CHECK(rig.xfer.count_io(FakeTransfer::kNvmeRead, knj::IoClass::DEMAND) == 1);
  CHECK(rig.xfer.pending_count() == 1);              // still outstanding => no wait
  CHECK(rig.world.now == now_at_call);               // no clock advanced => no stall

  // A second demand while in flight must not enqueue a second read.
  CHECK(!m.demand(e0, knj::KVTier::VRAM));
  CHECK(rig.xfer.count(FakeTransfer::kNvmeRead) == 1);

  // The whole two-hop path, one tick at a time.
  rig.tick_after(rig.world.nvme_latency);  // NVMe done -> RAM_RESIDENT -> h2d
  CHECK(m.locate(e0) == knj::Residency::LOADING_RAM);
  rig.tick_after(rig.world.h2d_latency);   // h2d done + event fired -> VRAM
  CHECK(m.locate(e0) == knj::Residency::VRAM_RESIDENT);
  CHECK(m.ready(e0));
  CHECK(m.demand(e0, knj::KVTier::VRAM));  // served, and only now

  // --- 1b: RAM does not have it in time -> not ready, C19 covers the step -------
  // Expert e1 goes to RAM only; the single slot is taken by e0, so the VRAM hop
  // cannot start and the demand must still return immediately.
  CHECK(!m.demand(e1, knj::KVTier::RAM));
  rig.tick_after(rig.world.nvme_latency);
  CHECK(m.locate(e1) == knj::Residency::RAM_RESIDENT);
  // A RAM demand that was served must not be recorded as a miss.
  CHECK(rig.tel.misses.empty());

  // Fill every slot with e0, then ask for e1 in VRAM: no room, no wait.
  const uint64_t refusals_before = rig.slots.acquire_refusals();
  const int64_t now_before = rig.world.now;
  const uint64_t reads_before = rig.xfer.count(FakeTransfer::kNvmeRead);
  CHECK(!m.demand(e1, knj::KVTier::VRAM));
  CHECK(m.locate(e1) == knj::Residency::RAM_RESIDENT);  // stayed where it was
  CHECK(rig.slots.acquire_refusals() == refusals_before + 1);
  CHECK(rig.world.now == now_before);
  CHECK(rig.xfer.count(FakeTransfer::kNvmeRead) == reads_before);  // no new I/O

  // The scheduler's move when it is not ready: hand the contribution to C19.
  CHECK(!m.demand_or_fallback(e1));
  CHECK(rig.fb.experts.size() == 1);
  CHECK(rig.fb.experts[0].id == e1);
  CHECK(rig.fb.experts[0].observed == knj::Residency::RAM_RESIDENT);

  // --- 1c: an unregistered id is ABSENT, enqueues nothing, and is recorded ------
  const knj::ExpertId ghost{9, 9};
  CHECK(m.locate(ghost) == knj::Residency::ABSENT);
  CHECK(!m.demand(ghost, knj::KVTier::VRAM));
  CHECK(rig.xfer.count(FakeTransfer::kNvmeRead) == reads_before);
  CHECK(!m.demand_or_fallback(ghost));
  CHECK(rig.fb.count_for(ghost) == 1);
  CHECK(rig.fb.experts.back().observed == knj::Residency::ABSENT);

  std::printf("          nvme_reads=%llu demand_reads=%llu h2d=%llu slot_refusals=%llu "
              "fallback=%llu\n",
              (unsigned long long)rig.xfer.count(FakeTransfer::kNvmeRead),
              (unsigned long long)rig.xfer.count_io(FakeTransfer::kNvmeRead,
                                                    knj::IoClass::DEMAND),
              (unsigned long long)rig.xfer.count(FakeTransfer::kH2D),
              (unsigned long long)rig.slots.acquire_refusals(),
              (unsigned long long)rig.fb.experts.size());
  return g_failed == before;
}

// ===========================================================================
// GATE 2 -- the kernel asks only ready(), and if it is false the scheduler has
// already failed over to C19 for that expert. Never a stall.
// ===========================================================================

static bool gate2() {
  std::printf("  GATE 2  ready() is the only question a kernel asks\n");
  const int before = g_failed;
  // One slot per expert in the step's working set (1 warm + 3 cold), because the
  // second step asserts that all three cold experts became resident and that C19 was
  // not called again. A smaller pool would make the assertion about the slot pool
  // rather than about C9's hand-off discipline.
  Rig rig(4, 8192, 1 << 16);
  knj::ResidencyManager& m = *rig.mgr;
  const knj::ResidencyManager::KernelView view = m.kernel_view();

  const knj::ExpertId warm{0, 0};
  const knj::ExpertId cold[3] = {{0, 1}, {0, 2}, {0, 3}};
  m.register_expert(warm, knj::BackingExtent{0, kExpertBytes});
  for (int i = 0; i < 3; ++i) {
    m.register_expert(cold[i], knj::BackingExtent{16384 + (uint64_t)i * 8192, kExpertBytes});
  }

  // Make `warm` resident before the step, so the step has both answers to give.
  m.prefetch(warm, knj::KVTier::VRAM, knj::PrefetchPriority::URGENT);
  rig.tick_after(rig.world.nvme_latency);
  rig.tick_after(rig.world.h2d_latency);
  CHECK(m.ready(warm));

  // One step: the scheduler runs first, the kernel second, exactly as the layer
  // graph orders them.
  int kernel_true = 0;
  int kernel_false = 0;
  int fallback_after_kernel = 0;

  const knj::ExpertId routed[4] = {warm, cold[0], cold[1], cold[2]};
  bool scheduled[4] = {false, false, false, false};
  for (int i = 0; i < 4; ++i) {
    // Scheduler phase: this is the only place a demand is made, and it never waits.
    scheduled[i] = m.demand_or_fallback(routed[i]);
  }
  // Kernel phase: the KernelView and nothing else.
  for (int i = 0; i < 4; ++i) {
    const bool r = view.ready(routed[i]);
    if (r) ++kernel_true; else ++kernel_false;
    // The kernel's answer must be the scheduler's answer, and a false must already
    // have a C19 hand-off behind it -- the kernel never discovers a miss itself.
    CHECK(r == scheduled[i]);
    if (!r) ++fallback_after_kernel;
  }
  CHECK(kernel_true == 1);
  CHECK(kernel_false == 3);
  CHECK(fallback_after_kernel == 3);
  CHECK(rig.fb.experts.size() == 3);
  for (int i = 0; i < 3; ++i) CHECK(rig.fb.count_for(cold[i]) == 1);
  CHECK(rig.fb.count_for(warm) == 0);  // the resident expert never reached C19

  // A second step over the same set: the three cold experts were escalated in step
  // one, so with the world advanced they are resident and C19 is not called again.
  const std::size_t fb_before = rig.fb.experts.size();
  rig.tick_after(rig.world.nvme_latency);
  rig.tick_after(rig.world.h2d_latency);
  for (int i = 0; i < 3; ++i) CHECK(view.ready(cold[i]));
  for (int i = 0; i < 3; ++i) CHECK(m.demand_or_fallback(cold[i]));
  CHECK(rig.fb.experts.size() == fb_before);

  std::printf("          kernel ready=%d not_ready=%d c19_handoffs=%llu "
              "second_step_handoffs=0\n",
              kernel_true, kernel_false, (unsigned long long)rig.fb.experts.size());
  return g_failed == before;
}

// ===========================================================================
// GATE 3 -- a full-execution trace shows zero synchronous NVMe reads in the
// attention path (I4).
// ===========================================================================

static bool gate3() {
  std::printf("  GATE 3  zero synchronous NVMe reads in the attention path\n");
  const int before = g_failed;
  // One slot per page: this trace is about *where* the NVMe traffic is issued, not
  // about eviction pressure (gate 6 owns that), so every page can stay resident and
  // the only variable left is whether the kernel window ever causes a transfer.
  Rig rig(8, 8192, 1 << 16);
  knj::ResidencyManager& m = *rig.mgr;
  const knj::ResidencyManager::KernelView view = m.kernel_view();

  const int kLayers = 4;
  const int kPages = 2;
  for (int L = 0; L < kLayers; ++L) {
    for (int p = 0; p < kPages; ++p) {
      m.register_kv_page(knj::KvPageId{(uint32_t)L, (uint32_t)p},
                         knj::BackingExtent{512 + (uint64_t)(L * kPages + p) * 4096, kKvBytes});
    }
  }

  // A page crosses two hops (NVMe -> RAM -> VRAM), and tick() observes exactly one
  // completion per transfer per call, so a page demanded in layer L cannot possibly
  // be resident during layer L's own kernel phase. The trace is therefore a pipeline
  // with a two-layer lookahead: each layer's scheduler demands the pages of the layer
  // two ahead, and its kernel attends over pages that were demanded two layers ago.
  // A gate that demanded and then immediately read the same page would be asserting
  // something C9 deliberately does not do -- the demand would never be satisfiable,
  // and the gate would be measuring the fake's tick resolution, not I4.
  const int kLookahead = 2;

  // Cold start: the first kLookahead layers' pages are demanded up front so the
  // pipeline is already full when the trace begins. This is still asynchronous --
  // nothing here waits -- and it is the only NVMe work not overlapped with a kernel.
  for (int L = 0; L < kLookahead; ++L) {
    for (int p = 0; p < kPages; ++p) {
      m.demand_kv(knj::KvPageId{(uint32_t)L, (uint32_t)p}, knj::KVTier::VRAM);
    }
  }
  rig.tick_after(rig.world.nvme_latency);
  rig.tick_after(rig.world.h2d_latency);

  int kernel_ready = 0;
  for (int L = 0; L < kLayers; ++L) {
    m.begin_layer((uint32_t)L);

    // Scheduler phase: demand the pages of the layer kLookahead ahead. This is the
    // only NVMe traffic the attention path is allowed to cause.
    const int ahead = L + kLookahead;
    if (ahead < kLayers) {
      for (int p = 0; p < kPages; ++p) {
        m.demand_kv(knj::KvPageId{(uint32_t)ahead, (uint32_t)p}, knj::KVTier::VRAM);
      }
    }

    // Kernel phase. Any transfer-engine activity inside this window is by definition
    // a synchronous read, because a blocking read cannot be built without a submit.
    rig.xfer.set_kernel_window(true);
    for (int p = 0; p < kPages; ++p) {
      if (view.ready_kv(knj::KvPageId{(uint32_t)L, (uint32_t)p})) ++kernel_ready;
    }
    rig.xfer.set_kernel_window(false);

    rig.world.now += rig.world.nvme_latency + rig.world.h2d_latency;
    m.tick();
  }

  CHECK(rig.xfer.kernel_activity() == 0);
  CHECK(rig.xfer.unknown_polls() == 0);
  CHECK(knj::ResidencyManager::blocking_transfer_calls() == 0);
  // Every NVMe read fetched a whole object: never a tiny random read.
  CHECK(rig.xfer.every_read_is_whole(kKvBytes));
  // Demand-class traffic owns the bandwidth; nothing in the attention path is a
  // speculative write.
  CHECK(rig.xfer.count_io(FakeTransfer::kNvmeRead, knj::IoClass::DEMAND) == kLayers * kPages);
  CHECK(rig.xfer.count_io(FakeTransfer::kNvmeRead, knj::IoClass::PREFETCH) == 0);
  CHECK(rig.xfer.writes() == 0);
  // The attention path never recorded a stall: no KV demand missed its deadline.
  CHECK(rig.tel.misses.empty());
  // Every page the kernel attended over was already resident: the lookahead really
  // did cover both hops, and the loop really did reach resident pages.
  CHECK(kernel_ready == kLayers * kPages);

  std::printf("          kernel_window_transfer_calls=%llu nvme_reads=%llu (demand=%llu "
              "prefetch=%llu) writes=%llu kv_pages_ready_in_kernel=%d\n",
              (unsigned long long)rig.xfer.kernel_activity(),
              (unsigned long long)rig.xfer.count(FakeTransfer::kNvmeRead),
              (unsigned long long)rig.xfer.count_io(FakeTransfer::kNvmeRead,
                                                    knj::IoClass::DEMAND),
              (unsigned long long)rig.xfer.count_io(FakeTransfer::kNvmeRead,
                                                    knj::IoClass::PREFETCH),
              (unsigned long long)rig.xfer.writes(), kernel_ready);
  return g_failed == before;
}

// ===========================================================================
// GATE 4 -- every state transition emits a C21 telemetry event.
// ===========================================================================

static bool gate4() {
  std::printf("  GATE 4  every transition is observable (C21)\n");
  const int before = g_failed;
  Rig rig(3, 8192, 1 << 16);
  knj::ResidencyManager& m = *rig.mgr;

  const knj::ExpertId e0{0, 0};
  const knj::ExpertId e1{0, 1};
  const knj::KvPageId k0{0, 0};
  m.register_expert(e0, knj::BackingExtent{0, kExpertBytes});
  m.register_expert(e1, knj::BackingExtent{8192, kExpertBytes});
  m.register_kv_page(k0, knj::BackingExtent{512, kKvBytes});

  // Exercise as many transitions as the machine has: demand, prefetch, promote,
  // consume, pin, refuse a demotion, demote by one tier, evict to NVMe, re-read.
  m.demand(e0, knj::KVTier::VRAM);
  rig.tick_after(2);
  rig.tick_after(1);
  CHECK(m.ready(e0));
  m.pin(e0);
  CHECK(m.locate(e0) == knj::Residency::PINNED);
  m.demote(e0);  // refused: PINNED outranks eviction
  CHECK(m.refused_demotions() == 1);
  m.unpin(e0);
  m.prefetch(e1, knj::KVTier::VRAM, knj::PrefetchPriority::NORMAL);
  rig.tick_after(2);
  rig.tick_after(1);
  CHECK(m.locate(e1) == knj::Residency::PREFETCHED);
  m.demand(e1, knj::KVTier::VRAM);  // consumes the speculative copy
  CHECK(m.locate(e1) == knj::Residency::VRAM_RESIDENT);
  CHECK(rig.tel.saw(knj::Residency::PREFETCHED, knj::Residency::VRAM_RESIDENT, "consumed"));
  m.demote(e0);
  rig.tick_after(1);
  m.demote(e0);  // RAM -> NVMe, immutable so no write-back
  CHECK(m.locate(e0) == knj::Residency::NVME_RESIDENT);
  m.demand(e0, knj::KVTier::VRAM);  // re-read
  const uint64_t transitions_after_works = m.transitions();

  CHECK(rig.tel.events.size() == m.transitions());
  CHECK(rig.tel.events.size() >= transitions_after_works);
  CHECK(rig.tel.events.size() > 0);
  for (const knj::TransitionEvent& ev : rig.tel.events) {
    CHECK(ev.cause != nullptr && ev.cause[0] != '\0');
  }
  CHECK(rig.tel.misses.size() == m.deadline_misses());
  // The event stream names the transitions an auditor would look for.
  CHECK(rig.tel.count_cause("demand_miss") >= 1);
  CHECK(rig.tel.count_cause("nvme_read_done") >= 1);
  CHECK(rig.tel.count_cause("h2d_issued") >= 1);
  CHECK(rig.tel.count_cause("event_fired") >= 1);
  CHECK(rig.tel.saw(knj::Residency::VRAM_RESIDENT, knj::Residency::EVICTING_VRAM, "evict"));
  CHECK(rig.tel.count_cause("evict_done") >= 1);
  CHECK(rig.tel.saw(knj::Residency::ABSENT, knj::Residency::NVME_RESIDENT, "pin") == false);
  CHECK(rig.tel.saw(knj::Residency::VRAM_RESIDENT, knj::Residency::PINNED, "pin"));
  CHECK(rig.tel.saw(knj::Residency::PINNED, knj::Residency::VRAM_RESIDENT, "unpin"));
  CHECK(rig.tel.saw(knj::Residency::INVALID, knj::Residency::LOADING_NVME, "reread") ||
        rig.tel.count_cause("demand_miss") >= 2);

  // The counter must not depend on a sink being installed: the same work without
  // telemetry produces the same number of transitions.
  const uint64_t with_sink = m.transitions();
  Rig blind(3, 8192, 1 << 16, /*want_telemetry=*/false);
  knj::ResidencyManager& b = *blind.mgr;
  b.register_expert(e0, knj::BackingExtent{0, kExpertBytes});
  b.register_expert(e1, knj::BackingExtent{8192, kExpertBytes});
  b.register_kv_page(k0, knj::BackingExtent{512, kKvBytes});
  b.demand(e0, knj::KVTier::VRAM);
  blind.tick_after(2);
  blind.tick_after(1);
  b.pin(e0);
  b.demote(e0);
  b.unpin(e0);
  b.prefetch(e1, knj::KVTier::VRAM, knj::PrefetchPriority::NORMAL);
  blind.tick_after(2);
  blind.tick_after(1);
  b.demand(e1, knj::KVTier::VRAM);
  b.demote(e0);
  blind.tick_after(1);
  b.demote(e0);
  b.demand(e0, knj::KVTier::VRAM);
  CHECK(b.transitions() == with_sink);
  CHECK(blind.tel.events.empty());

  std::printf("          transitions=%llu telemetry_events=%llu deadline_misses=%llu "
              "miss_records=%llu (identical without a sink: %llu)\n",
              (unsigned long long)m.transitions(),
              (unsigned long long)rig.tel.events.size(),
              (unsigned long long)m.deadline_misses(),
              (unsigned long long)rig.tel.misses.size(), (unsigned long long)b.transitions());
  return g_failed == before;
}

// ===========================================================================
// GATE 5 -- tick() advances the per-layer state (prefetch horizon, eviction
// scoring, promotion/demotion), and a prefetch issued early arrives in VRAM before
// its layer.
// ===========================================================================

static bool gate5() {
  std::printf("  GATE 5  tick() advances the layers and prefetch arrives in time\n");
  const int before = g_failed;
  Rig rig(4, 8192, 1 << 16);
  knj::ResidencyManager& m = *rig.mgr;

  const knj::ExpertId future{3, 0};
  m.register_expert(future, knj::BackingExtent{0, kExpertBytes});

  // --- 5a: prefetch at layer 0 for a layer-3 expert -----------------------------
  m.begin_layer(0);
  m.prefetch(future, knj::KVTier::VRAM, knj::PrefetchPriority::NORMAL);
  CHECK(m.locate(future) == knj::Residency::LOADING_NVME);
  CHECK(rig.xfer.count_io(FakeTransfer::kNvmeRead, knj::IoClass::PREFETCH) == 1);

  for (uint32_t L = 0; L < 3; ++L) {
    m.begin_layer(L);
    rig.tick_after(3);  // both hops can complete inside one layer
  }
  // Issued at layer 0, landed at layer 1, still speculative until layer 3 asks.
  CHECK(m.locate(future) == knj::Residency::PREFETCHED);
  CHECK(m.ready(future));
  // The event names the hop that actually finished: with want == VRAM the object
  // does not rest at RAM_RESIDENT, it goes straight on into the h2d, so the state it
  // lands *from* is LOADING_RAM. (The intermediate RAM_RESIDENT is still observable --
  // it is the `nvme_read_done` event.) The gate's claim is that a speculative copy
  // becomes PREFETCHED, and that is what is asserted here.
  CHECK(rig.tel.saw(knj::Residency::LOADING_RAM, knj::Residency::PREFETCHED,
                    "prefetch_landed"));
  CHECK(rig.tel.count_cause("prefetch_landed") == 1);
  // The pipeline advanced with the layers, not on a background thread.
  CHECK(m.tick_index() == 3);
  CHECK(m.layer() == 2);

  // Layer 3 runs and the copy is consumed: not a miss, and no new I/O.
  const uint64_t reads_before = rig.xfer.count(FakeTransfer::kNvmeRead);
  m.begin_layer(3);
  CHECK(m.ready(future));  // "prefetch -> RAM -> VRAM -> attention"
  CHECK(m.demand(future, knj::KVTier::VRAM));
  CHECK(m.locate(future) == knj::Residency::VRAM_RESIDENT);
  CHECK(rig.xfer.count(FakeTransfer::kNvmeRead) == reads_before);
  CHECK(rig.fb.experts.empty());  // nothing fell to C19

  // --- 5b: eviction scoring is driven by tick() and the prefetch TTL ------------
  Rig r2(4, 8192, 1 << 16);
  knj::ResidencyManager& m2 = *r2.mgr;
  const knj::ExpertId stale{7, 0};
  m2.register_expert(stale, knj::BackingExtent{0, kExpertBytes});

  m2.begin_layer(0);
  m2.prefetch(stale, knj::KVTier::VRAM, knj::PrefetchPriority::NORMAL);
  int candidate_tick = -1;
  for (uint32_t L = 0; L <= 5; ++L) {
    m2.begin_layer(L);
    r2.tick_after(3);
    if (candidate_tick < 0 && !m2.eviction_candidates().empty()) {
      candidate_tick = (int)m2.tick_index();
    }
  }
  CHECK(m2.locate(stale) == knj::Residency::PREFETCHED);
  CHECK(candidate_tick == 6);  // landed at tick_index 2, + ttl 4 == 6
  CHECK(m2.eviction_candidates().size() == 1);
  CHECK(m2.eviction_candidates()[0] == stale);
  // A speculative copy nobody consumed is exactly what C12 is shown.
  CHECK(r2.tel.count_cause("prefetch_landed") == 1);

  // --- 5c: tick() also advances promotion and demotion -------------------------
  Rig r3(4, 8192, 1 << 16);
  knj::ResidencyManager& m3 = *r3.mgr;
  const knj::ExpertId p{1, 0};
  m3.register_expert(p, knj::BackingExtent{0, kExpertBytes});
  m3.begin_layer(1);
  m3.promote(p);  // one tier: NVMe -> RAM
  CHECK(m3.locate(p) == knj::Residency::LOADING_NVME);
  r3.tick_after(3);
  CHECK(m3.locate(p) == knj::Residency::RAM_RESIDENT);
  m3.promote(p);  // one tier: RAM -> VRAM. promote() is a *prefetch*, not a demand
                  // (residency.cpp: "a promotion is a scheduling decision, and it must
                  // not turn into a step's obligation"), so the copy lands speculative.
  r3.tick_after(3);
  CHECK(m3.locate(p) == knj::Residency::PREFETCHED);
  CHECK(m3.ready(p));  // speculative still means resident and usable
  CHECK(m3.demand(p, knj::KVTier::VRAM));  // a step asks, and that consumes it
  CHECK(m3.locate(p) == knj::Residency::VRAM_RESIDENT);
  m3.demote(p);   // one tier: VRAM -> RAM
  r3.tick_after(3);
  CHECK(m3.locate(p) == knj::Residency::RAM_RESIDENT);
  m3.demote(p);   // one tier: RAM -> NVMe
  CHECK(m3.locate(p) == knj::Residency::NVME_RESIDENT);

  std::printf("          prefetch issued at layer 0 -> PREFETCHED by layer 1 -> consumed at "
              "layer 3; first eviction candidate at tick_index=%d (ttl=4); "
              "promote/demote one tier each\n",
              candidate_tick);
  return g_failed == before;
}

// ===========================================================================
// GATE 6 -- eviction and reload are invisible: bit-for-bit for KV (I7), the same
// contribution for an expert (I1).
// ===========================================================================

static bool gate6() {
  std::printf("  GATE 6  evict + reload is invisible (I7 / I1)\n");
  const int before = g_failed;
  Rig rig(2, 8192, 1 << 16);
  knj::ResidencyManager& m = *rig.mgr;

  const knj::ExpertId expert{0, 0};
  const knj::KvPageId kv{0, 0};
  m.register_expert(expert, knj::BackingExtent{1024, kExpertBytes});
  m.register_kv_page(kv, knj::BackingExtent{512, kKvBytes});

  // ---------- expert: RAM reload ------------------------------------------------
  CHECK(!m.demand(expert, knj::KVTier::VRAM));
  rig.tick_after(rig.world.nvme_latency);
  rig.tick_after(rig.world.h2d_latency);
  CHECK(m.ready(expert));
  const knj::SlotHandle eslot = rig.slots.last_acquired();
  const std::vector<uint8_t> expert_first = rig.snapshot_slot(eslot);
  CHECK(expert_first.size() == kExpertBytes);

  m.demote(expert);  // VRAM -> EVICTING_VRAM -> RAM
  rig.tick_after(1);
  CHECK(m.locate(expert) == knj::Residency::RAM_RESIDENT);
  CHECK(!rig.slots.in_use(eslot));  // the slot really was retired
  CHECK(!m.ready(expert));

  CHECK(!m.demand(expert, knj::KVTier::VRAM));
  CHECK(m.locate(expert) == knj::Residency::LOADING_RAM);
  rig.tick_after(rig.world.h2d_latency);
  CHECK(m.ready(expert));
  const std::vector<uint8_t> expert_after_ram = rig.snapshot_slot(rig.slots.last_acquired());
  CHECK(expert_after_ram.size() == expert_first.size());
  CHECK(std::memcmp(expert_first.data(), expert_after_ram.data(), expert_first.size()) == 0);

  // ---------- expert: full NVMe round trip -------------------------------------
  m.demote(expert);  // VRAM -> RAM
  rig.tick_after(1);
  m.demote(expert);  // RAM -> NVMe; immutable bytes, so staging is simply dropped
  CHECK(m.locate(expert) == knj::Residency::NVME_RESIDENT);
  CHECK(rig.arena.live() == 0);
  const uint64_t writes_before = rig.xfer.writes();

  CHECK(!m.demand(expert, knj::KVTier::VRAM));
  rig.tick_after(rig.world.nvme_latency);
  rig.tick_after(rig.world.h2d_latency);
  CHECK(m.ready(expert));
  const std::vector<uint8_t> expert_after_nvme =
      rig.snapshot_slot(rig.slots.last_acquired());
  CHECK(std::memcmp(expert_first.data(), expert_after_nvme.data(), expert_first.size()) == 0);
  // Dropping an expert's staging copy writes nothing: the file is authoritative.
  CHECK(rig.xfer.writes() == writes_before);

  // ---------- KV: the mutable case, which is why the write-back exists ---------
  CHECK(!m.demand_kv(kv, knj::KVTier::VRAM));
  rig.tick_after(rig.world.nvme_latency);
  rig.tick_after(rig.world.h2d_latency);
  CHECK(m.ready_kv(kv));
  const knj::SlotHandle kslot = rig.slots.last_acquired();

  // The bytes as they came off NVMe, captured *before* the kernel touches the slot.
  // Snapshotting after the write would compare the update against itself -- and the
  // check would pass while proving nothing, which is the failure mode this whole
  // file exists to avoid. Equality with the backing store is asserted too, so the
  // baseline is known to be the file's bytes and not just some other bytes.
  const std::vector<uint8_t> stale = rig.snapshot_slot(kslot);
  CHECK(stale.size() == (size_t)kKvBytes);
  CHECK(std::memcmp(stale.data(), rig.xfer.store().data() + 512, (size_t)kKvBytes) == 0);

  // The kernel writes the newest KV contents into the slot.
  std::vector<uint8_t> updated((size_t)kKvBytes);
  for (size_t i = 0; i < updated.size(); ++i) updated[i] = (uint8_t)(0xA5u ^ (i * 7u));
  std::memcpy(rig.slots.vram_ptr(kslot), updated.data(), updated.size());
  CHECK(std::memcmp(updated.data(), stale.data(), updated.size()) != 0);

  // Demote off VRAM: the update has to travel down before the slot can go.
  m.demote_kv(kv);
  CHECK(m.locate_kv(kv) == knj::Residency::EVICTING_VRAM);
  CHECK(rig.tel.saw(knj::Residency::VRAM_RESIDENT, knj::Residency::EVICTING_VRAM, "evict"));
  CHECK(!rig.slots.released(kslot));  // the write-back still holds the slot
  rig.tick_after(rig.world.d2h_latency);
  CHECK(m.locate_kv(kv) == knj::Residency::RAM_RESIDENT);
  CHECK(rig.slots.released(kslot));
  const void* staging = rig.arena.live_of_size(kKvBytes);
  CHECK(staging != nullptr);
  // The staging copy now carries the update, not the bytes that came off NVMe.
  CHECK(std::memcmp(staging, updated.data(), updated.size()) == 0);

  m.demote_kv(kv);  // RAM -> NVMe: the one place C9 issues a write
  CHECK(m.locate_kv(kv) == knj::Residency::EVICTING_RAM);
  CHECK(rig.xfer.writes() == writes_before + 1);
  rig.tick_after(rig.world.nvme_latency);
  CHECK(m.locate_kv(kv) == knj::Residency::NVME_RESIDENT);
  // The page's staging buffer was freed with the page. This is deliberately *not*
  // `rig.arena.live() == 0`: the expert is still VRAM_RESIDENT from its own reload
  // above and legitimately still holds its staging buffer. Asserting the whole arena
  // were empty here would be a wrong claim about C9, not a stricter one.
  CHECK(rig.arena.live_of_size(kKvBytes) == nullptr);
  CHECK(rig.arena.live() == 1);  // exactly the expert's staging copy remains
  // ... and the update is what reached the backing store.
  CHECK(std::memcmp(rig.xfer.store().data() + 512, updated.data(), updated.size()) == 0);

  // Reload: bit-identical to what the kernel last wrote (I7).
  CHECK(!m.demand_kv(kv, knj::KVTier::VRAM));
  rig.tick_after(rig.world.nvme_latency);
  rig.tick_after(rig.world.h2d_latency);
  CHECK(m.ready_kv(kv));
  const std::vector<uint8_t> kv_after = rig.snapshot_slot(rig.slots.last_acquired());
  CHECK(kv_after.size() == updated.size());
  CHECK(std::memcmp(kv_after.data(), updated.data(), updated.size()) == 0);

  // ---------- hygiene: reset() leaves nothing behind ---------------------------
  // Both objects are resident in VRAM here, so both hold a RAM staging buffer and a
  // slot; reset() has to give all four back.
  const std::size_t arena_live_before_reset = rig.arena.live();
  const int slots_before_reset = rig.slots.in_use_count();
  m.reset();
  CHECK(m.locate(expert) == knj::Residency::NVME_RESIDENT);
  CHECK(m.locate_kv(kv) == knj::Residency::NVME_RESIDENT);
  CHECK(rig.arena.live() == 0);
  CHECK(rig.slots.in_use_count() == 0);
  CHECK(!m.ready(expert));
  CHECK(!m.ready_kv(kv));

  std::printf("          expert bytes identical across RAM reload=%s NVMe reload=%s; "
              "KV newest-contents survived write-back=%s and reload is bit-identical=%s; "
              "expert_staging_writes=%llu kv_writes=%llu\n",
              (std::memcmp(expert_first.data(), expert_after_ram.data(),
                           expert_first.size()) == 0) ? "yes" : "no",
              (std::memcmp(expert_first.data(), expert_after_nvme.data(),
                           expert_first.size()) == 0) ? "yes" : "no",
              (std::memcmp(staging, updated.data(), updated.size()) == 0) ? "yes" : "no",
              (std::memcmp(kv_after.data(), updated.data(), updated.size()) == 0) ? "yes" : "no",
              (unsigned long long)writes_before, (unsigned long long)rig.xfer.writes());
  std::printf("          teardown: staging live before reset=%llu after=%llu "
              "(allocs=%llu frees=%llu); slots held before reset=%d after=%d\n",
              (unsigned long long)arena_live_before_reset,
              (unsigned long long)rig.arena.live(),
              (unsigned long long)rig.arena.allocs(),
              (unsigned long long)rig.arena.frees(), slots_before_reset,
              rig.slots.in_use_count());
  return g_failed == before;
}

// ===========================================================================
// Main
// ===========================================================================

int main() {
  std::printf("=== C9 Residency Manager Unit Tests (host fakes, no device) ===\n\n");
  std::printf("  C8 slots: fixed pool, HIP-event load event, release bounded by transfers\n");
  std::printf("  C11 transfer: every call async, completion at a stated tick\n");
  std::printf("  C19 fallback: every hand-off recorded with the observed state\n");
  std::printf("  C21 telemetry: every transition, no sampling\n\n");

  const int gates_before = g_failed;
  const bool g1 = gate1();
  const bool g2 = gate2();
  const bool g3 = gate3();
  const bool g4 = gate4();
  const bool g5 = gate5();
  const bool g6 = gate6();
  const int gates_passed = (g1 ? 1 : 0) + (g2 ? 1 : 0) + (g3 ? 1 : 0) + (g4 ? 1 : 0) +
                           (g5 ? 1 : 0) + (g6 ? 1 : 0);
  (void)gates_before;

  std::printf("\n=== Summary ===\n");
  std::printf("  gates run:    6\n");
  std::printf("  gates passed: %d\n", gates_passed);
  std::printf("  gates failed: %d\n", 6 - gates_passed);
  std::printf("  checks run:   %d\n", g_checks);
  std::printf("  checks failed:%d\n", g_failed);

  if (g_failed > 0) {
    std::printf("\nRESULT: FAIL\n");
    return 1;
  }
  std::printf("\nRESULT: PASS\n");
  return 0;
}
