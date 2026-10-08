// src/residency/residency.cpp -- the C9 state machine.
//
// Read src/residency/residency.h first: it says why the rules exist. This file is
// the mechanism, and it is written so that the four things an auditor would look
// for are visible in the code rather than implied by it:
//
//   * there is no blocking call anywhere on the demand path -- the only transfers
//     are the async submissions in `start_nvme_read` / `start_h2d` / `demote`;
//   * `ready()` is a query to C8's load event, not a flag this file keeps;
//   * every state change goes through `set_state()`, which is the single place a
//     C21 event is built, so no transition can be added without an event;
//   * `tick()` is the only place a transfer completion is observed, so the whole
//     pipeline advances at the layer boundary instead of on a background thread
//     racing the step loop.

#include "src/residency/residency.h"

#include <cassert>

namespace knj {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const char* to_string(Residency r) {
  switch (r) {
    case Residency::ABSENT: return "ABSENT";
    case Residency::NVME_RESIDENT: return "NVME_RESIDENT";
    case Residency::LOADING_NVME: return "LOADING_NVME";
    case Residency::RAM_RESIDENT: return "RAM_RESIDENT";
    case Residency::LOADING_RAM: return "LOADING_RAM";
    case Residency::VRAM_RESIDENT: return "VRAM_RESIDENT";
    case Residency::PREFETCHED: return "PREFETCHED";
    case Residency::EVICTING_VRAM: return "EVICTING_VRAM";
    case Residency::EVICTING_RAM: return "EVICTING_RAM";
    case Residency::PINNED: return "PINNED";
    case Residency::INVALID: return "INVALID";
  }
  return "?";
}

bool is_vram_state(Residency r) {
  return r == Residency::VRAM_RESIDENT || r == Residency::PREFETCHED ||
         r == Residency::PINNED;
}

const char* to_string(KVTier t) {
  switch (t) {
    case KVTier::NVME: return "NVME";
    case KVTier::RAM: return "RAM";
    case KVTier::VRAM: return "VRAM";
  }
  return "?";
}

KVTier next_tier_above(KVTier t) {
  switch (t) {
    case KVTier::NVME: return KVTier::RAM;
    case KVTier::RAM: return KVTier::VRAM;
    case KVTier::VRAM: return KVTier::VRAM;
  }
  return KVTier::VRAM;
}

// Has the object's bytes reached the tier that was asked for? This is what makes an
// outstanding interest satisfiable without a second transfer: a request for RAM is
// served by the NVMe read itself, and a request for VRAM is served by either hop.
// In-flight and gone states are deliberately not "reached" -- issued is not ready.
static bool tier_reached(Residency r, KVTier want) {
  switch (r) {
    case Residency::VRAM_RESIDENT:
    case Residency::PREFETCHED:
    case Residency::PINNED:
      return true;  // top tier: anything below it is satisfied
    case Residency::RAM_RESIDENT:
      return want != KVTier::VRAM;
    default:
      return false;  // LOADING_*, EVICTING_*, NVME_RESIDENT, INVALID, ABSENT
  }
}

// ---------------------------------------------------------------------------
// Construction, registration, lookup
// ---------------------------------------------------------------------------

ResidencyManager::ResidencyManager(const Config& cfg) : cfg_(cfg) {}

void ResidencyManager::register_expert(ExpertId id, const BackingExtent& extent) {
  // A zero-byte extent is a caller bug that would otherwise show up as a
  // mysteriously "ready" object with no bytes in it.
  assert(extent.bytes > 0);
  Entry e;
  e.is_kv = false;
  e.mutable_bytes = false;  // the model file is authoritative, so a drop is free
  e.id = id;
  e.layer = id.layer;
  e.extent = extent;
  e.state = Residency::NVME_RESIDENT;  // the bytes are in the file by construction
  experts_[id.key()] = e;
}

void ResidencyManager::register_kv_page(KvPageId id, const BackingExtent& extent) {
  assert(extent.bytes > 0);
  Entry e;
  e.is_kv = true;
  e.mutable_bytes = true;  // a KV page off RAM has to be written back
  e.kv = id;
  e.layer = id.layer;
  e.extent = extent;
  e.state = Residency::NVME_RESIDENT;
  kv_[id.key()] = e;
}

ResidencyManager::Entry* ResidencyManager::find(ExpertId id) {
  auto it = experts_.find(id.key());
  return it == experts_.end() ? nullptr : &it->second;
}
const ResidencyManager::Entry* ResidencyManager::find(ExpertId id) const {
  auto it = experts_.find(id.key());
  return it == experts_.end() ? nullptr : &it->second;
}
ResidencyManager::Entry* ResidencyManager::find_kv(KvPageId id) {
  auto it = kv_.find(id.key());
  return it == kv_.end() ? nullptr : &it->second;
}
const ResidencyManager::Entry* ResidencyManager::find_kv(KvPageId id) const {
  auto it = kv_.find(id.key());
  return it == kv_.end() ? nullptr : &it->second;
}

// ---------------------------------------------------------------------------
// The single transition point
// ---------------------------------------------------------------------------

TransitionEvent ResidencyManager::make_event(const Entry& e, Residency from,
                                             Residency to, const char* cause) const {
  TransitionEvent ev;
  ev.is_kv = e.is_kv;
  ev.expert = e.id;
  ev.kv = e.kv;
  ev.from = from;
  ev.to = to;
  ev.cause = cause;
  ev.layer = e.layer;
  ev.tick = tick_;
  ev.now_ns = cfg_.now_ns ? cfg_.now_ns() : 0;
  return ev;
}

void ResidencyManager::set_state(Entry& e, Residency to, const char* cause) {
  const Residency from = e.state;
  e.state = to;
  ++transitions_;
  // A transition is the only way an entry becomes tick-relevant on its own:
  // advance() and the prefetch TTL both act on state, so anything that just
  // changed state must be visited by tick() until it goes quiet again.
  mark_dirty(e);
  if (!cfg_.telemetry) return;
  cfg_.telemetry->on_transition(make_event(e, from, to, cause));
}

void ResidencyManager::touch(Entry& e) {
  ++e.use_count;
  e.last_use_tick = tick_;
}

void ResidencyManager::drop_ram(Entry& e) {
  if (e.ram != nullptr && cfg_.arena != nullptr) cfg_.arena->free(e.ram);
  e.ram = nullptr;
}

// ---------------------------------------------------------------------------
// The two hops
// ---------------------------------------------------------------------------

void ResidencyManager::start_nvme_read(Entry& e, IoClass io, const char* cause) {
  if (cfg_.transfer == nullptr || cfg_.arena == nullptr) {
    set_state(e, Residency::INVALID, "read_failed");
    return;
  }
  if (e.ram == nullptr) e.ram = cfg_.arena->alloc(e.extent.bytes);
  if (e.ram == nullptr) {
    // No staging: the bytes cannot be fetched this tick. That is a not-ready, and
    // it is reported as a fault rather than papered over with a blocking read.
    set_state(e, Residency::INVALID, "read_failed");
    return;
  }
  e.load = cfg_.transfer->submit_nvme_read(e.ram, e.extent.offset, e.extent.bytes, io);
  set_state(e, Residency::LOADING_NVME, cause);
}

void ResidencyManager::start_h2d(Entry& e) {
  if (cfg_.slots == nullptr || cfg_.transfer == nullptr || e.ram == nullptr) return;
  const SlotHandle s = cfg_.slots->acquire(e.layer, e.is_kv ? 0u : e.id.index,
                                          e.extent.bytes);
  // No room means the object stays in RAM and reports not-ready. C9 does not evict
  // to make room -- that is C12's admission decision, and C9's lever for it is
  // `eviction_candidates()`. It does not wait, either.
  if (s == kNoSlot) return;
  e.slot = s;
  e.up = cfg_.transfer->submit_h2d(cfg_.slots->vram_ptr(s), e.ram, e.extent.bytes,
                                   e.demand_pending ? IoClass::DEMAND : IoClass::PREFETCH);
  set_state(e, Residency::LOADING_RAM, "h2d_issued");
}

void ResidencyManager::follow_want(Entry& e) {
  // One rule, so that no state has to special-case resumption: whenever bytes land
  // in a tier below the one that is still wanted, the next hop starts -- and when
  // they land *in* the tier that was wanted, the interest is satisfied and the
  // demand stops being outstanding.
  if (!e.want_active) return;
  if (tier_reached(e.state, e.want)) {
    // Without this the RAM hop would be a trap: a `request(id, RAM)` completes at
    // RAM_RESIDENT, nothing would clear `demand_pending`, C21 would record a
    // deadline miss for a demand that was in fact served, and C10's prefetch would
    // be refused for that object for the rest of the run -- i.e. exactly the hot
    // experts the predictor exists to prefetch. The header already promised that an
    // abandoned prefetch does not keep climbing tiers after it lands in RAM; this is
    // the mechanism that makes that promise true.
    e.want_active = false;
    e.demand_pending = false;
    e.deadline_tick = -1;
    e.miss_reported = false;
    return;
  }
  if (e.state == Residency::NVME_RESIDENT) {
    start_nvme_read(e, e.demand_pending ? IoClass::DEMAND : IoClass::PREFETCH,
                    e.demand_pending ? "demand_miss" : "prefetch_miss");
  } else if (e.state == Residency::RAM_RESIDENT && e.want == KVTier::VRAM) {
    start_h2d(e);
  }
}

// ---------------------------------------------------------------------------
// The per-entry advance (only ever called from tick())
// ---------------------------------------------------------------------------

void ResidencyManager::advance(Entry& e) {
  switch (e.state) {
    case Residency::LOADING_NVME: {
      const TransferState st = cfg_.transfer->poll(e.load);
      if (st == TransferState::DONE) {
        set_state(e, Residency::RAM_RESIDENT, "nvme_read_done");
        follow_want(e);  // a demand that wants VRAM continues straight into the H2D
      } else if (st == TransferState::FAILED || st == TransferState::CANCELLED) {
        // The bytes were not fetched. The record stays, the state does not claim a
        // residency, and the next demand re-reads from the same extent.
        set_state(e, Residency::INVALID, "read_failed");
      }
      break;
    }
    case Residency::LOADING_RAM: {
      const TransferState st = cfg_.transfer->poll(e.up);
      if (st == TransferState::DONE) {
        // Issued is not ready. The slot's load event has to have fired before this
        // is allowed to become a residency at all -- a transfer that completed
        // without a bounded event behind it is not proof that the copy is visible.
      if (e.slot != kNoSlot && cfg_.slots != nullptr &&
          cfg_.slots->load_event_fired(e.slot)) {
        e.ready_tick = (int64_t)tick_;
        e.want_active = false;
        e.deadline_tick = -1;
        e.miss_reported = false;
        // Read the flag before clearing it: it decides which of the two resident
        // states this becomes. Clearing it here is what makes `demand_pending` mean
        // "a demand is outstanding", not "this object was demanded once". Left set,
        // `prefetch()` would refuse every expert that is actually hot -- the exact
        // opposite of what the predictor needs.
        const bool was_demand = e.demand_pending;
        set_state(e, was_demand ? Residency::VRAM_RESIDENT : Residency::PREFETCHED,
                  was_demand ? "event_fired" : "prefetch_landed");
        e.demand_pending = false;
      }
      } else if (st == TransferState::FAILED || st == TransferState::CANCELLED) {
        if (e.slot != kNoSlot && cfg_.slots != nullptr) cfg_.slots->release(e.slot);
        e.slot = kNoSlot;
        set_state(e, Residency::RAM_RESIDENT, "h2d_failed");
      }
      break;
    }
    case Residency::EVICTING_VRAM: {
      // Mutable bytes: the VRAM copy may be newer than the staging copy, so the drop
      // waits for the D2H that carries the update down. A failure there would discard
      // a page's contents, which is a fault (INVALID) and not a quieter state.
      if (e.down != kNoTransfer && cfg_.transfer != nullptr) {
        const TransferState dst = cfg_.transfer->poll(e.down);
        if (dst == TransferState::FAILED || dst == TransferState::CANCELLED) {
          set_state(e, Residency::INVALID, "writeback_failed");
          break;
        }
        if (dst != TransferState::DONE) break;
        e.down = kNoTransfer;
      }
      if (e.slot == kNoSlot ||
          (cfg_.slots != nullptr && cfg_.slots->released(e.slot))) {
        e.slot = kNoSlot;
        set_state(e, e.ram != nullptr ? Residency::RAM_RESIDENT : Residency::NVME_RESIDENT,
                  "evict_done");
        follow_want(e);  // a demand that arrived mid-eviction resumes here
      }
      break;
    }
    case Residency::EVICTING_RAM: {
      if (e.mutable_bytes) {
        const TransferState st = cfg_.transfer->poll(e.out);
        if (st == TransferState::DONE) {
          drop_ram(e);
          set_state(e, Residency::NVME_RESIDENT, "writeback_done");
          follow_want(e);
        } else if (st == TransferState::FAILED || st == TransferState::CANCELLED) {
          set_state(e, Residency::INVALID, "writeback_failed");
        }
      } else {
        // Immutable bytes: the file is still authoritative, so dropping staging is
        // the whole of the eviction. No write-back is issued, ever -- reads beat
        // speculative writes.
        drop_ram(e);
        set_state(e, Residency::NVME_RESIDENT, "evict_done");
        follow_want(e);
      }
      break;
    }
    default:
      break;
  }
}

void ResidencyManager::scan_deadlines(Entry& e) {
  if (e.miss_reported || !e.demand_pending || e.deadline_tick < 0) return;
  if (ready_entry(e)) return;
  if ((int64_t)tick_ <= e.deadline_tick) return;
  e.miss_reported = true;
  ++deadline_misses_;
  if (cfg_.telemetry == nullptr) return;
  cfg_.telemetry->on_deadline_missed(make_event(e, e.state, e.state, "deadline_missed"),
                                     (uint64_t)((int64_t)tick_ - e.deadline_tick));
}

void ResidencyManager::preempt_prefetch() {
  // Throughput never wins against the current request (C10): a speculative read in
  // flight is cancelled so the demand read gets the bandwidth and the deadline.
  // Nothing is lost -- the bytes are still in the backing store, and the object is
  // simply no longer in flight, so a later prefetch re-issues it.
  if (cfg_.transfer == nullptr) return;
  for (auto& it : experts_) {
    Entry& e = it.second;
    if (e.demand_pending || e.state != Residency::LOADING_NVME || !e.want_active) continue;
    if (cfg_.transfer->cancel(e.load)) {
      e.load = kNoTransfer;
      ++preempted_;
      set_state(e, Residency::NVME_RESIDENT, "prefetch_cancelled");
    }
  }
  for (auto& it : kv_) {
    Entry& e = it.second;
    if (e.demand_pending || e.state != Residency::LOADING_NVME || !e.want_active) continue;
    if (cfg_.transfer->cancel(e.load)) {
      e.load = kNoTransfer;
      ++preempted_;
      set_state(e, Residency::NVME_RESIDENT, "prefetch_cancelled");
    }
  }
}

// ---------------------------------------------------------------------------
// tick -- the per-layer advance
// ---------------------------------------------------------------------------

void ResidencyManager::begin_layer(uint32_t layer) {
  assert(layer >= layer_ && "layers advance forwards; a backwards step means the "
                            "caller is not driving tick() once per layer");
  layer_ = layer;
}

void ResidencyManager::tick() {
  const uint64_t t0 = cfg_.now_ns ? cfg_.now_ns() : 0;
  ++tick_calls_;
  evict_candidates_.clear();
  kv_evict_candidates_.clear();

  // The dirty-set advance (Phase 36): tick() visits only entries whose state can
  // actually change here -- in flight, deadline running, or a PREFETCHED copy
  // aging toward its TTL. A quiet entry (NVME/RAM/VRAM resident, no deadline,
  // not speculative) cannot move on its own and is re-marked the moment any
  // action touches it, because every transition and every armed deadline passes
  // through mark_dirty(). The observable contract -- transitions, deadline
  // misses, preemptions, eviction candidates, C19 -- is unchanged; only the
  // visit count drops from O(registry) to O(active).
  auto is_quiet = [](const Entry& e) {
    return e.deadline_tick < 0 && e.state != Residency::PREFETCHED &&
           e.state != Residency::LOADING_NVME && e.state != Residency::LOADING_RAM &&
           e.state != Residency::EVICTING_VRAM && e.state != Residency::EVICTING_RAM;
  };
  {
    std::vector<ExpertId> next;
    next.reserve(dirty_experts_.size());
    for (const ExpertId id : dirty_experts_) {
      Entry* e = find(id);
      if (e == nullptr) continue;   // reset() dropped it
      ++entry_visits_;
      advance(*e);
      scan_deadlines(*e);
      if (e->state == Residency::PREFETCHED && e->ready_tick >= 0 &&
          (int64_t)tick_ - e->ready_tick >= cfg_.prefetch_ttl_ticks) {
        evict_candidates_.push_back(e->id);
      }
      if (!is_quiet(*e)) next.push_back(id);
      else dirty_expert_set_.erase(id.key());
    }
    dirty_experts_.swap(next);
  }
  {
    std::vector<KvPageId> next;
    next.reserve(dirty_kv_.size());
    for (const KvPageId id : dirty_kv_) {
      Entry* e = find_kv(id);
      if (e == nullptr) continue;
      ++entry_visits_;
      advance(*e);
      scan_deadlines(*e);
      if (e->state == Residency::PREFETCHED && e->ready_tick >= 0 &&
          (int64_t)tick_ - e->ready_tick >= cfg_.prefetch_ttl_ticks) {
        kv_evict_candidates_.push_back(e->kv);
      }
      if (!is_quiet(*e)) next.push_back(id);
      else dirty_kv_set_.erase(id.key());
    }
    dirty_kv_.swap(next);
  }
  if (cfg_.now_ns) tick_ns_ += cfg_.now_ns() - t0;
  ++tick_;
}

// ---------------------------------------------------------------------------
// ready -- the only question a kernel asks
// ---------------------------------------------------------------------------

bool ResidencyManager::ready_entry(const Entry& e) const {
  if (!is_vram_state(e.state)) return false;
  if (e.slot == kNoSlot || cfg_.slots == nullptr) return false;
  return cfg_.slots->load_event_fired(e.slot);
}

bool ResidencyManager::ready(ExpertId id) const {
  const Entry* e = find(id);
  return e != nullptr && ready_entry(*e);
}

bool ResidencyManager::ready_kv(KvPageId id) const {
  const Entry* e = find_kv(id);
  return e != nullptr && ready_entry(*e);
}

// ---------------------------------------------------------------------------
// The C9 API
// ---------------------------------------------------------------------------

Residency ResidencyManager::locate(ExpertId id) const {
  const Entry* e = find(id);
  return e != nullptr ? e->state : Residency::ABSENT;
}

Residency ResidencyManager::locate_kv(KvPageId id) const {
  const Entry* e = find_kv(id);
  return e != nullptr ? e->state : Residency::ABSENT;
}

void ResidencyManager::request(ExpertId id, KVTier tier) {
  Entry* e = find(id);
  if (e == nullptr) return;  // unregistered: nothing enqueued, locate() says ABSENT
  e->demand_pending = true;
  if ((uint8_t)tier > (uint8_t)e->want) e->want = tier;
  e->want_active = true;
  touch(*e);

  if (ready_entry(*e)) {
    // Already reachable in VRAM. A speculative copy becomes a real residency here
    // and nowhere else: something asked for it, so it is no longer speculative.
    if (e->state == Residency::PREFETCHED) set_state(*e, Residency::VRAM_RESIDENT, "consumed");
    e->want_active = false;
    e->demand_pending = false;  // satisfied on the spot: nothing is outstanding now
    e->deadline_tick = -1;
    e->miss_reported = false;
    return;
  }

  // A live deadline makes the entry tick-relevant even if no transfer starts
  // (RAM_RESIDENT wanting a busy pool, for instance): scan_deadlines must see it.
  if (e->deadline_tick < 0) {
    e->deadline_tick = (int64_t)tick_ + cfg_.demand_deadline_ticks;
    mark_dirty(*e);
  }

  switch (e->state) {
    case Residency::NVME_RESIDENT:
    case Residency::INVALID:
      preempt_prefetch();
      start_nvme_read(*e, IoClass::DEMAND,
                      e->state == Residency::INVALID ? "reread" : "demand_miss");
      break;
    case Residency::RAM_RESIDENT:
      // RAM does not have to be enough. If the caller wants VRAM, the second hop
      // starts now; if the slot pool is full, `start_h2d` reports not-ready and
      // C19 covers the step rather than this call blocking on RAM.
      if (e->want == KVTier::VRAM) start_h2d(*e);
      break;
    case Residency::LOADING_NVME:
    case Residency::LOADING_RAM:
    case Residency::EVICTING_VRAM:
    case Residency::EVICTING_RAM:
      break;  // already moving: the recorded intent is picked up by the pipeline
    default:
      break;
  }
}

void ResidencyManager::request_kv(KvPageId id, KVTier tier) {
  Entry* e = find_kv(id);
  if (e == nullptr) return;
  e->demand_pending = true;
  if ((uint8_t)tier > (uint8_t)e->want) e->want = tier;
  e->want_active = true;
  touch(*e);

  if (ready_entry(*e)) {
    if (e->state == Residency::PREFETCHED) set_state(*e, Residency::VRAM_RESIDENT, "consumed");
    e->want_active = false;
    e->demand_pending = false;  // satisfied on the spot: nothing is outstanding now
    e->deadline_tick = -1;
    e->miss_reported = false;
    return;
  }
  if (e->deadline_tick < 0) {
    e->deadline_tick = (int64_t)tick_ + cfg_.demand_deadline_ticks;
    mark_dirty(*e);
  }

  switch (e->state) {
    case Residency::NVME_RESIDENT:
    case Residency::INVALID:
      preempt_prefetch();
      start_nvme_read(*e, IoClass::DEMAND,
                      e->state == Residency::INVALID ? "reread" : "demand_miss");
      break;
    case Residency::RAM_RESIDENT:
      if (e->want == KVTier::VRAM) start_h2d(*e);
      break;
    default:
      break;
  }
}

void ResidencyManager::prefetch(ExpertId id, KVTier tier, PrefetchPriority prio) {
  Entry* e = find(id);
  if (e == nullptr) return;
  // Demand traffic owns the object. A prefetch must not restart or redirect a load
  // that a step is already waiting on.
  if (e->demand_pending) return;
  if ((uint8_t)tier > (uint8_t)e->want) e->want = tier;
  if (ready_entry(*e)) return;
  if (e->state == Residency::LOADING_NVME || e->state == Residency::LOADING_RAM) return;
  e->want_active = true;
  // URGENT is the only priority that buys demand-class bandwidth. Anything else is
  // cancellable, which is what keeps a wrong prediction a bandwidth cost and never a
  // latency cost.
  const IoClass io =
      (prio == PrefetchPriority::URGENT) ? IoClass::DEMAND : IoClass::PREFETCH;
  switch (e->state) {
    case Residency::NVME_RESIDENT:
    case Residency::INVALID:
      start_nvme_read(*e, io, "prefetch_miss");
      break;
    case Residency::RAM_RESIDENT:
      if (e->want == KVTier::VRAM) start_h2d(*e);
      break;
    default:
      break;  // mid-transition: the intent is recorded and the pipeline sees it
  }
}

void ResidencyManager::prefetch_kv(KvPageId id, KVTier tier, PrefetchPriority prio) {
  Entry* e = find_kv(id);
  if (e == nullptr) return;
  if (e->demand_pending) return;
  if ((uint8_t)tier > (uint8_t)e->want) e->want = tier;
  if (ready_entry(*e)) return;
  if (e->state == Residency::LOADING_NVME || e->state == Residency::LOADING_RAM) return;
  e->want_active = true;
  const IoClass io =
      (prio == PrefetchPriority::URGENT) ? IoClass::DEMAND : IoClass::PREFETCH;
  switch (e->state) {
    case Residency::NVME_RESIDENT:
    case Residency::INVALID:
      start_nvme_read(*e, io, "prefetch_miss");
      break;
    case Residency::RAM_RESIDENT:
      if (e->want == KVTier::VRAM) start_h2d(*e);
      break;
    default:
      break;
  }
}

void ResidencyManager::promote(ExpertId id) {
  Entry* e = find(id);
  if (e == nullptr) return;
  // One tier, as the name says, and as a prefetch rather than a demand: a promotion
  // is a scheduling decision, and it must not turn into a step's obligation.
  switch (e->state) {
    case Residency::NVME_RESIDENT:
    case Residency::INVALID:
      prefetch(id, KVTier::RAM, PrefetchPriority::NORMAL);
      break;
    case Residency::RAM_RESIDENT:
      prefetch(id, KVTier::VRAM, PrefetchPriority::NORMAL);
      break;
    default:
      break;
  }
}

void ResidencyManager::promote_kv(KvPageId id) {
  Entry* e = find_kv(id);
  if (e == nullptr) return;
  switch (e->state) {
    case Residency::NVME_RESIDENT:
    case Residency::INVALID:
      prefetch_kv(id, KVTier::RAM, PrefetchPriority::NORMAL);
      break;
    case Residency::RAM_RESIDENT:
      prefetch_kv(id, KVTier::VRAM, PrefetchPriority::NORMAL);
      break;
    default:
      break;
  }
}

void ResidencyManager::demote(ExpertId id) {
  Entry* e = find(id);
  if (e == nullptr) return;
  switch (e->state) {
    case Residency::PINNED:
      // Pinning outranks eviction. The refusal is counted rather than silent, so a
      // policy that pins too much shows up as pressure instead of as an unexplained
      // OOM later.
      ++refused_demotions_;
      return;
    case Residency::VRAM_RESIDENT:
    case Residency::PREFETCHED: {
      e->want = KVTier::RAM;
      e->want_active = false;
      e->demand_pending = false;
      e->deadline_tick = -1;
      if (e->slot == kNoSlot || cfg_.slots == nullptr) {
        set_state(*e, Residency::RAM_RESIDENT, "evict_done");
        return;
      }
      // Mutable bytes carry their newest contents down first (see advance()). For an
      // expert the staging copy is interchangeable with the file, so nothing moves.
      if (e->mutable_bytes && e->ram != nullptr && cfg_.transfer != nullptr) {
        e->down = cfg_.transfer->submit_d2h(e->ram, cfg_.slots->vram_ptr(e->slot),
                                           e->extent.bytes, IoClass::PREFETCH);
      }
      cfg_.slots->release(e->slot);
      set_state(*e, Residency::EVICTING_VRAM, "evict");
      return;
    }
    case Residency::RAM_RESIDENT: {
      e->want = KVTier::NVME;
      e->want_active = false;
      e->demand_pending = false;
      e->deadline_tick = -1;
      if (e->ram == nullptr) {
        set_state(*e, Residency::NVME_RESIDENT, "evict_done");
        return;
      }
      if (e->mutable_bytes) {
        e->out = cfg_.transfer->submit_nvme_write(e->ram, e->extent.offset,
                                                 e->extent.bytes, IoClass::PREFETCH);
        set_state(*e, Residency::EVICTING_RAM, "evict");
      } else {
        drop_ram(*e);
        set_state(*e, Residency::NVME_RESIDENT, "evict_done");
      }
      return;
    }
    default:
      return;  // nothing to demote from here
  }
}

void ResidencyManager::demote_kv(KvPageId id) {
  Entry* e = find_kv(id);
  if (e == nullptr) return;
  switch (e->state) {
    case Residency::PINNED:
      ++refused_demotions_;
      return;
    case Residency::VRAM_RESIDENT:
    case Residency::PREFETCHED: {
      e->want = KVTier::RAM;
      e->want_active = false;
      e->demand_pending = false;
      e->deadline_tick = -1;
      if (e->slot == kNoSlot || cfg_.slots == nullptr) {
        set_state(*e, Residency::RAM_RESIDENT, "evict_done");
        return;
      }
      // Mutable bytes carry their newest contents down first (see advance()). For an
      // expert the staging copy is interchangeable with the file, so nothing moves.
      if (e->mutable_bytes && e->ram != nullptr && cfg_.transfer != nullptr) {
        e->down = cfg_.transfer->submit_d2h(e->ram, cfg_.slots->vram_ptr(e->slot),
                                           e->extent.bytes, IoClass::PREFETCH);
      }
      cfg_.slots->release(e->slot);
      set_state(*e, Residency::EVICTING_VRAM, "evict");
      return;
    }
    case Residency::RAM_RESIDENT: {
      e->want = KVTier::NVME;
      e->want_active = false;
      e->demand_pending = false;
      e->deadline_tick = -1;
      if (e->ram == nullptr) {
        set_state(*e, Residency::NVME_RESIDENT, "evict_done");
        return;
      }
      // Mutable bytes: the RAM copy is the only copy, so leaving RAM means writing
      // the page back. This is the one place C9 issues a write.
      if (e->mutable_bytes) {
        e->out = cfg_.transfer->submit_nvme_write(e->ram, e->extent.offset,
                                                 e->extent.bytes, IoClass::PREFETCH);
        set_state(*e, Residency::EVICTING_RAM, "evict");
      } else {
        drop_ram(*e);
        set_state(*e, Residency::NVME_RESIDENT, "evict_done");
      }
      return;
    }
    default:
      return;
  }
}

void ResidencyManager::pin(ExpertId id) {
  Entry* e = find(id);
  if (e == nullptr || !ready_entry(*e) || cfg_.slots == nullptr) return;
  cfg_.slots->pin(e->slot);
  set_state(*e, Residency::PINNED, "pin");
}

void ResidencyManager::unpin(ExpertId id) {
  Entry* e = find(id);
  if (e == nullptr || e->state != Residency::PINNED || cfg_.slots == nullptr) return;
  cfg_.slots->unpin(e->slot);
  set_state(*e, Residency::VRAM_RESIDENT, "unpin");
}

// ---------------------------------------------------------------------------
// The scheduler's side
// ---------------------------------------------------------------------------

bool ResidencyManager::demand(ExpertId id, KVTier tier) {
  request(id, tier);
  return ready(id);
}

bool ResidencyManager::demand_or_fallback(ExpertId id) {
  if (demand(id, KVTier::VRAM)) return true;
  // Not ready, and the kernel is about to run. The contribution is not skipped and
  // the step does not wait: C19 computes it. That is the only shape of this call
  // that is compatible with I1 and with I4 at the same time.
  if (cfg_.fallback != nullptr) cfg_.fallback->expert_unavailable(id, locate(id));
  return false;
}

bool ResidencyManager::demand_kv(KvPageId id, KVTier tier) {
  request_kv(id, tier);
  return ready_kv(id);
}

bool ResidencyManager::demand_kv_or_fallback(KvPageId id) {
  if (demand_kv(id, KVTier::VRAM)) return true;
  if (cfg_.fallback != nullptr) cfg_.fallback->kv_unavailable(id, locate_kv(id));
  return false;
}

// ---------------------------------------------------------------------------
// Teardown
// ---------------------------------------------------------------------------

void ResidencyManager::reset_entry(Entry& e) {
  if (cfg_.transfer != nullptr) {
    if (e.load != kNoTransfer) cfg_.transfer->cancel(e.load);
    if (e.up != kNoTransfer) cfg_.transfer->cancel(e.up);
    if (e.down != kNoTransfer) cfg_.transfer->cancel(e.down);
    if (e.out != kNoTransfer) cfg_.transfer->cancel(e.out);
  }
  if (e.slot != kNoSlot && cfg_.slots != nullptr) cfg_.slots->release(e.slot);
  drop_ram(e);

  const bool is_kv = e.is_kv;
  const bool mutable_bytes = e.mutable_bytes;
  const ExpertId id = e.id;
  const KvPageId kv = e.kv;
  const uint32_t layer = e.layer;
  const BackingExtent extent = e.extent;
  const Residency before = e.state;

  e = Entry();
  e.is_kv = is_kv;
  e.mutable_bytes = mutable_bytes;
  e.id = id;
  e.kv = kv;
  e.layer = layer;
  e.extent = extent;
  e.state = Residency::NVME_RESIDENT;
  // reset() is a teardown between runs, not a runtime transition, but it still moves
  // an object from one state to another -- so it still emits, and the "every
  // transition emits an event" rule has no exception to remember.
  if (before != Residency::NVME_RESIDENT) {
    ++transitions_;
    if (cfg_.telemetry != nullptr) {
      cfg_.telemetry->on_transition(make_event(e, before, Residency::NVME_RESIDENT, "reset"));
    }
  }
}

void ResidencyManager::reset() {
  for (auto& it : experts_) reset_entry(it.second);
  for (auto& it : kv_) reset_entry(it.second);
  evict_candidates_.clear();
  kv_evict_candidates_.clear();
  dirty_experts_.clear();
  dirty_expert_set_.clear();
  dirty_kv_.clear();
  dirty_kv_set_.clear();
  layer_ = 0;
  tick_ = 0;
}

}  // namespace knj
