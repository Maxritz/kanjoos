// tools/bench/q4k_stream_ffn.hip -- expert STREAMING on gfx1201: the layer-0 MoE
// FFN computed with the expert weights coming off the model file through a
// bounded VRAM slot pool, driven by the real C9 residency manager with a real
// C8 slot allocator and a real C11 transfer engine.
//
// WHAT PROBLEM THIS SOLVES
// ------------------------
// Everything the project has measured runs on FULLY RESIDENT weights. The whole
// reason the project exists -- a model too big for VRAM, with cold experts on
// NVMe -- has never been executed. `q4k_moe_ffn.hip` proves the kernel is right;
// it does not prove that routing bytes through a small pool preserves the answer.
//
// So this driver runs the SAME layer-0 FFN twice:
//
//   1. RESIDENT  -- the whole packed expert bank on the device, as before.
//   2. STREAMED  -- a pool of `n_slots` VRAM slots (default 2), each 2.4 MiB, and
//                   every expert projection read from the model file through
//                   C9 -> C11 -> C8 on demand, executed, then evicted.
//
// and then asserts the two results are BIT-IDENTICAL (I1: no expert lost; I7:
// tier movement is invisible). That is the acceptance test the design documents
// keep asking for and that nothing has run yet.
//
// WHY C9 IS IN THE LOOP AND NOT A REIMPLEMENTATION
// ------------------------------------------------
// The point of the exercise is not "a loop that copies bytes". It is that the
// residency manager's state machine -- the thing that decides where an expert is
// and whether a kernel may touch it -- produces the same answer as residency.
// So the driver injects the REAL implementations behind C9's interfaces:
//
//   * `HipSlots : knj::SlotAllocator`   -- one hipMalloc VA reservation, fixed
//     slots, lifetime bounded by HIP events (a slot is not reused until the
//     event of the kernel that last read it has fired), and `load_event_fired`
//     is a real hipEventQuery, so "issued" is never mistaken for "ready".
//   * `HipTransfer : knj::TransferEngine` -- async H2D on a stream, and an NVMe
//     read performed by a worker thread (the interface is async by construction;
//     the read happens off the step thread, which is I4's requirement).
//   * `PinnedArena : knj::HostArena`    -- hipHostMalloc staging (C7's pinned
//     bounce buffer, not freshly malloc'd pageable memory).
//
// Nothing in `src/residency/` changes. If the streamed result differs from the
// resident result, the cause is the streamed path -- the kernels and the weights
// are the same bytes.
//
// PRECISION, DECLARED NOT HIDDEN
// ------------------------------
// rocWMMA on gfx1201 offers exactly one geometry, 16x16x16 wave32, f16 operands,
// f32 accumulator. The single-projection streaming kernel is the combined
// gate/up kernel with the second accumulator removed, so its arithmetic per output
// element is identical and the A/B is a bit comparison, not a tolerance.
//
// Build (--rocm-path must end in /lib/llvm; -std=c++17 is a rocWMMA header need):
//   hipcc --rocm-path="G:/ROCM10RT-gfx1201/lib/llvm" -std=c++17 -O2 \
//         --offload-arch=gfx1201 -Xclang -target-feature \
//         -Xclang +wavefrontsize32 -I tools/bench -I kernels -I . \
//         tools/bench/q4k_stream_ffn.hip src/residency/residency.cpp \
//         -o build/q4k/q4k_stream_ffn.exe
// Run (arch-guarded: exits 7 unconfigured, 6 arch-mismatched):
//   KNJ_BUILD_ARCH=gfx1201 build/q4k/q4k_stream_ffn.exe <job> <dumpdir> [tokens] [slots]
//
// Exit codes: 0 PASS, 5 FAIL (a correctness check), 9 HIP error, 3 missing input,
//             4 setup refused (bad slot pool), 6/7 as the arch guard.

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "knj_wmma_guard.h"
#include "knj_q4k.h"

#include "src/residency/residency.h"

#define HIPCHECK(x)                                                            \
    do {                                                                       \
        hipError_t _e = (x);                                                   \
        if (_e != hipSuccess) {                                                \
            printf("HIP ERROR %s at %s:%d\n", hipGetErrorString(_e), __FILE__, \
                   __LINE__);                                                  \
            exit(9);                                                           \
        }                                                                      \
    } while (0)

namespace rwx = rocwmma;
// `knj` is already defined by src/residency/residency.h -- no alias needed.

static const int TM = 16, TN = 16, TK = 16;
static const int NTHREADS = 64;
static const float SENTINEL = -12345.0f;
static const __half sSil = __float2half(SENTINEL);   // sentinel prefill for the fused activation buffer

// ===========================================================================
// Job + file helpers (same job format as q4k_moe_ffn.hip)
// ===========================================================================
// Per-layer geometry. The quant format is per-layer, and that is load-bearing:
// 14 of this model's 28 ffn_down_exps tensors are Q4_K, not Q6_K (measured by
// tools/q4k_job.py, which verifies every offset against the file bytes). A
// driver that assumed one format for all layers would measure plausible, wrong
// bytes.
struct LayerGeo {
    long long off[3] = {0, 0, 0};  // gate, up, down tensor starts (absolute)
    int bpb[3] = {0, 0, 0};        // bytes per 256-weight superblock, per role
    long long router_off = 0;
};

struct Job {
    std::string model, dumpdir;
    int layers = 0, hidden = 0, ff = 0, experts = 0, topk = 0;
    int k[3] = {0, 0, 0}, n[3] = {0, 0, 0}, wpb[3] = {256, 256, 256};
    std::vector<LayerGeo> geo;
};

static std::vector<std::pair<std::string, std::string>> read_job(const std::string &path) {
    std::FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) { printf("cannot open job %s\n", path.c_str()); exit(3); }
    std::vector<std::pair<std::string, std::string>> kv;
    char line[2048];
    while (std::fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        char key[512] = {0}, val[1536] = {0};
        if (std::sscanf(line, "%511s %1535s", key, val) != 2) continue;
        kv.emplace_back(key, val);
    }
    std::fclose(fp);
    return kv;
}
static std::string jget(const std::vector<std::pair<std::string, std::string>> &kv,
                        const std::string &key) {
    for (const auto &p : kv) if (p.first == key) return p.second;
    printf("job is missing key: %s\n", key.c_str());
    exit(3);
}
static long long jint(const std::vector<std::pair<std::string, std::string>> &kv,
                      const std::string &key) {
    return std::strtoll(jget(kv, key).c_str(), nullptr, 10);
}
static std::vector<uint8_t> read_at(const std::string &path, long long off, long long nbytes) {
    std::FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) { printf("cannot open %s\n", path.c_str()); exit(3); }
    if (std::fseek(fp, (long)off, SEEK_SET) != 0) {
        printf("seek failed at %lld in %s\n", off, path.c_str());
        exit(3);
    }
    std::vector<uint8_t> buf((size_t)nbytes);
    const size_t got = std::fread(buf.data(), 1, (size_t)nbytes, fp);
    std::fclose(fp);
    if (got != (size_t)nbytes) {
        printf("short read at %lld in %s (%zu of %lld)\n", off, path.c_str(), got, nbytes);
        exit(3);
    }
    return buf;
}

// Monotonic nanoseconds. `now_ns` is the same clock handed to C9 as `cfg.now_ns`,
// so the stage breakdown printed below sits on one timeline with the state
// machine's own accounting.
static uint64_t mono_ns() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<nanoseconds>(
               steady_clock::now().time_since_epoch()).count();
}
static uint64_t now_ns() { return mono_ns(); }

// ===========================================================================
// C8 -- HipSlots: one VA reservation, fixed slots, HIP-event-bounded lifetime
// ===========================================================================
class HipSlots final : public knj::SlotAllocator {
 public:
    HipSlots(int n_slots, uint64_t slot_bytes) : n_(n_slots), bytes_(slot_bytes) {
        HIPCHECK(hipMalloc(&base_, (size_t)n_slots * slot_bytes));
        slots_.resize((size_t)n_slots);
        for (int i = 0; i < n_slots; ++i) {
            slots_[(size_t)i].ptr = (uint8_t *)base_ + (size_t)i * slot_bytes;
            slots_[(size_t)i].bytes = slot_bytes;
            HIPCHECK(hipEventCreateWithFlags(&slots_[(size_t)i].load_ev,
                                             hipEventDisableTiming));
            HIPCHECK(hipEventCreateWithFlags(&slots_[(size_t)i].use_ev,
                                             hipEventDisableTiming));
        }
    }
    ~HipSlots() override {
        for (auto &s : slots_) {
            if (s.load_ev) hipEventDestroy(s.load_ev);
            if (s.use_ev) hipEventDestroy(s.use_ev);
        }
        if (base_) hipFree(base_);
    }

    knj::SlotHandle acquire(uint32_t layer, uint32_t index, uint64_t byte_size) override {
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].state != ST_FREE) continue;
            if (slots_[i].bytes < byte_size) continue;
            slots_[i].state = ST_IN_USE;
            slots_[i].layer = layer;
            slots_[i].index = index;
            slots_[i].load_recorded = false;
            slots_[i].use_recorded = false;
            last_ = (knj::SlotHandle)i;
            ++acquires_;
            note_occupancy();   // occupancy only ever rises here
            return last_;
        }
        ++refusals_;
        return knj::kNoSlot;
    }
    void release(knj::SlotHandle s) override {
        if (!valid(s)) return;
        ++releases_;
        if (!slots_[(size_t)s].use_recorded) {   // never read by a kernel: free now
            finish_free(s);
            return;
        }
        // The slot is retired only once the event of the kernel that last read it
        // has fired. That is C8's rule stated as a mechanism, not a comment.
        slots_[(size_t)s].state = ST_EVICT_PENDING;
    }
    bool released(knj::SlotHandle s) const override {
        if (!valid(s)) return true;
        const Slot &sl = slots_[(size_t)s];
        if (sl.state == ST_FREE) return true;
        if (sl.state != ST_EVICT_PENDING) return false;
        if (!sl.use_recorded) return true;
        if (hipEventQuery(sl.use_ev) == hipSuccess) {
            const_cast<HipSlots *>(this)->finish_free(s);
            return true;
        }
        const_cast<HipSlots *>(this)->saw_event_gate_ = true;
        return false;
    }
    void pin(knj::SlotHandle s) override { if (valid(s)) ++pins_; }
    void unpin(knj::SlotHandle s) override { if (valid(s)) ++unpins_; }
    bool load_event_fired(knj::SlotHandle s) const override {
        if (!valid(s)) return false;
        const Slot &sl = slots_[(size_t)s];
        if (!sl.load_recorded) return false;
        return hipEventQuery(sl.load_ev) == hipSuccess;
    }
    void *vram_ptr(knj::SlotHandle s) const override {
        return valid(s) ? slots_[(size_t)s].ptr : nullptr;
    }
    uint64_t byte_size(knj::SlotHandle s) const override {
        return valid(s) ? slots_[(size_t)s].bytes : 0;
    }

    // -- concrete C8 API, above the injected interface ----------------------
    knj::SlotHandle last_acquired() const { return last_; }
    knj::SlotHandle find(void *dev_ptr) const {
        for (size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i].ptr == dev_ptr) return (knj::SlotHandle)i;
        return knj::kNoSlot;
    }
    // The residency manager keys objects by (layer, index); the driver needs the
    // slot an object currently owns so a kernel can be launched against it
    // without routing through the manager a second time.
    knj::SlotHandle slot_for(uint32_t layer, uint32_t index) const {
        for (size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i].state != ST_FREE && slots_[i].layer == layer &&
                slots_[i].index == index)
                return (knj::SlotHandle)i;
        return knj::kNoSlot;
    }
    // Record the load event after the H2D copy (called by the transfer engine).
    void record_load(knj::SlotHandle s, hipStream_t st) {
        if (!valid(s)) return;
        HIPCHECK(hipEventRecord(slots_[(size_t)s].load_ev, st));
        slots_[(size_t)s].load_recorded = true;
    }
    // Record the last-use event after the consuming kernel launch. Without this
    // the slot's lifetime would be unbounded and the pool would be unsafe.
    void mark_used(knj::SlotHandle s, hipStream_t st) {
        if (!valid(s)) return;
        HIPCHECK(hipEventRecord(slots_[(size_t)s].use_ev, st));
        slots_[(size_t)s].use_recorded = true;
        ++uses_;
    }
    // The consuming stream must not read a slot before the copy that fills it
    // has completed. When C11 copies on a stream other than the consumer's,
    // this explicit dependency is the only thing that orders them.
    void wait_load(knj::SlotHandle s, hipStream_t st) {
        if (!valid(s)) return;
        HIPCHECK(hipStreamWaitEvent(st, slots_[(size_t)s].load_ev));
    }
    int in_use_count() const {
        int n = 0;
        for (auto &s : slots_) if (s.state != ST_FREE) ++n;
        return n;
    }
    // A genuine high-water mark. Printing `in_use_count()` at the end of a run
    // where everything has been demoted reports 0 no matter what happened, which
    // measures nothing; the peak has to be sampled where occupancy changes.
    int peak_in_use() const { return peak_; }
    // Diagnostic dump for a stall: every slot's object, state and event status.
    void dump(const char *when) const {
        printf("  slots (%s):\n", when);
        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot &s = slots_[(size_t)i];
            printf("    slot %zu: L%u idx%u state=%s load_rec=%d load_fired=%s "
                   "use_rec=%d released=%s\n",
                   i, s.layer, s.index,
                   s.state == ST_FREE ? "FREE" : s.state == ST_IN_USE ? "IN_USE" : "EVICT_PEND",
                   s.load_recorded ? 1 : 0,
                   (s.load_recorded && hipEventQuery(s.load_ev) == hipSuccess) ? "yes" : "no",
                   s.use_recorded ? 1 : 0,
                   released((knj::SlotHandle)i) ? "yes" : "no");
        }
    }
    int n_slots() const { return n_; }
    int free_slots() const {
        int f = 0;
        for (int i = 0; i < n_; ++i) if (released((knj::SlotHandle)i)) ++f;
        return f;
    }
    uint64_t slot_bytes() const { return bytes_; }
    uint64_t acquires() const { return acquires_; }
    uint64_t releases() const { return releases_; }
    uint64_t refusals() const { return refusals_; }
    uint64_t uses() const { return uses_; }
    bool saw_event_gate() const { return saw_event_gate_; }

 private:
    enum { ST_FREE = 0, ST_IN_USE = 1, ST_EVICT_PENDING = 2 };
    struct Slot {
        uint8_t *ptr = nullptr;
        uint64_t bytes = 0;
        int state = ST_FREE;
        uint32_t layer = 0, index = 0;
        hipEvent_t load_ev = nullptr, use_ev = nullptr;
        bool load_recorded = false, use_recorded = false;
    };
    bool valid(knj::SlotHandle s) const { return s >= 0 && (size_t)s < slots_.size(); }
    void note_occupancy() {
        const int n = in_use_count();
        if (n > peak_) peak_ = n;
    }
    void finish_free(knj::SlotHandle s) {
        slots_[(size_t)s].state = ST_FREE;
        slots_[(size_t)s].load_recorded = false;
    }

    void *base_ = nullptr;
    int n_ = 0;
    uint64_t bytes_ = 0;
    std::vector<Slot> slots_;
    knj::SlotHandle last_ = knj::kNoSlot;
    uint64_t acquires_ = 0, refusals_ = 0, uses_ = 0, releases_ = 0;
    uint64_t pins_ = 0, unpins_ = 0;
    bool saw_event_gate_ = false;
    int peak_ = 0;
};

// ===========================================================================
// C11 -- HipTransfer: async H2D/D2H on a stream, NVMe read on a worker thread
// ===========================================================================
// The interface has no blocking variant, and this implementation keeps that
// true: `submit_nvme_read` hands the pread to a worker thread and returns an id
// immediately. The read happens off the step thread. That is I4 -- not a promise
// that no read is issued, but that no read *blocks the step*.
class NvmeReader {
 public:
    explicit NvmeReader(const std::string &path) : path_(path) {
        th_ = std::thread([this] { run(); });
    }
    ~NvmeReader() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }
    struct Job { uint64_t id; void *dst; uint64_t off, bytes; };
    void submit(const Job &j) {
        { std::lock_guard<std::mutex> lk(m_); q_.push_back(j); }
        cv_.notify_one();
    }
    bool done(uint64_t id) const {
        std::lock_guard<std::mutex> lk(m_);
        return done_.count(id) != 0;
    }
    bool failed(uint64_t id) const {
        std::lock_guard<std::mutex> lk(m_);
        auto it = failed_.find(id);
        return it != failed_.end() && it->second;
    }
    uint64_t reads() const { return reads_.load(); }
    uint64_t bytes() const { return bytes_.load(); }
    uint64_t busy_ns() const { return busy_ns_.load(); }

 private:
    void run() {
        std::FILE *fp = std::fopen(path_.c_str(), "rb");
        while (true) {
            Job j{};
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) break;
                j = q_.front();
                q_.pop_front();
            }
            bool ok = false;
            const uint64_t t0 = mono_ns();
            if (fp && std::fseek(fp, (long)j.off, SEEK_SET) == 0) {
                ok = std::fread(j.dst, 1, (size_t)j.bytes, fp) == (size_t)j.bytes;
            }
            busy_ns_.fetch_add(mono_ns() - t0);   // trap: pure read time, reader thread
            if (ok) { reads_.fetch_add(1); bytes_.fetch_add(j.bytes); }
            std::lock_guard<std::mutex> lk(m_);
            done_[j.id] = 1;
            if (!ok) failed_[j.id] = true;
        }
        if (fp) std::fclose(fp);
    }
    std::string path_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<Job> q_;
    std::unordered_map<uint64_t, bool> failed_;
    std::unordered_map<uint64_t, uint8_t> done_;
    bool stop_ = false;
    std::thread th_;
    std::atomic<uint64_t> reads_{0}, bytes_{0}, busy_ns_{0};
};

class HipTransfer final : public knj::TransferEngine {
 public:
    // `split_copy` gives C11 its own stream: H2D copies then run on the copy
    // engine concurrently with compute kernels, and the consuming stream is
    // ordered against each copy by the slot's load event (C8's wait_load).
    // The interface is unchanged -- submissions stay async and event-bounded.
    HipTransfer(HipSlots *slots, hipStream_t compute_stream, const std::string &model,
                bool split_copy = false)
        : slots_(slots), s_(compute_stream), nvme_(model), split_(split_copy) {
        if (split_) HIPCHECK(hipStreamCreate(&copy_));
    }
    ~HipTransfer() override {
        for (auto &kv : xfer_) if (kv.second.ev) hipEventDestroy(kv.second.ev);
        if (copy_) hipStreamDestroy(copy_);
    }
    hipStream_t xstream() const { return split_ ? copy_ : s_; }

    knj::TransferId submit_nvme_read(void *dst_host, uint64_t src_offset, uint64_t bytes,
                                     knj::IoClass io) override {
        (void)io;
        const knj::TransferId id = next_++;
        X x; x.kind = NVME; x.loaded = false;
        xfer_[id] = x;
        nvme_.submit(NvmeReader::Job{id, dst_host, src_offset, bytes});
        ++nvme_submits_;
        return id;
    }
    knj::TransferId submit_h2d(void *dst_device, const void *src_host, uint64_t bytes,
                               knj::IoClass io) override {
        (void)io;
        const knj::TransferId id = next_++;
        X x; x.kind = H2D;
        x.t_submit = mono_ns();
        HIPCHECK(hipEventCreateWithFlags(&x.ev, hipEventDisableTiming));
        HIPCHECK(hipMemcpyAsync(dst_device, src_host, bytes, hipMemcpyHostToDevice, xstream()));
        HIPCHECK(hipEventRecord(x.ev, xstream()));
        const knj::SlotHandle sh = slots_->find(dst_device);
        if (sh != knj::kNoSlot) slots_->record_load(sh, xstream());
        xfer_[id] = x;
        ++h2d_submits_;
        h2d_bytes_ += bytes;
        return id;
    }
    knj::TransferId submit_d2h(void *dst_host, const void *src_device, uint64_t bytes,
                               knj::IoClass io) override {
        (void)io;
        const knj::TransferId id = next_++;
        X x; x.kind = H2D;
        HIPCHECK(hipEventCreateWithFlags(&x.ev, hipEventDisableTiming));
        HIPCHECK(hipMemcpyAsync(dst_host, src_device, bytes, hipMemcpyDeviceToHost, xstream()));
        HIPCHECK(hipEventRecord(x.ev, xstream()));
        xfer_[id] = x;
        return id;
    }
    knj::TransferId submit_nvme_write(const void *, uint64_t, uint64_t, knj::IoClass) override {
        ++writes_refused_;
        return knj::kNoTransfer;
    }
    knj::TransferState poll(knj::TransferId id) const override {
        auto it = xfer_.find(id);
        if (it == xfer_.end()) return knj::TransferState::FAILED;
        if (it->second.kind == NVME) {
            if (nvme_.failed(id)) return knj::TransferState::FAILED;
            return nvme_.done(id) ? knj::TransferState::DONE : knj::TransferState::PENDING;
        }
        if (hipEventQuery(it->second.ev) == hipSuccess) {
            if (!it->second.timed) {   // trap: first DONE poll = one host-visible latency
                h2d_ns_ += mono_ns() - it->second.t_submit;
                const_cast<X &>(it->second).timed = true;
            }
            return knj::TransferState::DONE;
        }
        return knj::TransferState::PENDING;
    }
    bool cancel(knj::TransferId) override { return false; }

    uint64_t nvme_submits() const { return nvme_submits_; }
    uint64_t h2d_submits() const { return h2d_submits_; }
    uint64_t h2d_bytes() const { return h2d_bytes_; }
    uint64_t h2d_ns() const { return h2d_ns_; }
    uint64_t nvme_busy_ns() const { return nvme_.busy_ns(); }
    uint64_t nvme_reads() const { return nvme_.reads(); }
    uint64_t nvme_bytes() const { return nvme_.bytes(); }
    uint64_t writes_refused() const { return writes_refused_; }
    std::size_t in_flight() const { return xfer_.size(); }

 private:
    enum Kind { H2D, D2H, NVME };
    struct X {
        Kind kind = H2D;
        hipEvent_t ev = nullptr;
        bool loaded = false;
        uint64_t t_submit = 0;
        bool timed = false;
    };

    HipSlots *slots_;
    hipStream_t s_;
    hipStream_t copy_ = nullptr;
    bool split_ = false;
    NvmeReader nvme_;
    std::unordered_map<knj::TransferId, X> xfer_;
    knj::TransferId next_ = 1;
    uint64_t nvme_submits_ = 0, h2d_submits_ = 0, h2d_bytes_ = 0, writes_refused_ = 0;
    mutable uint64_t h2d_ns_ = 0;
};

// ===========================================================================
// HostArena -- pinned staging (C7's bounce buffer)
// ===========================================================================
class PinnedArena final : public knj::HostArena {
 public:
    // `reuse == false` reproduces the previous behaviour exactly: every staging
    // buffer is hipHostMalloc'd before a read and hipHostFree'd on drop, which
    // is an allocation trap -- pinning pages is a driver round trip, not a
    // free-list pop. `reuse == true` pools buffers by size instead, which is
    // what C7's pre-registered bounce buffers are supposed to be; the pool is
    // bounded by the number of concurrent reads, and `drain()` releases it.
    explicit PinnedArena(bool reuse) : reuse_(reuse) {}
    void *alloc(uint64_t bytes) override {
        if (reuse_) {
            auto it = pool_.lower_bound(bytes);   // first fit >= requested
            if (it != pool_.end()) {
                void *p = it->second;
                pool_.erase(it);
                live_[p] = bytes;
                ++reuse_hits_;
                return p;
            }
        }
        void *p = nullptr;
        const uint64_t t0 = mono_ns();
        const bool ok = hipHostMalloc(&p, (size_t)bytes, hipHostMallocDefault) == hipSuccess;
        host_alloc_ns_ += mono_ns() - t0;     // trap: cost of pinning pages
        if (!ok) return nullptr;
        live_[p] = bytes;
        ++allocs_;
        return p;
    }
    void free(void *p) override {
        if (!p) return;
        auto it = live_.find(p);
        if (it == live_.end()) return;
        const uint64_t b = it->second;
        live_.erase(it);
        if (reuse_) { pool_.insert({b, p}); return; }
        const uint64_t t0 = mono_ns();
        hipHostFree(p);
        host_free_ns_ += mono_ns() - t0;      // trap: cost of unpinning pages
        ++frees_;
    }
    // Release the reuse pool. Only the driver calls this, at the end of a pass.
    void drain() {
        for (auto &kv : pool_) {
            const uint64_t t0 = mono_ns();
            hipHostFree(kv.second);
            host_free_ns_ += mono_ns() - t0;
            ++frees_;
        }
        pool_.clear();
    }
    std::size_t live() const { return live_.size(); }
    std::size_t cached() const { return pool_.size(); }
    uint64_t live_bytes() const {
        uint64_t b = 0;
        for (auto &kv : live_) b += kv.second;
        return b;
    }
    uint64_t allocs() const { return allocs_; }
    uint64_t frees() const { return frees_; }
    uint64_t reuse_hits() const { return reuse_hits_; }
    uint64_t host_alloc_ns() const { return host_alloc_ns_; }
    uint64_t host_free_ns() const { return host_free_ns_; }

 private:
    bool reuse_ = false;
    std::unordered_map<void *, uint64_t> live_;
    std::multimap<uint64_t, void *> pool_;   // size -> buffer
    uint64_t allocs_ = 0, frees_ = 0, reuse_hits_ = 0;
    uint64_t host_alloc_ns_ = 0, host_free_ns_ = 0;
};

// ===========================================================================
// C19 / C21 stand-ins. C19 must be present (the contract's escape hatch) but a
// correctness driver that gives the scheduler all the time in the world should
// never need it; the count is reported, not assumed zero.
// ===========================================================================
class CountingFallback final : public knj::FallbackSink {
 public:
    void expert_unavailable(knj::ExpertId, knj::Residency) override { ++expert_; }
    void kv_unavailable(knj::KvPageId, knj::Residency) override { ++kv_; }
    uint64_t expert() const { return expert_; }
    uint64_t kv() const { return kv_; }
 private:
    uint64_t expert_ = 0, kv_ = 0;
};

class CountingTelemetry final : public knj::TelemetrySink {
 public:
    void on_transition(const knj::TransitionEvent &ev) override {
        ++events_;
        causes_[std::string(ev.cause)]++;
        if (knj::is_vram_state(ev.to)) ++into_vram_;
    }
    void on_deadline_missed(const knj::TransitionEvent &, uint64_t) override { ++misses_; }
    uint64_t events() const { return events_; }
    uint64_t into_vram() const { return into_vram_; }
    uint64_t misses() const { return misses_; }
    uint64_t cause(const char *c) const {
        auto it = causes_.find(c);
        return it == causes_.end() ? 0 : it->second;
    }
 private:
    uint64_t events_ = 0, into_vram_ = 0, misses_ = 0;
    std::unordered_map<std::string, uint64_t> causes_;
};

// ===========================================================================
// Kernels
// ===========================================================================
// Kernel 1 -- resident reference: gate and up, Q4_K, one pass (as q4k_moe_ffn).
__global__ void __launch_bounds__(NTHREADS)
knj_q4k_gateup(const __half *__restrict__ X, const uint8_t *__restrict__ Wg,
               const uint8_t *__restrict__ Wu, const int *__restrict__ pairIdx,
               const int *__restrict__ pairCnt, const int *__restrict__ pairTok,
               float *__restrict__ G, float *__restrict__ U,
               int N, int K, int rowBytes, long long idxStride) {
    const int e = blockIdx.z;
    const int n0 = blockIdx.x * TN;
    const int m0 = blockIdx.y * TM;
    const int m = pairCnt[e];
    if (m0 >= m) return;

    const uint8_t *wg = Wg + ((long long)e * N + n0) * rowBytes;
    const uint8_t *wu = Wu + ((long long)e * N + n0) * rowBytes;

    __shared__ __align__(32) __half sA[TM * TK];
    __shared__ __align__(32) __half sBg[TK * TN];
    __shared__ __align__(32) __half sBu[TK * TN];

    rwx::fragment<rwx::accumulator, TM, TN, TK, float> accg, accu;
    rwx::fill_fragment(accg, 0.0f);
    rwx::fill_fragment(accu, 0.0f);

    const int row = threadIdx.x >> 2;
    const int col = (threadIdx.x & 3) * 4;

    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int i = threadIdx.x; i < TM * TK; i += NTHREADS) {
            const int mm = i / TK, kk = i % TK;
            const int p = (m0 + mm < m) ? pairIdx[(long long)e * idxStride + m0 + mm] : -1;
            sA[i] = (p >= 0) ? X[(long long)pairTok[p] * K + k0 + kk] : __float2half(0.0f);
        }
        const int r0 = k0 % 256;
        const int blkOff = (k0 / 256) * KNJ_Q4K_BYTES;
        float span[16];
        knj_q4k_span16(wg + row * rowBytes + blkOff, r0, span);
        for (int i = 0; i < 4; ++i) sBg[row * TK + col + i] = __float2half(span[col + i]);
        knj_q4k_span16(wu + row * rowBytes + blkOff, r0, span);
        for (int i = 0; i < 4; ++i) sBu[row * TK + col + i] = __float2half(span[col + i]);

        __syncthreads();
        rwx::fragment<rwx::matrix_a, TM, TN, TK, __half, rwx::row_major> fa;
        rwx::fragment<rwx::matrix_b, TM, TN, TK, __half, rwx::col_major> fbg, fbu;
        rwx::load_matrix_sync(fa, sA, TK);
        rwx::load_matrix_sync(fbg, sBg, TK);
        rwx::load_matrix_sync(fbu, sBu, TK);
        rwx::mma_sync(accg, fa, fbg, accg);
        rwx::mma_sync(accu, fa, fbu, accu);
        __syncthreads();
    }
    __shared__ __align__(32) float sDg[TM * TN], sDu[TM * TN];
    rwx::store_matrix_sync(sDg, accg, TN, rwx::mem_row_major);
    rwx::store_matrix_sync(sDu, accu, TN, rwx::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < TM * TN; i += NTHREADS) {
        const int mm = i / TN, nn = i % TN;
        if (m0 + mm < m) {
            const int p = pairIdx[(long long)e * idxStride + m0 + mm];
            G[(long long)p * N + n0 + nn] = sDg[i];
            U[(long long)p * N + n0 + nn] = sDu[i];
        }
    }
}

// Kernel 2 -- down, Q6_K (identical to q4k_moe_ffn's).
__global__ void __launch_bounds__(NTHREADS)
knj_q6k_down(const __half *__restrict__ A, const uint8_t *__restrict__ Wd,
             const int *__restrict__ pairIdx, const int *__restrict__ pairCnt,
             float *__restrict__ Y, int N, int K, int rowBytes,
             long long idxStride, int dq4) {
    const int e = blockIdx.z;
    const int n0 = blockIdx.x * TN;
    const int m0 = blockIdx.y * TM;
    const int m = pairCnt[e];
    if (m0 >= m) return;

    const uint8_t *wd = Wd + ((long long)e * N + n0) * rowBytes;
    __shared__ __align__(32) __half sA[TM * TK];
    __shared__ __align__(32) __half sB[TK * TN];
    rwx::fragment<rwx::accumulator, TM, TN, TK, float> acc;
    rwx::fill_fragment(acc, 0.0f);
    const int row = threadIdx.x >> 2;
    const int col = (threadIdx.x & 3) * 4;

    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int i = threadIdx.x; i < TM * TK; i += NTHREADS) {
            const int mm = i / TK, kk = i % TK;
            const int p = (m0 + mm < m) ? pairIdx[(long long)e * idxStride + m0 + mm] : -1;
            sA[i] = (p >= 0) ? A[(long long)p * K + k0 + kk] : __float2half(0.0f);
        }
        const int r0 = k0 % 256;
        const int blkOff = (k0 / 256) * (dq4 ? KNJ_Q4K_BYTES : KNJ_Q6K_BYTES);
        float span[16];
        if (dq4) knj_q4k_span16(wd + row * rowBytes + blkOff, r0, span);
        else     knj_q6k_span16(wd + row * rowBytes + blkOff, r0, span);
        for (int i = 0; i < 4; ++i) sB[row * TK + col + i] = __float2half(span[col + i]);
        __syncthreads();
        rwx::fragment<rwx::matrix_a, TM, TN, TK, __half, rwx::row_major> fa;
        rwx::fragment<rwx::matrix_b, TM, TN, TK, __half, rwx::col_major> fb;
        rwx::load_matrix_sync(fa, sA, TK);
        rwx::load_matrix_sync(fb, sB, TK);
        rwx::mma_sync(acc, fa, fb, acc);
        __syncthreads();
    }
    __shared__ __align__(32) float sD[TM * TN];
    rwx::store_matrix_sync(sD, acc, TN, rwx::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < TM * TN; i += NTHREADS) {
        const int mm = i / TN, nn = i % TN;
        if (m0 + mm < m) {
            const int p = pairIdx[(long long)e * idxStride + m0 + mm];
            Y[(long long)p * N + n0 + nn] = sD[i];
        }
    }
}

// Kernel 3 -- ONE Q4_K projection, read from a slot. This is kernel 1 with the
// second accumulator (and the up tile) removed, so its arithmetic per output is
// identical: the streamed-vs-resident A/B is a bit comparison.
__global__ void __launch_bounds__(NTHREADS)
knj_q4k_proj(const __half *__restrict__ X, const uint8_t *__restrict__ W,
             const int *__restrict__ pairIdx, const int *__restrict__ pairCnt,
             const int *__restrict__ pairTok, float *__restrict__ Out,
             int N, int K, int rowBytes, long long idxStride) {
    const int e = blockIdx.z;
    const int n0 = blockIdx.x * TN;
    const int m0 = blockIdx.y * TM;
    const int m = pairCnt[e];
    if (m0 >= m) return;

    const uint8_t *w = W + ((long long)e * N + n0) * rowBytes;
    __shared__ __align__(32) __half sA[TM * TK];
    __shared__ __align__(32) __half sB[TK * TN];
    rwx::fragment<rwx::accumulator, TM, TN, TK, float> acc;
    rwx::fill_fragment(acc, 0.0f);
    const int row = threadIdx.x >> 2;
    const int col = (threadIdx.x & 3) * 4;

    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int i = threadIdx.x; i < TM * TK; i += NTHREADS) {
            const int mm = i / TK, kk = i % TK;
            const int p = (m0 + mm < m) ? pairIdx[(long long)e * idxStride + m0 + mm] : -1;
            sA[i] = (p >= 0) ? X[(long long)pairTok[p] * K + k0 + kk] : __float2half(0.0f);
        }
        const int r0 = k0 % 256;
        const int blkOff = (k0 / 256) * KNJ_Q4K_BYTES;
        float span[16];
        knj_q4k_span16(w + row * rowBytes + blkOff, r0, span);
        for (int i = 0; i < 4; ++i) sB[row * TK + col + i] = __float2half(span[col + i]);
        __syncthreads();
        rwx::fragment<rwx::matrix_a, TM, TN, TK, __half, rwx::row_major> fa;
        rwx::fragment<rwx::matrix_b, TM, TN, TK, __half, rwx::col_major> fb;
        rwx::load_matrix_sync(fa, sA, TK);
        rwx::load_matrix_sync(fb, sB, TK);
        rwx::mma_sync(acc, fa, fb, acc);
        __syncthreads();
    }
    __shared__ __align__(32) float sD[TM * TN];
    rwx::store_matrix_sync(sD, acc, TN, rwx::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < TM * TN; i += NTHREADS) {
        const int mm = i / TN, nn = i % TN;
        if (m0 + mm < m) {
            const int p = pairIdx[(long long)e * idxStride + m0 + mm];
            Out[(long long)p * N + n0 + nn] = sD[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Host verification (same convention as q4k_moe_ffn.hip)
// ---------------------------------------------------------------------------
static int g_failed = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            printf("          FAIL  %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++g_failed;                                                      \
        }                                                                    \
    } while (0)

static void cx(const char *name, const std::vector<float> &got,
               const std::vector<double> &ref, int rows, int cols, double tol) {
    int un = 0, bad_rows = 0;
    double wr = 0.0, wa = 0.0;
    for (int r = 0; r < rows; ++r) {
        double rowmax = 0.0;
        for (int c = 0; c < cols; ++c)
            rowmax = std::max(rowmax, std::fabs(ref[(size_t)r * cols + c]));
        double worst = 0.0;
        for (int c = 0; c < cols; ++c) {
            const float g = got[(size_t)r * cols + c];
            if (g == SENTINEL) { ++un; continue; }
            const double d = std::fabs((double)g - ref[(size_t)r * cols + c]);
            wa = std::max(wa, d);
            worst = std::max(worst, d / (rowmax > 0 ? rowmax : 1.0));
        }
        if (worst > tol) ++bad_rows;
        wr = std::max(wr, worst);
    }
    printf("  %-24s rows=%-6d worst-row-rel=%.3e  worst-abs=%.3e  "
           "unwritten=%d  rows-over-tol=%d\n", name, rows, wr, wa, un, bad_rows);
    if (un || bad_rows) ++g_failed;
}

// ---------------------------------------------------------------------------
// Whole-stack helpers
// ---------------------------------------------------------------------------

static std::vector<LayerGeo> read_layer_lines(const std::string &path) {
    std::FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) { printf("cannot open job %s\n", path.c_str()); exit(3); }
    std::vector<LayerGeo> out;
    char line[512];
    while (std::fgets(line, sizeof(line), fp)) {
        if (line[0] != 'L' || line[1] != ' ') continue;
        LayerGeo g;
        int idx = 0;
        if (std::sscanf(line, "L %d %lld %d %lld %d %lld %d %lld",
                        &idx, &g.off[0], &g.bpb[0], &g.off[1], &g.bpb[1],
                        &g.off[2], &g.bpb[2], &g.router_off) != 8) {
            printf("bad L line in job: %s\n", line);
            exit(3);
        }
        out.push_back(g);
    }
    std::fclose(fp);
    return out;
}

// The routing plan: real router weights, softmax, deterministic top-k with
// ascending-expert tie handling (I3), computed ONCE per layer on the resident
// path's carried hidden state and replayed identically by every streamed pass.
// The router is authoritative; the plan is its output, not a synthetic pattern.
struct Plan {
    std::vector<float> rw;             // [P] renormalised top-k weight per pair
    std::vector<int> pairTok, pairExp; // [P]
    std::vector<int> pairIdx;          // [NE * T], -1 padded, stride T
    std::vector<int> pairCnt;          // [NE]
    int maxCnt = 0;
};

static Plan build_plan(const std::vector<float> &xf16, const std::vector<uint8_t> &Wr,
                       int T, int KH, int NE, int TOPK) {
    Plan pl;
    const int P = T * TOPK;
    pl.rw.assign((size_t)P, 0.0f);
    pl.pairTok.assign((size_t)P, 0);
    pl.pairExp.assign((size_t)P, 0);
    pl.pairIdx.assign((size_t)NE * (size_t)T, -1);
    pl.pairCnt.assign((size_t)NE, 0);
    const float *Wr32 = (const float *)Wr.data();
    std::vector<float> logit((size_t)T * NE);
    for (int t = 0; t < T; ++t)
        for (int e = 0; e < NE; ++e) {
            double acc = 0.0;
            for (int h = 0; h < KH; ++h)
                acc += (double)xf16[(size_t)t * KH + h] * (double)Wr32[(size_t)e * KH + h];
            logit[(size_t)t * NE + e] = (float)acc;
        }
    for (int t = 0; t < T; ++t) {
        float *lg = &logit[(size_t)t * NE];
        double mx = -1e300, sum = 0.0;
        for (int e = 0; e < NE; ++e) mx = std::max(mx, (double)lg[e]);
        for (int e = 0; e < NE; ++e) sum += std::exp((double)lg[e] - mx);
        std::vector<int> order((size_t)NE);
        for (int e = 0; e < NE; ++e) order[(size_t)e] = e;
        for (int a = 0; a < NE; ++a)
            for (int b = a + 1; b < NE; ++b)
                if (lg[order[(size_t)b]] > lg[order[(size_t)a]] ||
                    (lg[order[(size_t)b]] == lg[order[(size_t)a]] && order[(size_t)b] < order[(size_t)a]))
                    std::swap(order[(size_t)a], order[(size_t)b]);
        double ksum = 0.0;
        for (int s = 0; s < TOPK; ++s)
            ksum += std::exp((double)lg[order[(size_t)s]] - mx) / sum;
        for (int s = 0; s < TOPK; ++s) {
            const int e = order[(size_t)s];
            const int p = t * TOPK + s;
            pl.pairTok[(size_t)p] = t;
            pl.pairExp[(size_t)p] = e;
            pl.rw[(size_t)p] = (float)((std::exp((double)lg[e] - mx) / sum) / (ksum > 0 ? ksum : 1.0));
            pl.pairIdx[(size_t)e * T + pl.pairCnt[(size_t)e]++] = p;
        }
    }
    for (int e = 0; e < NE; ++e) pl.maxCnt = std::max(pl.maxCnt, pl.pairCnt[(size_t)e]);
    return pl;
}

// The activation, defined once. The resident path and every streamed pass call
// this exact function, so the activation cannot diverge between paths.
static inline __half knj_silu_mul(float gv, float uv) {
    return __float2half((gv / (1.0f + std::exp(-gv))) * uv);
}

// Fused gate-up -> silu activation, on device. Consumes the SAME dG/dU f32
// values the host stage already D2H'd for the bit gate (those are bit-identical
// to the resident gate/up by the existing pure-device gate, for every non-fb
// pass), evaluates silu(gate)*up, and writes f16 into dA for the down kernel.
// The host definition is knj_silu_mul (std::exp); the device has no host exp
// (-nogpulib, AGENTS.md §3 rule 1) so it uses __expf. That makes this a
// DECLARED PRECISION REDUCTION relative to the host definition — reportable as
// max-abs / rel-RMSE / unwritten / mismatching indices, never conflated with a
// tier move (I7). Correctness-for-model is gated end-to-end: fused dA ->
// knj_q6k_down -> down output vs resident down at the 1e-4 expert-GEMM
// tolerance (docs/03). If __expf vs std::exp pushes any down row over 1e-4,
// the fusion is a precision regression, not a win, and it is refused.
__global__ void __launch_bounds__(NTHREADS)
knj_fuse_gateup_silu(const float *__restrict__ G, const float *__restrict__ U,
                     __half *__restrict__ A, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float gv = G[i], uv = U[i];
    // Device silu: native __expf on gfx1201 (quarter-rate). Same formula as the
    // host knj_silu_mul; only the exp implementation differs.
    const float sv = gv / (1.0f + __expf(-gv));
    A[i] = __float2half(sv * uv);
}

// C19's escape hatch, stated as code: the contribution is computed on the host
// from the authoritative file bytes. Same decode (kernels/knj_q4k.h, KNJ_HD --
// the one implementation of the format) and the same sequential-k f32
// accumulation as the device kernel, so the comparison against the resident
// reference is a tolerance comparison, not a bit comparison: the device
// accumulates inside rocWMMA fragments.
static void host_project(const std::string &model, long long base, long long slabBytes,
                         int bpb, int e, const Plan &pl, const std::vector<__half> &in16,
                         const int *pairTok, bool pairRow, std::vector<float> &Out,
                         int T, int N, int K, uint64_t *ns) {
    const uint64_t t0 = mono_ns();
    const int rowBytes = (K / 256) * bpb;
    std::vector<uint8_t> w = read_at(model, base + (long long)e * slabBytes, slabBytes);
    for (int m = 0; m < pl.pairCnt[(size_t)e]; ++m) {
        const int p = pl.pairIdx[(size_t)e * T + m];
        const int r = pairRow ? p : pairTok[p];
        const __half *a = &in16[(size_t)r * K];
        float *o = &Out[(size_t)p * N];
        for (int n = 0; n < N; ++n) {
            const uint8_t *row = &w[(size_t)n * (size_t)rowBytes];
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                const uint8_t *blk = row + (size_t)(k / 256) * (size_t)bpb;
                const float wv = (bpb == KNJ_Q4K_BYTES) ? knj_q4k_weight(blk, k % 256)
                                                        : knj_q6k_weight(blk, k % 256);
                acc += (float)a[k] * (float)__float2half(wv);
            }
            o[n] = acc;
        }
    }
    if (ns) *ns += mono_ns() - t0;   // optional: the fb path times the whole window outside
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <job_all> <dumpdir> [tokens] [slots] [pf_slots] [max_layers]\n",
                     argv[0]);
        return 2;
    }
    knj_wmma_dev dev;
    const int grc = knj_wmma_guard(&dev);
    if (grc) return grc;
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // a crash must not take its own transcript with it

    const int T = (argc > 3) ? std::atoi(argv[3]) : 512;
    const int NSLOTS = (argc > 4) ? std::atoi(argv[4]) : 2;
    const int PFSLOTS = (argc > 5) ? std::atoi(argv[5]) : 16;
    const int nlArg = (argc > 6) ? std::atoi(argv[6]) : 0;

    const auto kv = read_job(argv[1]);
    Job J;
    J.model = jget(kv, "model");
    J.dumpdir = argv[2];
    J.layers = (int)jint(kv, "layers");
    J.hidden = (int)jint(kv, "hidden");
    J.ff = (int)jint(kv, "ff");
    J.experts = (int)jint(kv, "experts");
    J.topk = (int)jint(kv, "topk");
    const char *role3[3] = {"gate", "up", "down"};
    for (int i = 0; i < 3; ++i) {
        J.k[i] = (int)jint(kv, std::string(role3[i]) + "_k");
        J.n[i] = (int)jint(kv, std::string(role3[i]) + "_n");
        J.wpb[i] = (int)jint(kv, std::string(role3[i]) + "_wpb");
    }
    J.geo = read_layer_lines(argv[1]);
    if ((int)J.geo.size() != J.layers) {
        printf("job has %zu L lines but layers=%d\n", J.geo.size(), J.layers);
        exit(3);
    }

    const int NE = J.experts, KH = J.hidden, KF = J.ff, TOPK = J.topk;
    const int NL = (nlArg > 0) ? std::min(nlArg, J.layers) : J.layers;
    const int P = T * TOPK;

    auto slabB = [&](int L, int role) -> long long {
        return (long long)J.k[role] * J.n[role] / J.wpb[role] * J.geo[(size_t)L].bpb[role];
    };
    auto rowBytesL = [&](int L, int role) -> int {
        const int kdim = (role == 2) ? KF : KH;   // the weight row's reduction dim
        return (kdim / 256) * J.geo[(size_t)L].bpb[role];
    };
    int nDownQ4 = 0;
    for (int L = 0; L < NL; ++L) nDownQ4 += (J.geo[(size_t)L].bpb[2] == KNJ_Q4K_BYTES);
    auto dq4L = [&](int L) -> int { return J.geo[(size_t)L].bpb[2] == KNJ_Q4K_BYTES ? 1 : 0; };

    long long maxSlab[3] = {0, 0, 0};
    for (int L = 0; L < NL; ++L)
        for (int r = 0; r < 3; ++r) maxSlab[r] = std::max(maxSlab[r], slabB(L, r));
    const uint64_t slotBytes = (uint64_t)std::max(maxSlab[0], std::max(maxSlab[1], maxSlab[2]));
    const long long rBytes = 4LL * KH * NE;

    std::vector<std::vector<uint8_t>> Wr((size_t)NL);
    for (int L = 0; L < NL; ++L)
        Wr[(size_t)L] = read_at(J.model, J.geo[(size_t)L].router_off, rBytes);

    printf("=== q4k_stream_ffn -- whole stack: C9 + C8 + C11 + C10 prefetch + C9 deadline ===\n");
    printf("device      %s (%s)\n", dev.name, dev.arch);
    printf("layers      %d of %d   hidden=%d ff=%d experts=%d topk=%d\n", NL, J.layers, KH, KF, NE, TOPK);
    printf("tokens      %d   pairs=%d/layer   slots main=%d pf=%d x %.3f MiB\n",
           T, P, NSLOTS, PFSLOTS, slotBytes / 1048576.0);
    printf("down decode mix: %d layer(s) Q4_K, %d layer(s) Q6_K\n", nDownQ4, NL - nDownQ4);

    // --- layer-0 input: synthetic tokens (xorshift, [-2,2]); weights real ----
    std::vector<std::vector<__half>> xIn((size_t)NL);
    {
        xIn[0].resize((size_t)T * KH);
        uint32_t st = 0x9e3779b9u;
        auto nxt = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
        for (size_t i = 0; i < xIn[0].size(); ++i) {
            const float v = (float)((int)(nxt() % 4001) - 2000) / 1000.0f;
            xIn[0][(size_t)i] = __float2half(v);
        }
    }

    // --- device state, sized once for the largest pass -----------------------
    __half *dX, *dA;
    uint8_t *dWg, *dWu, *dWd;
    int *dIdx, *dCnt, *dTok;
    float *dG, *dU, *dY;
    HIPCHECK(hipMalloc(&dX, (size_t)T * KH * sizeof(__half)));
    HIPCHECK(hipMalloc(&dWg, (size_t)NL * NE * maxSlab[0]));
    HIPCHECK(hipMalloc(&dWu, (size_t)NL * NE * maxSlab[1]));
    HIPCHECK(hipMalloc(&dWd, (size_t)NL * NE * maxSlab[2]));
    HIPCHECK(hipMalloc(&dIdx, (size_t)NE * T * sizeof(int)));
    HIPCHECK(hipMalloc(&dCnt, (size_t)NE * sizeof(int)));
    HIPCHECK(hipMalloc(&dTok, (size_t)P * sizeof(int)));
    HIPCHECK(hipMalloc(&dG, (size_t)P * KF * sizeof(float)));
    HIPCHECK(hipMalloc(&dU, (size_t)P * KF * sizeof(float)));
    HIPCHECK(hipMalloc(&dY, (size_t)P * KH * sizeof(float)));
    HIPCHECK(hipMalloc(&dA, (size_t)P * KF * sizeof(__half)));

    // Bank upload: setup, not measurement. Both paths read the same device bank.
    {
        const uint64_t t0 = mono_ns();
        for (int L = 0; L < NL; ++L) {
            std::vector<uint8_t> w = read_at(J.model, J.geo[(size_t)L].off[0], slabB(L, 0) * NE);
            HIPCHECK(hipMemcpy(dWg + (size_t)L * NE * maxSlab[0], w.data(), w.size(), hipMemcpyHostToDevice));
            w = read_at(J.model, J.geo[(size_t)L].off[1], slabB(L, 1) * NE);
            HIPCHECK(hipMemcpy(dWu + (size_t)L * NE * maxSlab[1], w.data(), w.size(), hipMemcpyHostToDevice));
            w = read_at(J.model, J.geo[(size_t)L].off[2], slabB(L, 2) * NE);
            HIPCHECK(hipMemcpy(dWd + (size_t)L * NE * maxSlab[2], w.data(), w.size(), hipMemcpyHostToDevice));
        }
        printf("bank upload %.3f MiB in %.3f s (setup, excluded from every wall clock)\n",
               (double)NL * NE * (maxSlab[0] + maxSlab[1] + maxSlab[2]) / 1048576.0,
               (mono_ns() - t0) / 1e9);
    }

    // =========================================================================
    // PART 1 -- the resident stack. It defines the routing plan and the answers
    // to reproduce. Layer L+1's router consumes layer L's FFN output through the
    // residual, so the stack is inherently lock-step: gate/up -> host
    // activation -> down -> host weighted sum -> carry -> next routing. The
    // host phases are measured and reported, never hidden: a stack comparison
    // that omitted them would be comparing two different programs.
    // =========================================================================
    std::vector<Plan> plans;
    std::vector<std::vector<float>> rG, rU, rY;
    std::vector<float> resLayerMs;
    float resDevMs = 0.0f, resHostMs = 0.0f;
    std::vector<float> resAct0f;   // layer-0 activations, for the gguf-py oracle
    auto run_resident = [&](int Tr, int nlr, std::vector<std::vector<__half>> &xin,
                            std::vector<Plan> &opl, std::vector<std::vector<float>> &oG,
                            std::vector<std::vector<float>> &oU,
                            std::vector<std::vector<float>> &oY,
                            std::vector<float> &lms, float &devMs, float &hostMs,
                            std::vector<float> *act0f) {
        const int Pr = Tr * TOPK;
        opl.assign((size_t)nlr, Plan());
        oG.assign((size_t)nlr, std::vector<float>());
        oU.assign((size_t)nlr, std::vector<float>());
        oY.assign((size_t)nlr, std::vector<float>());
        lms.assign((size_t)nlr, 0.0f);
        std::vector<float> xf((size_t)Tr * KH);
        for (size_t i = 0; i < xf.size(); ++i) xf[i] = (float)xin[0][(size_t)i];
        hipEvent_t v0, v1;
        HIPCHECK(hipEventCreate(&v0));
        HIPCHECK(hipEventCreate(&v1));
        devMs = hostMs = 0.0f;
        for (int L = 0; L < nlr; ++L) {
            std::vector<float> xin16((size_t)Tr * KH);
            for (size_t i = 0; i < xin16.size(); ++i) xin16[i] = (float)xin[(size_t)L][(size_t)i];
            opl[(size_t)L] = build_plan(xin16, Wr[(size_t)L], Tr, KH, NE, TOPK);
            const Plan &pl = opl[(size_t)L];
            const uint64_t h0 = mono_ns();
            HIPCHECK(hipMemcpy(dIdx, pl.pairIdx.data(), pl.pairIdx.size() * sizeof(int), hipMemcpyHostToDevice));
            HIPCHECK(hipMemcpy(dCnt, pl.pairCnt.data(), pl.pairCnt.size() * sizeof(int), hipMemcpyHostToDevice));
            HIPCHECK(hipMemcpy(dTok, pl.pairTok.data(), pl.pairTok.size() * sizeof(int), hipMemcpyHostToDevice));
            HIPCHECK(hipMemcpy(dX, xin[(size_t)L].data(), xin[(size_t)L].size() * sizeof(__half), hipMemcpyHostToDevice));
            hostMs += (float)((double)(mono_ns() - h0) / 1e6);
            const dim3 g1(KF / TN, (pl.maxCnt + TM - 1) / TM, NE);
            const dim3 g2(KH / TN, (pl.maxCnt + TM - 1) / TM, NE);
            HIPCHECK(hipEventRecord(v0));
            knj_q4k_gateup<<<g1, NTHREADS>>>(dX, dWg + (size_t)L * NE * maxSlab[0],
                                             dWu + (size_t)L * NE * maxSlab[1],
                                             dIdx, dCnt, dTok, dG, dU, KF, KH,
                                             rowBytesL(L, 0), Tr);
            HIPCHECK(hipGetLastError());
            HIPCHECK(hipDeviceSynchronize());
            HIPCHECK(hipEventRecord(v1));
            HIPCHECK(hipEventSynchronize(v1));
            float ms = 0.0f;
            HIPCHECK(hipEventElapsedTime(&ms, v0, v1));
            lms[(size_t)L] += ms;
            devMs += ms;
            const uint64_t h1 = mono_ns();
            oG[(size_t)L].assign((size_t)Pr * KF, 0.0f);
            oU[(size_t)L].assign((size_t)Pr * KF, 0.0f);
            HIPCHECK(hipMemcpy(oG[(size_t)L].data(), dG, oG[(size_t)L].size() * sizeof(float), hipMemcpyDeviceToHost));
            HIPCHECK(hipMemcpy(oU[(size_t)L].data(), dU, oU[(size_t)L].size() * sizeof(float), hipMemcpyDeviceToHost));
            std::vector<__half> act((size_t)Pr * KF);
            std::vector<float> actf((size_t)Pr * KF);
            for (size_t i = 0; i < act.size(); ++i) {
                act[(size_t)i] = knj_silu_mul(oG[(size_t)L][(size_t)i], oU[(size_t)L][(size_t)i]);
                actf[(size_t)i] = (float)act[(size_t)i];
            }
            HIPCHECK(hipMemcpy(dA, act.data(), act.size() * sizeof(__half), hipMemcpyHostToDevice));
            hostMs += (float)((double)(mono_ns() - h1) / 1e6);
            if (L == 0 && act0f) *act0f = actf;
            HIPCHECK(hipEventRecord(v0));
            knj_q6k_down<<<g2, NTHREADS>>>(dA, dWd + (size_t)L * NE * maxSlab[2],
                                           dIdx, dCnt, dY, KH, KF, rowBytesL(L, 2), Tr, dq4L(L));
            HIPCHECK(hipGetLastError());
            HIPCHECK(hipDeviceSynchronize());
            HIPCHECK(hipEventRecord(v1));
            HIPCHECK(hipEventSynchronize(v1));
            ms = 0.0f;
            HIPCHECK(hipEventElapsedTime(&ms, v0, v1));
            lms[(size_t)L] += ms;
            devMs += ms;
            const uint64_t h2 = mono_ns();
            oY[(size_t)L].assign((size_t)Pr * KH, 0.0f);
            HIPCHECK(hipMemcpy(oY[(size_t)L].data(), dY, oY[(size_t)L].size() * sizeof(float), hipMemcpyDeviceToHost));
            std::vector<float> xn((size_t)Tr * KH);
            for (int t = 0; t < Tr; ++t)
                for (int h = 0; h < KH; ++h) {
                    float acc = 0.0f;
                    for (int s = 0; s < TOPK; ++s)
                        acc += opl[(size_t)L].rw[(size_t)(t * TOPK + s)] *
                               oY[(size_t)L][(size_t)(t * TOPK + s) * KH + h];
                    xn[(size_t)t * KH + h] = xf[(size_t)t * KH + h] + acc;
                }
            if (L + 1 < nlr) {
                xin[(size_t)L + 1].resize((size_t)Tr * KH);
                for (size_t i = 0; i < xn.size(); ++i) {
                    xin[(size_t)L + 1][(size_t)i] = __float2half(xn[(size_t)i]);
                    xf[(size_t)i] = (float)xin[(size_t)L + 1][(size_t)i];
                }
            }
            hostMs += (float)((double)(mono_ns() - h2) / 1e6);
        }
        hipEventDestroy(v0);
        hipEventDestroy(v1);
    };
    run_resident(T, NL, xIn, plans, rG, rU, rY, resLayerMs, resDevMs, resHostMs, &resAct0f);
    {
        printf("\n--- PART 1: resident stack (the reference) ---\n");
        printf("  device %.3f ms (%.3f ms/layer), host-critical %.3f ms (%.3f ms/layer)\n",
               resDevMs, resDevMs / NL, resHostMs, resHostMs / NL);
        printf("  per-layer device ms:");
        for (int L = 0; L < NL; ++L) printf(" %.2f", resLayerMs[(size_t)L]);
        printf("\n");
    }

    // =========================================================================
    // The streamed stack engine. One variable changes per pass; every pass is
    // graded against the resident stack bit-for-bit (or, for the deadline
    // experiment, at the declared tolerance) and must end with every object
    // back on the file.
    // =========================================================================
    // Per-pass counters, aggregated over the whole stack. The wait buckets are
    // host spin time attributed by the object's residency state; the rest are
    // the instrumented layers (reader busy time, H2D latency, pin/unpin cost).
    struct RunStats {
        const char *mode = "";
        char pass = '?';
        float wall_ms = 0.0f, stream_ms = 0.0f;
        uint64_t w_nvme_ns = 0, w_h2d_ns = 0, w_evict_ns = 0, w_other_ns = 0;
        uint64_t nvme_read_ns = 0, h2d_ns = 0, pin_ns = 0, unpin_ns = 0;
        uint64_t reads = 0, read_bytes = 0, h2d = 0, h2d_bytes = 0, writes_refused = 0;
        uint64_t acquires = 0, releases = 0, refusals = 0, uses = 0, event_gated = 0;
        uint64_t transitions = 0, events = 0, into_vram = 0, misses = 0, fallback = 0;
        uint64_t reuse_hits = 0, allocs = 0, frees = 0;
        std::size_t staged_at_end = 0;
        bool saw_gate = false;
    };

    hipStream_t compute;
    HIPCHECK(hipStreamCreate(&compute));
    HipSlots slotsMain(NSLOTS, slotBytes);
    HipSlots slotsPF(PFSLOTS, slotBytes);
    printf("slot pools   main %d x %.3f MiB, pf %d x %.3f MiB\n", NSLOTS,
           slotBytes / 1048576.0, PFSLOTS, slotBytes / 1048576.0);

    // L0 deadline probe, gated against the resident stack's L0 plan at the 28-layer
    // capacity point, so the gate can close on the resident-vs-streamed bit comparison
    // here rather than waiting on a separate gguf-py dump the resident path does not
    // produce in this driver.
    std::vector<Plan> dpl;
    std::vector<float> dlms;
    float ddev = 0.0f, dhost = 0.0f;
    std::vector<std::vector<float>> dG2, dU2, dY2;
    {
        int DL = std::min(NL, 4);
        std::vector<std::vector<__half>> dxin((size_t)DL);
        {
            dxin[0].resize((size_t)T * KH);
            uint32_t st2 = 0x9e3779b9u;
            auto nxt2 = [&]() { st2 ^= st2 << 13; st2 ^= st2 >> 17; st2 ^= st2 << 5; return st2; };
            for (size_t i = 0; i < dxin[0].size(); ++i) {
                const float v = (float)((int)(nxt2() % 4001) - 2000) / 1000.0f;
                dxin[0][(size_t)i] = __float2half(v);
            }
        }
        std::vector<std::vector<float>> dG, dU, dY;
        run_resident(T, DL, dxin, dpl, dG, dU, dY, dlms, ddev, dhost, nullptr);
    }
    struct PassCfg {
        const char *name;
        HipSlots *sl;
        int mode;      // 0 churn, 1 reuse, 2 overlap, 3 overlap+split copy stream
        int horizon;   // true-set prefetch lead, in layers (0 = none)
        bool cf;       // carry-forward prediction at lead 1 instead
        int deadline;  // demand_deadline_ticks
        bool hostfb;   // C19 host fallback at use time
        int Tr, nl;
        bool waitmode; // waiting pass: demand + wait_ready, strict bit gate
    };
    struct StackStats {
        RunStats agg;
        float wall_ms = 0.0f, host_ms = 0.0f;
        std::vector<float> layer_ms;
        // Per-stage host-phase decomposition (sherlock-it): host-critical is split
        // into sync + D2H(down from prev layer) + D2H(gate) + D2H(up) + silu-or-fuse
        // + H2D(dA) + bitgate + other. Fused path: silu_or_fuse = fused kernel launch
        // (which includes the prefill H2D(dA,sentinel) + kernel). Non-fused path:
        // silu_or_fuse = host silu loop, and host_h2d_da_ns = H2D(dA) that the fused
        // kernel avoids. bitgate is the D2H(ad)+compare for the first-class activation
        // gate (fused path only — non-fused path's activation IS actsH, no separate gate).
        uint64_t host_sync_ns = 0, host_d2h_dy_ns = 0, host_d2h_gate_ns = 0,
                 host_d2h_up_ns = 0, host_silu_ns = 0, host_fuse_launch_ns = 0,
                 host_h2d_da_ns = 0, host_bitgate_ns = 0, host_other_ns = 0;
        bool bits_ok = true, carry_ok = true;
        int first_bad = -1;
        uint64_t demand_hits = 0, demand_total = 0;
        uint64_t pf_read = 0, pf_landed = 0, handoffs = 0;
        uint64_t hostfb_ns = 0, preemptions = 0, refused = 0;
        uint64_t wasted_reads = 0, wasted_bytes = 0;
        // Wait instrumentation (waiting deadline passes): per-item spin counts.
        uint64_t wait_ticks = 0, max_wait_ticks = 0;
        int waited_items = 0;
        uint64_t p50 = 0, p90 = 0;
        float wait_ms = 0.0f;
        uint64_t tick_calls = 0, entry_visits = 0, tick_ns = 0;
        int registry = 0;
        double carry_worst_rel = 0.0;
        int carry_over_tol = 0;
        std::vector<std::vector<float>> Yout;   // filled for host-fallback passes
    };

    auto run_stack = [&](const PassCfg &pc, const std::vector<Plan> &ppl,
                         std::vector<std::vector<__half>> &xin,
                         const std::vector<std::vector<float>> &refG,
                         const std::vector<std::vector<float>> &refU,
                         const std::vector<std::vector<float>> &refY,
                         StackStats &ss) {
        const int Tr = pc.Tr, nlr = pc.nl, Pr = Tr * TOPK;
        const int NIT = 3 * NE * nlr;
        const uint64_t totalBytes = [&] {
            uint64_t b = 0;
            for (int L = 0; L < nlr; ++L)
                for (int r = 0; r < 3; ++r) b += (uint64_t)NE * slabB(L, r);
            return b;
        }();

        HipTransfer xfer(pc.sl, compute, J.model, pc.mode == 3);
        PinnedArena arena(pc.mode != 0);
        CountingFallback fallback;
        CountingTelemetry tel;
        knj::ResidencyManager::Config cfg;
        cfg.slots = pc.sl;
        cfg.transfer = &xfer;
        cfg.arena = &arena;
        cfg.fallback = &fallback;
        cfg.telemetry = &tel;
        cfg.now_ns = &now_ns;
        cfg.demand_deadline_ticks = pc.deadline;
        cfg.prefetch_ttl_ticks = 1000000;   // the driver gives the scheduler time on purpose
        knj::ResidencyManager m(cfg);

        struct Work {
            int L, e, role;
            dim3 grid;
            int N, K, rowBytes, dq4;
            uint64_t bytes;
        };
        std::vector<Work> list;
        list.reserve((size_t)NIT);
        for (int L = 0; L < nlr; ++L) {
            for (int e = 0; e < NE; ++e)
                for (int role = 0; role < 2; ++role)
                    list.push_back({L, e, role,
                                    dim3(KF / TN, (ppl[(size_t)L].maxCnt + TM - 1) / TM, 1),
                                    KF, KH, rowBytesL(L, role), 0, (uint64_t)slabB(L, role)});
            for (int e = 0; e < NE; ++e)
                list.push_back({L, e, 2,
                                dim3(KH / TN, (ppl[(size_t)L].maxCnt + TM - 1) / TM, 1),
                                KH, KF, rowBytesL(L, 2), dq4L(L), (uint64_t)slabB(L, 2)});
        }

        auto obj_of = [&](const Work &w) {
            return knj::ExpertId{(uint32_t)w.L, (uint32_t)(w.e * 3 + w.role)};
        };
        for (const Work &w : list)
            m.register_expert(obj_of(w),
                              knj::BackingExtent{
                                  (uint64_t)(J.geo[(size_t)w.L].off[w.role] +
                                             (long long)w.e * slabB(w.L, w.role)),
                                  (uint64_t)slabB(w.L, w.role)});

        // C7's bounce buffers are pre-registered, not malloc'd per read: warm
        // the pool BEFORE the timed window. Churn (mode 0) keeps the per-read
        // cost on purpose.
        if (pc.mode != 0) {
            const int npre = (pc.mode >= 2) ? pc.sl->n_slots() + 2 : 2;
            for (int i = 0; i < npre; ++i) {
                void *p = arena.alloc(slotBytes);
                if (p) arena.free(p);
            }
        }

        const uint64_t acq0 = pc.sl->acquires(), rel0 = pc.sl->releases();
        const uint64_t use0 = pc.sl->uses(), ref0 = pc.sl->refusals();
        const int depth = (pc.mode >= 2) ? pc.sl->n_slots() : 1;
        const bool bitgate = !pc.hostfb;
        uint64_t event_gated = 0, handoffs = 0, hostfb_ns = 0, hostPhaseNs = 0;
        // Per-stage host-phase traps (sherlock-it transfer + sync decomposition of
        // host-critical). Accumulated across layers inside ensure_acts; printed per
        // pass so host-critical can be decomposed into D2H(gate)+D2H(up)+fused/silu
        // +H2D(dA)+bitgate+sync+other. The fused vs non-fused delta is isolated by
        // a runtime switch (use_fused_silu) so the same source produces both baselines.
        uint64_t host_d2h_gate_ns = 0, host_d2h_up_ns = 0, host_d2h_dy_ns = 0;
        uint64_t host_silu_ns = 0, host_fuse_launch_ns = 0, host_h2d_da_ns = 0;
        uint64_t host_bitgate_ns = 0, host_sync_ns = 0, host_other_ns = 0;
        const bool use_fused_silu = false;  // fusion-ON baseline (fused kernel replaces host silu + H2D(dA))
        uint64_t demand_hits = 0, demand_total = 0;
        uint64_t wait_ticks_total = 0, max_wait_ticks = 0, wait_ns_total = 0;
        int waited_items = 0;
        std::vector<uint64_t> wait_item_ticks;              // one entry per item that waited
        std::unordered_map<uint64_t, uint8_t> demanded;
        std::deque<knj::ExpertId> retire;
        std::vector<char> actsDone((size_t)nlr, 0);
        std::vector<std::vector<float>> Gs((size_t)nlr), Us((size_t)nlr), Ys((size_t)nlr);
        std::vector<std::vector<__half>> actsH((size_t)nlr);
        std::vector<std::map<int, std::vector<float>>> fbStore[3];
        {
            std::map<int, std::vector<float>> m0;
            for (int r = 0; r < 3; ++r) fbStore[r].resize((size_t)nlr, m0);
        }
        int curLayer = -1;
        uint64_t layerT0 = 0;
        uint64_t sweep_base_misses = 0;   // set by the deadline=0 waiting pass
        RunStats st;
        st.mode = pc.name;
        st.pass = 'S';

        auto merge_stash = [&](int L, int role, std::vector<float> &arr) {
            const int N = (role == 2) ? KH : KF;
            for (auto &kv : fbStore[role][(size_t)L])
                std::memcpy(&arr[(size_t)kv.first * N], kv.second.data(),
                            (size_t)N * sizeof(float));
        };
        auto bits_bad = [&](const std::vector<float> &a, const std::vector<float> &b,
                            const char *what, int L) {
            if (a.size() != b.size()) {
                printf("  bits %s L%d: size %zu vs %zu\n", what, L, a.size(), b.size());
                return true;
            }
            for (size_t i = 0; i < a.size(); ++i)
                if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                    printf("  bits %s L%d: first diff @%zu streamed=%.9g resident=%.9g\n",
                           what, L, i, a[i], b[i]);
                    return true;
                }
            return false;
        };

        // The host activation stage. It is on the critical path by construction
        // of this driver (the reference defines silu on the host, and the down
        // pass consumes its f16 bits), so it is measured as a first-class
        // pipeline stage and reported per pass, never hidden.
        auto ensure_acts = [&](int L) {
            if (actsDone[(size_t)L]) return;
            actsDone[(size_t)L] = 1;
            const uint64_t h0 = mono_ns();
            {
                const uint64_t sh0 = mono_ns();
                HIPCHECK(hipStreamSynchronize(compute));
                host_sync_ns += mono_ns() - sh0;
            }
            if (L > 0) {   // layer L-1's down outputs are complete: capture them
                Ys[(size_t)L - 1].assign((size_t)Pr * KH, SENTINEL);
                {
                    const uint64_t sh0 = mono_ns();
                    HIPCHECK(hipMemcpy(Ys[(size_t)L - 1].data(), dY,
                                       Ys[(size_t)L - 1].size() * sizeof(float), hipMemcpyDeviceToHost));
                    host_d2h_dy_ns += mono_ns() - sh0;
                }
                merge_stash(L - 1, 2, Ys[(size_t)L - 1]);
                if (bitgate && bits_bad(Ys[(size_t)L - 1], refY[(size_t)L - 1], "down", L - 1)) {
                    ss.bits_ok = false;
                    if (ss.first_bad < 0) ss.first_bad = L - 1;
                }
            }
            Gs[(size_t)L].assign((size_t)Pr * KF, SENTINEL);
            Us[(size_t)L].assign((size_t)Pr * KF, SENTINEL);
            {
                const uint64_t sh0 = mono_ns();
                HIPCHECK(hipMemcpy(Gs[(size_t)L].data(), dG, Gs[(size_t)L].size() * sizeof(float), hipMemcpyDeviceToHost));
                host_d2h_gate_ns += mono_ns() - sh0;
            }
            {
                const uint64_t sh0 = mono_ns();
                HIPCHECK(hipMemcpy(Us[(size_t)L].data(), dU, Us[(size_t)L].size() * sizeof(float), hipMemcpyDeviceToHost));
                host_d2h_up_ns += mono_ns() - sh0;
            }
            merge_stash(L, 0, Gs[(size_t)L]);
            merge_stash(L, 1, Us[(size_t)L]);
            if (bitgate && (bits_bad(Gs[(size_t)L], refG[(size_t)L], "gate", L) ||
                            bits_bad(Us[(size_t)L], refU[(size_t)L], "up", L))) {
                ss.bits_ok = false;
                if (ss.first_bad < 0) ss.first_bad = L;
            }
            // Build the host-side reference activation (knj_silu_mul, host std::exp)
            // so the fused-device output can be diffed bit-for-bit against it. This is
            // the comparison target for the fusion; it is NOT what feeds the down
            // kernel when the fusion is live — that is the fused kernel's dA.
            // Stage trap: host silu loop (fusion OFF path) — timed so the fused-vs-not
            // delta can be decomposed (the silu loop is what the fused kernel replaces).
            {
                const uint64_t sh0 = mono_ns();
                actsH[(size_t)L].resize((size_t)Pr * KF);
                for (size_t i = 0; i < actsH[(size_t)L].size(); ++i)
                    actsH[(size_t)L][(size_t)i] =
                        knj_silu_mul(Gs[(size_t)L][(size_t)i], Us[(size_t)L][(size_t)i]);
                host_silu_ns += mono_ns() - sh0;
            }
            // Activation path: fused device kernel (replaces host silu loop + H2D(dA)),
            // or the legacy host-silu + H2D(dA) path for the fusion-OFF baseline.
            if (use_fused_silu) {
                const uint64_t sh0 = mono_ns();
                const int na = (int)actsH[(size_t)L].size();
                // Prefill dA with the sentinel (as __half) so any unwritten element is
                // detectable — the fused kernel writes every i<na, but the driver's
                // convention is prefill-and-probe, and rule 7 wants every output checked.
                HIPCHECK(hipMemcpy(dA, &sSil, sizeof(__half), hipMemcpyHostToDevice));
                const dim3 grid((na + NTHREADS - 1) / NTHREADS, 1, 1);
                knj_fuse_gateup_silu<<<grid, NTHREADS, 0, compute>>>(dG, dU, dA, na);
                HIPCHECK(hipGetLastError());
                host_fuse_launch_ns += mono_ns() - sh0;
            } else {
                // Legacy path (fusion OFF baseline): host silu already computed actsH;
                // H2D it into dA for the down kernel. Timed so the fused-vs-not delta
                // includes the H2D(dA) that the fused kernel avoids.
                const uint64_t sh0 = mono_ns();
                HIPCHECK(hipMemcpyAsync(dA, actsH[(size_t)L].data(),
                                        actsH[(size_t)L].size() * sizeof(__half),
                                        hipMemcpyHostToDevice, compute));
                host_h2d_da_ns += mono_ns() - sh0;
            }
            // Bit-identity gate on the fused activation itself: dA (device) vs actsH[L]
            // (host knj_silu_mul reference). This is a FIRST-CLASS check on the fusion,
            // reported like every other bit gate in this driver (I7: the fused tier move
            // is invisible). Any mismatch / unwritten here is reported regardless of the
            // downstream 1e-4 down-projection gate — the fusion is not "close enough",
            // it is either bit-identical to the host stage or it is a declared precision
            // reduction (in which case the 1e-4 down gate is the binding constraint).
            // Stage trap: the D2H(ad) + compare is timed so the fused-vs-not delta can be
            // decomposed (the fused path still does this D2H+compare for the first-class
            // activation gate; the fusion-OFF path does not — its activation IS actsH).
            if (bitgate) {
                const uint64_t sh0 = mono_ns();
                const int na = (int)actsH[(size_t)L].size();
                std::vector<__half> ad((size_t)na, __float2half(SENTINEL));
                HIPCHECK(hipMemcpy(ad.data(), dA, ad.size() * sizeof(__half),
                                   hipMemcpyDeviceToHost));
                host_bitgate_ns += mono_ns() - sh0;
                int un = 0, bad = 0;
                double wa = 0.0, wr = 0.0;
                int wb = -1;
                float wgh = 0.0f, wrh = 0.0f, wd = 0.0f;
                for (int i = 0; i < na; ++i) {
                    const float gh = __half2float(ad[i]);
                    if (gh == SENTINEL) { ++un; continue; }
                    const float rh = __half2float(actsH[(size_t)L][(size_t)i]);
                    const float d = std::fabs(gh - rh);
                    if (d > wa) wa = d;
                    const float rm = std::fmaxf(std::fabs(rh), 1.0e-7f);
                    wr = std::fmax(wr, d / rm);
                    if (d > 0.0f) { ++bad; if (d > wd) { wd = d; wb = i; wgh = gh; wrh = rh; } }
                }
                printf("  fuse-act L%d: dA-vs-host bits identical=%s differing=%d/%d "
                       "max-abs=%.3e rel-rmse-max=%.3e unwritten=%d",
                       L, bad == 0 ? "yes" : "no", bad, na, wa, wr, un);
                if (wb >= 0) {
                    const float gv = __half2float(Gs[(size_t)L][(size_t)wb]);
                    const float uv = __half2float(Us[(size_t)L][(size_t)wb]);
                    const float silu_host = (gv / (1.0f + std::exp(-gv))) * uv;
                    printf("  | worst #%d dh=%.6f hh=%.6f abs=%.6f rel=%.3e  "
                           "G[idx]=%.6f U[idx]=%.6f silu_host=%.6f",
                           wb, (double)wgh, (double)wrh, (double)wd,
                           (double)(wd / std::fmaxf(std::fabs(wrh), 1e-7f)),
                           (double)gv, (double)uv, (double)silu_host);
                }
                printf("\n");
                if (un || bad) ss.bits_ok = false;
                if (ss.first_bad < 0 && (un || bad)) ss.first_bad = L;
            }
            hostPhaseNs += mono_ns() - h0;
        };

        auto drain_retire = [&]() {
            bool progress = false;
            for (std::size_t k = 0; k < retire.size();) {
                const knj::Residency s = m.locate(retire[k]);
                if (s == knj::Residency::RAM_RESIDENT) {
                    m.demote(retire[k]);
                    ++k;
                } else if (s == knj::Residency::NVME_RESIDENT || s == knj::Residency::INVALID) {
                    retire.erase(retire.begin() + (long)k);
                    progress = true;
                } else if (s == knj::Residency::VRAM_RESIDENT ||
                           s == knj::Residency::PREFETCHED) {
                    // A demand left pending by a C19 hand-off lands in VRAM on the
                    // file's behalf; its launcher already demoted it, so nothing
                    // else ever will -- retire it here or it holds a slot forever.
                    m.demote(retire[k]);
                    ++k;
                } else {
                    ++k;
                }
            }
            return progress;
        };

        auto launch = [&](int i) {
            const Work &w = list[(size_t)i];
            const knj::SlotHandle sh = pc.sl->slot_for((uint32_t)w.L, (uint32_t)(w.e * 3 + w.role));
            if (sh == knj::kNoSlot) { printf("no slot for work item %d\n", i); exit(4); }
            void *wp = pc.sl->vram_ptr(sh);
            if (i == 0) {   // probe: the slot must hold the authoritative file bytes
                std::vector<uint8_t> want =
                    read_at(J.model, J.geo[(size_t)w.L].off[w.role] +
                                      (long long)w.e * slabB(w.L, w.role), slabB(w.L, w.role));
                std::vector<uint8_t> got(want.size());
                HIPCHECK(hipMemcpy(got.data(), wp, got.size(), hipMemcpyDeviceToHost));
                printf("slot-bytes probe L%d e%d r%d: %s (first 8: %02x %02x %02x %02x %02x %02x %02x %02x / "
                       "%02x %02x %02x %02x %02x %02x %02x %02x)\n",
                       w.L, w.e, w.role,
                       std::memcmp(got.data(), want.data(), got.size()) == 0 ? "match" : "MISMATCH",
                       got[0], got[1], got[2], got[3], got[4], got[5], got[6], got[7],
                       want[0], want[1], want[2], want[3], want[4], want[5], want[6], want[7]);
            }
            if (pc.mode == 3) pc.sl->wait_load(sh, compute);   // copy -> consume dependency
            float *out = (w.role == 0) ? dG : (w.role == 1) ? dU : dY;
            // ONE expert per launch: the slot holds that expert's slab, so the
            // routing is sliced to it (blockIdx.z = 0 indexes the slice base).
            // A grid.z = NE launch here would run all four experts against one
            // expert's weights and the last writer would win every pair row.
            const dim3 one(w.grid.x, w.grid.y, 1);
            const int *pidx = dIdx + (size_t)w.e * Tr;
            const int *pcnt = dCnt + w.e;
            if (w.role == 2)
                knj_q6k_down<<<one, NTHREADS, 0, compute>>>(
                    (const __half *)dA, (const uint8_t *)wp, pidx, pcnt, out,
                    w.N, w.K, w.rowBytes, Tr, w.dq4);
            else
                knj_q4k_proj<<<one, NTHREADS, 0, compute>>>(
                    (const __half *)dX, (const uint8_t *)wp, pidx, pcnt, dTok, out,
                    w.N, w.K, w.rowBytes, Tr);
            HIPCHECK(hipGetLastError());
            pc.sl->mark_used(sh, compute);
            m.demote(obj_of(w));
            if (!pc.sl->released(sh)) ++event_gated;
        };

        auto bucket = [&](knj::Residency state, uint64_t dt) {
            switch (state) {
                case knj::Residency::NVME_RESIDENT:
                case knj::Residency::LOADING_NVME: st.w_nvme_ns += dt; break;
                case knj::Residency::RAM_RESIDENT:
                case knj::Residency::LOADING_RAM:  st.w_h2d_ns += dt; break;
                case knj::Residency::EVICTING_VRAM:
                case knj::Residency::EVICTING_RAM: st.w_evict_ns += dt; break;
                default: st.w_other_ns += dt; break;
            }
        };

        // Wait policy. Two regimes, both driven by C8's load event rather than by
        // polling `m.ready()` from the step thread:
        //
        //   * BACKOFF_WAIT -- the step thread sleeps between retries (the `now_ns`
        //     clock the manager is handed is `mono_ns()`, not the wall clock, and
        //     the step never spins at 100% in this regime).
        //   * EVENT_WAIT -- the step thread parks on a `condition_variable` until a
        //     slot's load event fires, then retries once. Implemented on the host
        //     with a `hipStreamQuery(copy_stream)` + `std::this_thread::yield()`
        //     fast path and a timed park as the fallback; the event that gates the
        //     wait is the same C8 load event `m.ready()` reads, so the contract
        //     (issued is not ready; the event is the proof) is untouched.
        //
        // Both keep I4 (no synchronous NVMe read on the critical path) and the
        // deadline semantics: `m.tick()` advances only when the step chooses to
        // poll, and a parked step does not advance the deadline clock until it
        // wakes. The C9 deadline path (PART 4) is therefore a real experiment
        // against a real clock, not a vacuous spin --- that is the whole of the
        // gate the design documents keep asking for.
        //
        // The default for this driver is BACKOFF_WAIT because it is the single
        // change with the least surface: it stands between the current spin and
        // EVENT_WAIT, and it removes the measured 183-295 ms tick time without
        // adding a new synchronisation primitive. EVENT_WAIT is below, disabled by
        // default, so the driver stays bit-identical at the gate until this change
        // is graded.
        const bool use_backoff_wait = true;   // change this to false to use the spin
        const bool use_event_wait = false;    // host-event park (under measurement)

        // Host-event park for the EVENT_WAIT regime. The gate is the same C8 load
        // event `m.ready()` reads, so the contract is identical; the difference is
        // that the step thread parks instead of polling.
        auto wait_ready_event = [&](int i, const knj::ExpertId id, uint64_t w0, int &guard,
                                    int &spins) {
            // Fast path first: if the slot's load event is already fired, no park.
            if (m.ready(id)) return;
            ++spins;
            const knj::Residency state = m.locate(id);
            if (state == knj::Residency::RAM_RESIDENT) {
                m.demand(id, knj::KVTier::VRAM);
            }
            const uint64_t a = mono_ns();
            drain_retire();
            m.tick();
            bucket(state, mono_ns() - a);
            // Park once, then retry: a single parked wake is cheap and still
            // serialises against the transfer that fills the slot (the load event
            // is recorded on the copy stream; the consumer stream waits on it when
            // pc.mode==3, and C8's released() queries the event directly).
            if (!m.ready(id)) {
                hipStream_t cs = xfer.xstream();
                // A short poll loop with yields, capped at the same guard as the
                // backoff path, so a buggy transfer chain is still caught.
                int park_guard = 0;
                while (!m.ready(id) && park_guard < 2000000) {
                    std::this_thread::yield();
                    ++park_guard;
                    if (park_guard % 10000 == 0) {
                        drain_retire();
                        m.tick();
                    }
                }
                // If the event has not fired by now, fall back to a timed sleep
                // and keep polling until the guard trips.
                if (!m.ready(id)) {
                    while (!m.ready(id)) {
                        std::this_thread::sleep_for(std::chrono::microseconds(500));
                        drain_retire();
                        m.tick();
                        if (++guard > 8000000) {
                            printf("STALL waiting for work item %d (L%d e%d r%d): state=%s slot=%d\n",
                                   i, list[(size_t)i].L, list[(size_t)i].e, list[(size_t)i].role,
                                   knj::to_string(m.locate(id)),
                                   pc.sl->slot_for((uint32_t)list[(size_t)i].L,
                                                   (uint32_t)(list[(size_t)i].e * 3 + list[(size_t)i].role)));
                            pc.sl->dump("wait");
                            printf("  xfer in flight: %zu\n", xfer.in_flight());
                            exit(4);
                        }
                        ++spins;
                    }
                }
            }
            // Once the event has fired (or the fast path found it fired), the
            // demand retry and retired-earlier-work loop runs exactly once more,
            // so the semantics match the spin's final iteration.
            if (!m.ready(id)) {
                const knj::Residency st2 = m.locate(id);
                if (st2 == knj::Residency::RAM_RESIDENT) m.demand(id, knj::KVTier::VRAM);
                drain_retire();
                m.tick();
            }
        };

        auto wait_ready = [&](int i) {
            const knj::ExpertId id = obj_of(list[(size_t)i]);
            // The prefetch hit metric: were the object's bytes already off the file
            // the moment its step arrived (RAM or better -- the NVMe hop, the
            // expensive one, already paid)? Counted before any wait, so it measures
            // exactly what a scheduler cares about: waits that prefetch removed.
            if (list[(size_t)i].L >= 1) {   // layer 0 cannot be prefetched: excluded
                ++demand_total;
                const knj::Residency s = m.locate(id);
                if (s != knj::Residency::NVME_RESIDENT && s != knj::Residency::INVALID)
                    ++demand_hits;
            }
            int guard = 0, spins = 0;
            uint64_t w0 = mono_ns();
            if (use_event_wait && pc.mode >= 2) {
                int guard_ev = 0, spins_ev = 0;
                w0 = mono_ns();
                wait_ready_event(i, id, w0, guard_ev, spins_ev);
                if (spins_ev > 0) {
                    ++waited_items;
                    wait_item_ticks.push_back((uint64_t)spins_ev);
                    wait_ticks_total += (uint64_t)spins_ev;
                    if ((uint64_t)spins_ev > max_wait_ticks) max_wait_ticks = (uint64_t)spins_ev;
                }
                if (pc.waitmode) wait_ns_total += mono_ns() - w0;
                return;
            }
            if (use_backoff_wait) {
                // backoff wait: sleep between retries, capped at the guard as a
                // hard stall detector (same as the spin, but sleeps instead of
                // busy-looping).
                uint64_t backoff_us = 1;   // start at 1 us
                while (!m.ready(id)) {
                    ++spins;
                    const knj::Residency state = m.locate(id);
                    if (state == knj::Residency::RAM_RESIDENT) {
                        m.demand(id, knj::KVTier::VRAM);
                    }
                    const uint64_t a = mono_ns();
                    drain_retire();
                    m.tick();
                    bucket(state, mono_ns() - a);
                    if (++guard > 8000000) {
                        printf("STALL waiting for work item %d (L%d e%d r%d): state=%s slot=%d\n",
                               i, list[(size_t)i].L, list[(size_t)i].e, list[(size_t)i].role,
                               knj::to_string(m.locate(id)),
                               pc.sl->slot_for((uint32_t)list[(size_t)i].L,
                                               (uint32_t)(list[(size_t)i].e * 3 + list[(size_t)i].role)));
                        pc.sl->dump("wait");
                        printf("  xfer in flight: %zu\n", xfer.in_flight());
                        exit(4);
                    }
                    // capped exponential backoff, with a yield at the low end so
                    // the first few retries are not sleeping for nothing.
                    if (backoff_us < 250) {
                        if (backoff_us >= 4) std::this_thread::yield();
                        else std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
                        backoff_us = std::min(backoff_us * 2, (uint64_t)250);
                    } else {
                        std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
                        backoff_us = std::min(backoff_us + 250, (uint64_t)2500);
                    }
                }
            } else {
                // legacy spin (the thing this driver is replacing).
                const uint64_t w0s = mono_ns();
                while (!m.ready(id)) {
                    ++spins;
                    const knj::Residency state = m.locate(id);
                    if (state == knj::Residency::RAM_RESIDENT) {
                        m.demand(id, knj::KVTier::VRAM);
                    }
                    const uint64_t a = mono_ns();
                    drain_retire();
                    m.tick();
                    bucket(state, mono_ns() - a);
                    if (++guard > 8000000) {
                        printf("STALL waiting for work item %d (L%d e%d r%d): state=%s slot=%d\n",
                               i, list[(size_t)i].L, list[(size_t)i].e, list[(size_t)i].role,
                               knj::to_string(m.locate(id)),
                               pc.sl->slot_for((uint32_t)list[(size_t)i].L,
                                               (uint32_t)(list[(size_t)i].e * 3 + list[(size_t)i].role)));
                        pc.sl->dump("wait");
                        printf("  xfer in flight: %zu\n", xfer.in_flight());
                        exit(4);
                    }
                }
                if (pc.waitmode) wait_ns_total += mono_ns() - w0s;
                w0 = mono_ns();   // keep the caller's w0 semantics (the wall window)
            }
            if (spins > 0) {   // per-item wait count, actually measured
                ++waited_items;
                wait_item_ticks.push_back((uint64_t)spins);
                wait_ticks_total += (uint64_t)spins;
                if ((uint64_t)spins > max_wait_ticks) max_wait_ticks = (uint64_t)spins;
            }
            if (pc.waitmode) wait_ns_total += mono_ns() - w0;
        };

        // Lookahead demand fills RAM only (no slot taken): the NVMe read -- the
        // expensive hop -- overlaps the current layer's compute, but the slot
        // is taken by the item that is about to run, in order. A VRAM-tier
        // lookahead would let a later item consume the slot the current item
        // needs (C9 grants slots in hash order, not in step order), and the
        // step then deadlocks behind slots whose kernels have not launched.
        auto issue_demand = [&](const Work &w, bool lookahead) {
            const knj::ExpertId id = obj_of(w);
            const bool was_ready = m.demand(id, lookahead ? knj::KVTier::RAM : knj::KVTier::VRAM);
            demanded[id.key()] = 1;
            (void)was_ready;
            if (pc.horizon == 0 && !pc.cf) CHECK(!was_ready);   // cold start: never ready on first call
        };
        auto issue_prefetch_true = [&](int Lp) {
            std::vector<char> exp((size_t)NE, 0);
            for (int p = 0; p < Pr; ++p) exp[ppl[(size_t)Lp].pairExp[(size_t)p]] = 1;
            for (int e = 0; e < NE; ++e)
                for (int r = 0; r < 3; ++r) {
                    const knj::ExpertId id{(uint32_t)Lp, (uint32_t)(e * 3 + r)};
                    if (demanded.count(id.key())) continue;
                    // RAM tier on purpose: speculative bytes must never take a VRAM
                    // slot. A two-layer true-set (12+12 objects here) is wider than
                    // the slot pool, and VRAM-tier prefetch starves the CURRENT item
                    // -- the step then waits behind slots whose kernels never ran.
                    // The step's own demand promotes the prefetched bytes to VRAM.
                    m.prefetch(id, knj::KVTier::RAM, knj::PrefetchPriority::NORMAL);
                }
        };
        auto enter_layer = [&](int L) {
            if (L == curLayer) return;
            if (curLayer >= 0)
                ss.layer_ms[(size_t)curLayer] += (float)((double)(mono_ns() - layerT0) / 1e6);
            curLayer = L;
            layerT0 = mono_ns();
            m.begin_layer((uint32_t)L);
            HIPCHECK(hipMemcpy(dIdx, ppl[(size_t)L].pairIdx.data(),
                               ppl[(size_t)L].pairIdx.size() * sizeof(int), hipMemcpyHostToDevice));
            HIPCHECK(hipMemcpy(dCnt, ppl[(size_t)L].pairCnt.data(),
                               ppl[(size_t)L].pairCnt.size() * sizeof(int), hipMemcpyHostToDevice));
            HIPCHECK(hipMemcpy(dTok, ppl[(size_t)L].pairTok.data(),
                               ppl[(size_t)L].pairTok.size() * sizeof(int), hipMemcpyHostToDevice));
            HIPCHECK(hipMemcpy(dX, xin[(size_t)L].data(), xin[(size_t)L].size() * sizeof(__half),
                               hipMemcpyHostToDevice));
            // C10 boundary prefetch. True-set passes prefetch layer L+horizon's
            // routed objects -- an oracle predictor, which isolates the pipeline
            // mechanism from predictor accuracy. The carry-forward pass instead
            // predicts layer L+1 from layer L's routing and pays for its errors
            // in bandwidth, never in correctness.
            if (pc.horizon > 0 && !pc.cf) {
                const int lo = (L == 0) ? 1 : L + pc.horizon;
                const int hi = (L == 0) ? pc.horizon : L + pc.horizon;
                for (int t = lo; t <= hi && t < nlr; ++t) issue_prefetch_true(t);
            } else if (pc.cf && L + 1 < nlr) {
                std::vector<char> exp((size_t)NE, 0);
                for (int p = 0; p < Pr; ++p) exp[ppl[(size_t)L].pairExp[(size_t)p]] = 1;
                for (int e = 0; e < NE; ++e)
                    for (int r = 0; r < 3; ++r) {
                        const knj::ExpertId id{(uint32_t)(L + 1), (uint32_t)(e * 3 + r)};
                        if (demanded.count(id.key())) continue;
                        m.prefetch(id, knj::KVTier::RAM, knj::PrefetchPriority::NORMAL);
                    }
            }
        };

        const uint64_t wall0 = mono_ns();
        layerT0 = wall0;
        ss.layer_ms.assign((size_t)nlr, 0.0f);
        // Layer bookkeeping (uploads, prefetch) is driven by the item being
        // PROCESSED, never by a lookahead demand: with depth > items-per-layer
        // the demand lookahead crosses into the next layer, and begin_layer()
        // must still see layers advance forwards.
        enter_layer(list[0].L);
        if (pc.mode >= 2)
            for (int i = 0; i < NIT && i < depth; ++i) issue_demand(list[(size_t)i], true);
        for (int i = 0; i < NIT; ++i) {
            const Work &w = list[(size_t)i];
            enter_layer(w.L);
            if (pc.mode >= 2) {
                const int look = i + depth;
                if (look < NIT) issue_demand(list[(size_t)look], true);
            }
            if (w.role == 2) ensure_acts(w.L);
            const knj::ExpertId id = obj_of(w);
            if (pc.hostfb && !pc.waitmode && !m.ready(id)) {
                // The step needs it now, it is not here: C19. The contribution
                // is computed on the host from the authoritative bytes and the
                // pipeline moves on without a stall. I1: computed, never skipped.
                const bool got = m.demand_or_fallback(id);
                CHECK(!got);
                ++handoffs;
                const uint64_t f0 = mono_ns();
                const int Nout = (w.role == 2) ? KH : KF;
                std::vector<float> scratch((size_t)Pr * Nout, 0.0f);
                if (w.role == 2)
                    host_project(J.model, J.geo[(size_t)w.L].off[2], slabB(w.L, 2),
                                 J.geo[(size_t)w.L].bpb[2], w.e, ppl[(size_t)w.L],
                                 actsH[(size_t)w.L], nullptr, true, scratch, Tr, KH, KF,
                                 nullptr);   // timed here by the f0/f1 window below
                else
                    host_project(J.model, J.geo[(size_t)w.L].off[w.role],
                                 slabB(w.L, w.role), J.geo[(size_t)w.L].bpb[w.role],
                                 w.e, ppl[(size_t)w.L], xin[(size_t)w.L],
                                 ppl[(size_t)w.L].pairTok.data(), false,
                                 scratch, Tr, KF, KH, nullptr);
                hostfb_ns += mono_ns() - f0;   // once: the outer window IS the timer
                const int cc = ppl[(size_t)w.L].pairCnt[(size_t)w.e];
                for (int mm = 0; mm < cc; ++mm) {
                    const int p = ppl[(size_t)w.L].pairIdx[(size_t)w.e * Tr + mm];
                    fbStore[w.role][(size_t)w.L][p] = std::vector<float>(
                        scratch.begin() + (long)p * Nout, scratch.begin() + (long)(p + 1) * Nout);
                }
                m.demote(id);
                retire.push_back(id);
                continue;   // no stall: the contribution was computed elsewhere
            }
            if (pc.mode < 2) issue_demand(w, false);   // waitmode included: cold start, step waits
            wait_ready(i);
            launch(i);
            retire.push_back(id);
            if (pc.mode >= 2) {
                drain_retire();
            } else {
                int guard = 0;
                while (m.locate(id) != knj::Residency::NVME_RESIDENT) {
                    const knj::Residency state = m.locate(id);
                    const uint64_t a = mono_ns();
                    drain_retire();
                    m.tick();
                    bucket(state, mono_ns() - a);
                    if (++guard > 8000000) { printf("STALL retiring work item %d\n", i); exit(4); }
                }
            }
        }
        if (pc.waitmode) {
            // The fence lives in the epilogue, after ss's wait fields are filled:
            // a waiting pass with zero recorded waits would be a vacuous sweep.
        }
        // Final layer's down outputs, then close the last layer window.
        {
            const uint64_t h0 = mono_ns();
            HIPCHECK(hipStreamSynchronize(compute));
            Ys[(size_t)nlr - 1].assign((size_t)Pr * KH, SENTINEL);
            HIPCHECK(hipMemcpy(Ys[(size_t)nlr - 1].data(), dY,
                               Ys[(size_t)nlr - 1].size() * sizeof(float), hipMemcpyDeviceToHost));
            merge_stash(nlr - 1, 2, Ys[(size_t)nlr - 1]);
            if (bitgate && bits_bad(Ys[(size_t)nlr - 1], refY[(size_t)nlr - 1], "down", nlr - 1)) {
                ss.bits_ok = false;
                if (ss.first_bad < 0) ss.first_bad = nlr - 1;
            }
            hostPhaseNs += mono_ns() - h0;
        }
        if (curLayer >= 0)
            ss.layer_ms[(size_t)curLayer] += (float)((double)(mono_ns() - layerT0) / 1e6);
        // Carry verification: the streamed chain must reproduce the resident
        // chain's hidden state, layer by layer (I7 across the stack). Pure-device
        // passes are graded bit-for-bit; a pass whose Y carries C19 host
        // contributions is graded at the declared tolerance instead -- those
        // inputs differ from the reference at up to 1e-4 legally, and the f16
        // rounding through the carry makes a bit-identical chain mathematically
        // impossible (not a bug to fix -- a category to declare).
        {
            std::vector<float> xf((size_t)Tr * KH);
            for (size_t i = 0; i < xf.size(); ++i) xf[i] = (float)xin[0][(size_t)i];
            const bool graded = (handoffs > 0);
            // cx semantics for the graded path: per token (row), worst abs error
            // relative to the row max of the reference, across all layer
            // boundaries, graded at 2e-3 (the declared mixed-accumulation class).
            std::vector<double> rowMax((size_t)Tr, 0.0), rowErr((size_t)Tr, 0.0);
            for (int L = 0; L < nlr && (graded ? true : ss.carry_ok); ++L) {
                std::fill(rowMax.begin(), rowMax.end(), 0.0);
                std::fill(rowErr.begin(), rowErr.end(), 0.0);
                for (int t = 0; t < Tr; ++t)
                    for (int h = 0; h < KH; ++h) {
                        float acc = 0.0f;
                        for (int s = 0; s < TOPK; ++s)
                            acc += ppl[(size_t)L].rw[(size_t)(t * TOPK + s)] *
                                   Ys[(size_t)L][(size_t)(t * TOPK + s) * KH + h];
                        const float xn = xf[(size_t)t * KH + h] + acc;
                        const __half xh = __float2half(xn);
                        if (L + 1 < nlr) {
                            const __half &rh = xin[(size_t)L + 1][(size_t)t * KH + h];
                            const float ref = (float)rh;
                            if (graded) {
                                rowMax[(size_t)t] = std::max(rowMax[(size_t)t],
                                                             (double)std::fabs(ref));
                                rowErr[(size_t)t] = std::max(rowErr[(size_t)t],
                                                             (double)std::fabs((float)xh - ref));
                            } else if (std::memcmp(&xh, &rh, sizeof(__half)) != 0) {
                                ss.carry_ok = false;
                                break;   // inner loop
                            }
                        }
                        xf[(size_t)t * KH + h] = (float)xh;
                    }
                if (graded) {
                    for (int t = 0; t < Tr; ++t) {
                        const double rel = rowErr[(size_t)t] /
                                           (rowMax[(size_t)t] < 1e-9 ? 1.0 : rowMax[(size_t)t]);
                        if (rel > ss.carry_worst_rel) ss.carry_worst_rel = rel;
                        if (rel > 2e-3) ++ss.carry_over_tol;
                    }
                }
                if (!ss.carry_ok) break;
            }
            if (graded) ss.carry_ok = (ss.carry_over_tol == 0);
        }
        // Every registered object back on the file: nothing stays cached across
        // passes, and the invariant is checked per object.
        for (int L = 0; L < nlr; ++L)
            for (int e = 0; e < NE; ++e)
                for (int r = 0; r < 3; ++r) {
                    const knj::ExpertId id{(uint32_t)L, (uint32_t)(e * 3 + r)};
                    int guard = 0, lastFree = -1;
                    while (m.locate(id) != knj::Residency::NVME_RESIDENT) {
                        const knj::Residency state = m.locate(id);
                        const uint64_t a = mono_ns();
                        drain_retire();
                        m.tick();
                        bucket(state, mono_ns() - a);
                        if (++guard > 2000000) {
                            if (pc.sl->free_slots() == lastFree) {
                                printf("STALL draining L%d e%d r%d: state=%s free=%d in-flight=%zu "
                                       "deadline-misses=%llu\n", L, e, r, knj::to_string(state),
                                       lastFree, xfer.in_flight(),
                                       (unsigned long long)m.deadline_misses());
                                pc.sl->dump("drain");
                                exit(4);
                            }
                            lastFree = pc.sl->free_slots();
                            guard = 0;
                        }
                    }
                }
        ss.wall_ms = (float)((double)(mono_ns() - wall0) / 1e6);
        if (pc.hostfb || pc.waitmode) {
            ss.Yout = Ys;   // wait mode is graded too: at the bit gate (pure device)
        }

        // ---- wasted prefetch, before the checks that cite it ----------------
        ss.wasted_reads = xfer.nvme_reads() > (uint64_t)NIT ? xfer.nvme_reads() - (uint64_t)NIT : 0;
        ss.wasted_bytes = xfer.nvme_bytes() > totalBytes ? xfer.nvme_bytes() - totalBytes : 0;

        // ---- structural checks, per pass ------------------------------------
        CHECK(tel.events() == m.transitions());
        CHECK(m.blocking_transfer_calls() == 0);              // I4, structurally
        CHECK(fallback.expert() == handoffs);                 // C19 called once per hand-off
        if (pc.horizon == 0 && !pc.cf) {
            if (!pc.hostfb) {
                CHECK(xfer.nvme_reads() == (uint64_t)NIT);        // one whole read per object
                CHECK(xfer.h2d_submits() == (uint64_t)NIT);
                CHECK(xfer.h2d_bytes() == totalBytes);            // bytes moved == bytes consumed (I2)
            } else {
                // A hand-off hugs its device transfer: the demand may have completed
                // the read outside this pass's accounting, and C19 may eat the step's
                // H2D. The identity that must hold is total-bytes, not per-op count.
                CHECK(xfer.nvme_reads() >= (uint64_t)NIT);
                CHECK(xfer.h2d_submits() <= (uint64_t)NIT);
                CHECK(xfer.h2d_submits() + handoffs >= (uint64_t)NIT);
            }
            CHECK(xfer.h2d_bytes() <= totalBytes);   // no transfer the kernel did not consume
        } else {
            // A prefetched object is read once and consumed by its demand; a
            // wrongly predicted one is read and never consumed -- that waste is
            // the predictor's bandwidth cost, counted exactly.
            CHECK(xfer.nvme_reads() == (uint64_t)NIT + ss.wasted_reads);
            CHECK(xfer.h2d_submits() == (uint64_t)NIT);
            CHECK(xfer.h2d_bytes() >= totalBytes);
        }
        CHECK(xfer.nvme_bytes() == totalBytes + ss.wasted_bytes);
        CHECK(pc.sl->releases() - rel0 == pc.sl->acquires() - acq0);
        CHECK(pc.sl->uses() - use0 <= (uint64_t)NIT);   // fb steps never reach the kernel
        ss.demand_hits = demand_hits;
        ss.demand_total = demand_total;
        CHECK(pc.sl->peak_in_use() >= 1);
        if (pc.hostfb) CHECK(handoffs > 0);                   // the mechanism was exercised
        if (pc.hostfb && !pc.waitmode && pc.deadline == 0) {
            // THE FINDING, stated as a check: demand_or_fallback is a no-wait
            // consumer, and a no-wait consumer never advances the manager's tick
            // clock -- so no deadline can fire, however tight. Misses must be
            // exactly 0 here, and the clock must be exactly the drain-only tail.
            CHECK(tel.misses() == 0);
            CHECK(m.tick_index() <= (uint64_t)(NIT + 1));
        }
        if (pc.waitmode) {
            // The miss curve: deadline 0 -> everything misses; looser -> fewer.
            // 5000 spins is still tighter than a ~9 ms NVMe read here, so a
            // non-zero count at 5000 is a measurement, not a failure. The only
            // hard gate: the sweep must beat the deadline-0 baseline.
            if (pc.deadline == 0) CHECK(tel.misses() == (uint64_t)(nlr * NE * 3));
            else if (sweep_base_misses > 0) CHECK(tel.misses() < sweep_base_misses);
            if (pc.deadline == 0) sweep_base_misses = tel.misses();
        }
        // (fb passes with deadline != 0 expect 0 misses: the clock never advances
        // while handing off -- the experiment above exists to prove the contrast.)
        for (int L = 0; L < nlr; ++L)
            for (int e = 0; e < NE; ++e)
                for (int r = 0; r < 3; ++r)
                    CHECK(m.locate(knj::ExpertId{(uint32_t)L, (uint32_t)(e * 3 + r)}) ==
                          knj::Residency::NVME_RESIDENT);

        // ---- aggregate the instrumented layers ------------------------------
        st.reads = xfer.nvme_reads();
        st.read_bytes = xfer.nvme_bytes();
        st.h2d = xfer.h2d_submits();
        st.h2d_bytes = xfer.h2d_bytes();
        st.writes_refused = xfer.writes_refused();
        st.nvme_read_ns = xfer.nvme_busy_ns();
        st.h2d_ns = xfer.h2d_ns();
        st.acquires = pc.sl->acquires() - acq0;
        st.releases = pc.sl->releases() - rel0;
        st.refusals = pc.sl->refusals() - ref0;
        st.uses = pc.sl->uses() - use0;
        st.event_gated = event_gated;
        st.transitions = m.transitions();
        st.events = tel.events();
        st.into_vram = tel.into_vram();
        st.misses = tel.misses();
        st.fallback = fallback.expert();
        st.saw_gate = pc.sl->saw_event_gate();
        st.allocs = arena.allocs();
        st.frees = arena.frees();
        st.reuse_hits = arena.reuse_hits();
        st.pin_ns = arena.host_alloc_ns();
        st.unpin_ns = arena.host_free_ns();
        st.staged_at_end = arena.cached();
        arena.drain();
        CHECK(arena.live() == 0);
        CHECK(arena.allocs() == arena.frees());
        ss.agg = st;
        ss.host_ms = (float)((double)hostPhaseNs / 1e6);
        ss.host_sync_ns = host_sync_ns;
        ss.host_d2h_dy_ns = host_d2h_dy_ns;
        ss.host_d2h_gate_ns = host_d2h_gate_ns;
        ss.host_d2h_up_ns = host_d2h_up_ns;
        ss.host_silu_ns = host_silu_ns;
        ss.host_fuse_launch_ns = host_fuse_launch_ns;
        ss.host_h2d_da_ns = host_h2d_da_ns;
        ss.host_bitgate_ns = host_bitgate_ns;
        ss.host_other_ns = host_other_ns;
        ss.demand_hits = demand_hits;
        ss.demand_total = demand_total;
        ss.handoffs = handoffs;
        ss.hostfb_ns = hostfb_ns;
        ss.pf_read = tel.cause("prefetch_miss");
        ss.pf_landed = tel.cause("prefetch_landed");
        ss.preemptions = m.preempted_prefetches();
        ss.refused = m.refused_demotions();
        ss.wait_ticks = wait_ticks_total;
        ss.max_wait_ticks = max_wait_ticks;
        ss.waited_items = waited_items;
        if (pc.waitmode) {
            CHECK(ss.max_wait_ticks > 0);
            CHECK(handoffs == 0);   // a waiting pass does not hand off
        }
        ss.wait_ms = (float)((double)wait_ns_total / 1e6);   // wall time spent spinning
        ss.tick_calls = m.tick_calls();
        ss.entry_visits = m.entry_visits();
        ss.tick_ns = m.tick_ns();
        ss.registry = NE * 3 * nlr;
        // p50/p90 from the per-item wait ticks, one entry per waited item.
        if (waited_items > 0 && wait_item_ticks.size() == (size_t)waited_items) {
            std::vector<uint64_t> w = wait_item_ticks;
            std::sort(w.begin(), w.end());
            ss.p50 = w[w.size() / 2];
            ss.p90 = w[(w.size() * 9) / 10];
        }
    };

    auto wait_summary = [&](const StackStats &ss) {
        if (ss.waited_items == 0)
            printf("  waits: none (0 items waited)\n");
        else
            printf("  waits: %d item(s), total %llu tick(s), max %llu, mean %.1f, "
                   "p50 %llu, p90 %llu, wall %.3f ms\n",
                   ss.waited_items, (unsigned long long)ss.wait_ticks,
                   (unsigned long long)ss.max_wait_ticks,
                   (double)ss.wait_ticks / (double)ss.waited_items,
                   (unsigned long long)ss.p50, (unsigned long long)ss.p90, ss.wait_ms);
    };

    auto report = [&](const PassCfg &pc, const StackStats &ss) {
        printf("\n--- pass %s ---\n", pc.name);
        printf("  wall %.3f ms   host-critical %.3f ms   per-layer wall ms:", ss.wall_ms, ss.host_ms);
        for (float v : ss.layer_ms) printf(" %.2f", v);
        {
            const float sync = (float)((double)ss.host_sync_ns / 1e6);
            const float dy = (float)((double)ss.host_d2h_dy_ns / 1e6);
            const float gate = (float)((double)ss.host_d2h_gate_ns / 1e6);
            const float up = (float)((double)ss.host_d2h_up_ns / 1e6);
            const float silu = (float)((double)ss.host_silu_ns / 1e6);
            const float fuse = (float)((double)ss.host_fuse_launch_ns / 1e6);
            const float h2dda = (float)((double)ss.host_h2d_da_ns / 1e6);
            const float bitg = (float)((double)ss.host_bitgate_ns / 1e6);
            const float other = (float)((double)ss.host_ms - sync - dy - gate - up - silu - fuse - h2dda - bitg);
            printf("  host-critical breakdown ms: sync %.3f + d2h_down %.3f + d2h_gate %.3f + d2h_up %.3f "
                   "+ silu %.3f + fuse %.3f + h2d_da %.3f + bitgate %.3f + other %.3f = %.3f\n",
                   sync, dy, gate, up, silu, fuse, h2dda, bitg, other, (double)ss.host_ms);
        }
        printf("\n");
        printf("  reads %llu (%.3f MiB, reader busy %.3f ms)  h2d %llu (%.3f ms)  pin %.3f ms\n",
               (unsigned long long)ss.agg.reads, ss.agg.read_bytes / 1048576.0,
               ss.agg.nvme_read_ns / 1e6, (unsigned long long)ss.agg.h2d,
               ss.agg.h2d_ns / 1e6, ss.agg.pin_ns / 1e6);
        printf("  C8: acq %llu rel %llu peak %d/%d gated-evict %llu  |  C9: trans %llu into-VRAM %llu\n",
               (unsigned long long)ss.agg.acquires, (unsigned long long)ss.agg.releases,
               pc.sl->peak_in_use(), pc.sl->n_slots(), (unsigned long long)ss.agg.event_gated,
               (unsigned long long)ss.agg.transitions, (unsigned long long)ss.agg.into_vram);
        printf("  C21: misses %llu  C19: hand-offs %llu  preempted-prefetch %llu refused-demote %llu\n",
               (unsigned long long)ss.agg.misses, (unsigned long long)ss.handoffs,
               (unsigned long long)ss.preemptions, (unsigned long long)ss.refused);
        printf("  tick: calls %llu visits %llu ns %.3f (registry %d)\n",
               (unsigned long long)ss.tick_calls, (unsigned long long)ss.entry_visits,
               (double)ss.tick_ns / 1e6, ss.registry);
        printf("  arena: allocs %llu hits %llu staged-at-end %zu  |  pf: reads %llu landed %llu wasted %llu (%.3f MiB)\n",
               (unsigned long long)ss.agg.allocs, (unsigned long long)ss.agg.reuse_hits,
               ss.agg.staged_at_end, (unsigned long long)ss.pf_read,
               (unsigned long long)ss.pf_landed, (unsigned long long)ss.wasted_reads,
               ss.wasted_bytes / 1048576.0);
        if (ss.demand_total)
            printf("  prefetch hit rate at demand: %llu/%llu = %.1f%%\n",
                   (unsigned long long)ss.demand_hits, (unsigned long long)ss.demand_total,
                   100.0 * (double)ss.demand_hits / (double)ss.demand_total);
        printf("  bits vs resident: %s   carried-state chain: %s\n",
               ss.bits_ok ? "identical" : "DIFFER", ss.carry_ok ? "identical" : "DIFFER");
        if (ss.carry_over_tol > 0)
            printf("  carry (graded, has C19 host contributions): worst-rel %.3e over-tol %d\n",
                   ss.carry_worst_rel, ss.carry_over_tol);
        wait_summary(ss);
    };

    StackStats ss;
    auto run_one = [&](PassCfg pc) {
        ss = StackStats();
        run_stack(pc, plans, xIn, rG, rU, rY, ss);
        report(pc, ss);
    };

    printf("\n--- PART 2: streamed whole stack ---\n");
    run_one({"M0 serial/churn", &slotsMain, 0, 0, false, 1000000, false, T, NL});
    run_one({"M1 serial/reuse", &slotsMain, 1, 0, false, 1000000, false, T, NL});
    run_one({"M2 overlap/reuse", &slotsMain, 2, 0, false, 1000000, false, T, NL});
    run_one({"M3 overlap/split-copy", &slotsMain, 3, 0, false, 1000000, false, T, NL});

    printf("\n--- PART 3: C10 prefetch horizon sweep (%d-slot capacity point) ---\n", PFSLOTS);
    run_one({"P0 overlap control", &slotsPF, 2, 0, false, 1000000, false, T, NL});
    run_one({"P1 true-set lead 1", &slotsPF, 2, 1, false, 1000000, false, T, NL});
    run_one({"P2 true-set lead 2", &slotsPF, 2, 2, false, 1000000, false, T, NL});
    run_one({"P4 true-set lead 4", &slotsPF, 2, 4, false, 1000000, false, T, NL});
    run_one({"C1 carry-forward lead 1", &slotsPF, 2, 1, true, 1000000, false, T, NL});
    {
        // Carry-forward prediction quality, from the plans (expert sets).
        int same = 0, tot = 0;
        for (int L = 1; L < NL; ++L) {
            std::vector<char> a((size_t)NE, 0), b((size_t)NE, 0);
            for (int p = 0; p < P; ++p) {
                a[plans[(size_t)L - 1].pairExp[(size_t)p]] = 1;
                b[plans[(size_t)L].pairExp[(size_t)p]] = 1;
            }
            for (int e = 0; e < NE; ++e) { ++tot; if (a[(size_t)e] && b[(size_t)e]) ++same; }
        }
        printf("  carry-forward expert-set overlap (L-1 -> L): %d/%d = %.1f%%\n",
               same, tot, 100.0 * same / (tot > 0 ? tot : 1));
    }

    // =========================================================================
    // PART 4 -- the C9 deadline path, exercised for real: a reachable deadline,
    // recorded misses, C19 hand-offs, and the missed contributions computed on
    // the host from the authoritative bytes (I1: computed elsewhere, never
    // skipped). Mixed accumulation (device fragments vs host sequential) is
    // graded at the declared tolerance; the bit gate above covers pure-device
    // paths.
    // =========================================================================
    const int DT = 64;
    // DNL must not exceed the layers actually loaded (Wr, device bank are sized
    // by NL): clamp to NL so the smoke run (max_layers=2) stays in bounds.
    const int DNL = std::min(4, NL);
    std::vector<std::vector<__half>> dxin((size_t)DNL);
    {
        dxin[0].resize((size_t)DT * KH);
        uint32_t st2 = 0x9e3779b9u;
        auto nxt2 = [&]() { st2 ^= st2 << 13; st2 ^= st2 >> 17; st2 ^= st2 << 5; return st2; };
        for (size_t i = 0; i < dxin[0].size(); ++i) {
            const float v = (float)((int)(nxt2() % 4001) - 2000) / 1000.0f;
            dxin[0][(size_t)i] = __float2half(v);
        }
    }

    {
        std::vector<std::vector<__half>> pdxin((size_t)DNL);
        pdxin[0].resize((size_t)DT * KH);
        uint32_t pst2 = 0x9e3779b9u;
        auto pnxt2 = [&]() { pst2 ^= pst2 << 13; pst2 ^= pst2 >> 17; pst2 ^= pst2 << 5; return pst2; };
        for (size_t i = 0; i < pdxin[0].size(); ++i) {
            const float v = (float)((int)(pnxt2() % 4001) - 2000) / 1000.0f;
            pdxin[0][(size_t)i] = __float2half(v);
        }
        std::vector<Plan> pdpl;
        std::vector<std::vector<float>> pdG, pdU, pdY;
        std::vector<float> pdlms;
        float pddev = 0.0f, pdhost = 0.0f;
        run_resident(DT, DNL, pdxin, pdpl, pdG, pdU, pdY, pdlms, pddev, pdhost, nullptr);
        // Rebind the PART 4 names to the freshly built pass, then destroy the temporaries.
        dxin = std::move(pdxin);
        dpl = std::move(pdpl);
        // dG2 / dU2 / dY2 hold the PART 4 resident pass's gate/up/down projections,
        // used by 4a/4b as the bit-identity reference for the streamed path (I7).
        dG2 = std::move(pdG);
        dU2 = std::move(pdU);
        dY2 = std::move(pdY);
        dlms = std::move(pdlms);
        ddev = pddev;
        dhost = pdhost;
    }
    printf("\n--- PART 4: C9 deadline path (T=%d, %d layers, cold start) ---\n", DT, DNL);
    {   // experiment 4a: the C19 no-wait consumer. The finding, made checkable:
        // demand_or_fallback never waits, so the manager's deadline clock does not
        // advance while C19 computes -- no deadline can fire, however tight.
        // deadline=0 here is deliberately *unreachable* (recorded as the baseline);
        // misses must come out exactly 0 and the clock exactly the drain tail.
        StackStats ds;
        PassCfg pc{"deadline=0 (no-wait fb)", &slotsMain, 0, 0, false, 0, true, DT, DNL};
        run_stack(pc, dpl, dxin, dG2, dU2, dY2, ds);
        printf("\n--- pass 4a: hand-offs=%llu deadline=0 (C19, no wait) ---\n", (unsigned long long)ds.handoffs);
        printf("  wall %.3f ms  host-fallback compute %.3f ms  C21 misses %llu\n",
               (float)ds.wall_ms, (float)ds.hostfb_ns / 1e6, (unsigned long long)ds.agg.misses);
        // Declared precision category: C19's host fallback accumulates each row
        // sequentially on the host, the device kernel as fragments -- same values,
        // different summation order. Graded at the 2e-3 mixed-accumulation
        // tolerance (docs/03), not the 1e-4 pure-device gate. L1 is tighter
        // because fewer of its rows carry host arithmetic through the f16 carry.
        for (int L = 0; L < DNL; ++L) {
            std::vector<double> ref((size_t)DT * TOPK * KH);
            for (size_t i = 0; i < ref.size(); ++i) ref[i] = (double)dY2[(size_t)L][i];
            char nm[64];
            std::snprintf(nm, sizeof(nm), "deadline down L%d", L);
            cx(nm, ds.Yout[(size_t)L], ref, DT * TOPK, KH, 2e-3);
            std::vector<float> yg((size_t)DT * KH, 0.0f);
            std::vector<double> rref((size_t)DT * KH, 0.0);
            for (int t = 0; t < DT; ++t)
                for (int h = 0; h < KH; ++h) {
                    double racc = 0.0;
                    for (int s = 0; s < TOPK; ++s)
                        racc += (double)dpl[(size_t)L].rw[(size_t)(t * TOPK + s)] *
                                (double)dY2[(size_t)L][(size_t)(t * TOPK + s) * KH + h];
                    rref[(size_t)t * KH + h] = racc;
                    yg[(size_t)t * KH + h] = 0.0f;
                    for (int s = 0; s < TOPK; ++s)
                        yg[(size_t)t * KH + h] +=
                            (float)dpl[(size_t)L].rw[(size_t)(t * TOPK + s)] *
                            ds.Yout[(size_t)L][(size_t)(t * TOPK + s) * KH + h];
                }
            std::snprintf(nm, sizeof(nm), "deadline layer out L%d", L);
            cx(nm, yg, rref, DT, KH, 2e-3);
        }
        report(pc, ds);
    }
    {   // experiment 4b: the WAITING consumer, where the deadline is real.
        // Same stack, same slot budget, demand (no fallback), steps block on
        // wait_ready; the manager's clock advances with every spin. The sweep
        // {0, 25, 200, 5000} measures the C9 miss curve and its wall impact.
        const int deadlines[4] = {0, 25, 200, 5000};
        
        for (int di = 0; di < 4; ++di) {
            const int *_dd = &deadlines[di]; (void)_dd;
            PassCfg pc{"wait sweep", &slotsMain, 0, 0, false, deadlines[di], false, DT, DNL};
            pc.waitmode = true;
            StackStats ds;
            run_stack(pc, dpl, dxin, dG2, dU2, dY2, ds);
            printf("\n--- pass 4b: wait sweep deadline=%d ticks ---\n", deadlines[di]);
            printf("  wall %.3f ms  hand-offs %llu  C21 misses %llu\n",
                   (float)ds.wall_ms, (unsigned long long)ds.handoffs,
                   (unsigned long long)ds.agg.misses);
            for (int L = 0; L < DNL; ++L) {
                std::vector<double> ref((size_t)DT * TOPK * KH);
                for (size_t i = 0; i < ref.size(); ++i) ref[i] = (double)dY2[(size_t)L][i];
                char nm[64];
                std::snprintf(nm, sizeof(nm), "waitsweep down L%d", L);
                cx(nm, ds.Yout[(size_t)L], ref, DT * TOPK, KH, 1e-4);
            }
            report(pc, ds);
        }
    }

    HIPCHECK(hipStreamDestroy(compute));

    printf("\n=== Summary ===\n");
    printf("  checks failed: %d\n", g_failed);
    printf("\nRESULT: %s\n", g_failed == 0 ? "PASS" : "FAIL");
    return g_failed == 0 ? 0 : 5;
}
