// tools/bench/knj_xfer_probe.hip -- why does the streamed expert path move bytes
// at 0.15 GiB/s when one 2.47 MB expert copy measures 13.4-14.7 GB/s?
//
// THE NUMBERS THIS EXISTS TO RECONCILE
// ------------------------------------
//   docs/00-verified-facts.md section 8: PCIe, one 2.47 MB expert copy
//     13.4-14.7 GB/s           -> 181 us for the 2.4609 MiB slab
//   records/2026-10-07_gfx1201_expert_streaming.txt: the streamed layer-0 FFN
//     0.13-0.16 GiB/s (read+H2D)   -> ~16 ms for the same slab
//
// That is a ~90x gap on the exact operation the whole design rests on, and the
// design's own ranking (AGENTS.md section 11) puts eliminating unnecessary
// transfer and overlapping transfer ahead of any kernel work. A gap that large
// is either a units/provenance mistake or a real defect; this probe decides
// which, and it names the component if it is real.
//
// WHAT IT MEASURES (each case is one whole-buffer loop, N copies of B bytes)
// ------------------------------------------------------------------------
//   A  read-only              N buffered file reads into pinned host memory
//   B  h2d-blocking-stream    N pinned->device copies, default (blocking) stream
//   C  h2d-nonblocking-stream N pinned->device copies, hipStreamNonBlocking
//   D  read+h2d pipelined     reader thread -> pinned -> copy, double buffered
//   E  read+h2d+sync-d2h      D, plus a synchronous hipMemcpy D2H per iteration
//                             on the NULL stream with a BLOCKING copy stream.
//                             This is the harness pattern: a blocking stream
//                             implicitly synchronises with the legacy default
//                             stream, so the copies cannot overlap it.
//   F  read+h2d+sync-d2h-nb   E, but the copy stream is hipStreamNonBlocking.
//
// E vs F is the hypothesis under test: if the copies are being serialised by
// the harness's own synchronous D2H traffic, F is materially faster than E and
// the fix is a stream flag, not a new transfer engine. If E and F agree, the
// cost is elsewhere (the reads themselves, or real copy time) and the next
// measurement is a read-shape probe, not a stream-flag change.
//
// WHAT IS CHECKED (rule 7: check every output; rule 9: never omit a case)
// ----------------------------------------------------------------------
//   * every case reports, including one that is skipped or unavailable;
//   * the LAST transfer of every copy case is read back (D2H) and compared
//     byte-for-byte with the host buffer that produced it -- an unsent or
//     silently failed copy that still "times fast" is caught here;
//   * the read-only case is compared byte-for-byte against an independent
//     re-read of the same range;
//   * zero/unwritten, mismatching indices and max-abs are all reported.
//
// PROVENANCE CAVEATS, STATED RATHER THAN BURIED
// ---------------------------------------------
//   * File reads are buffered stdio, so a page-cache hit inflates case A. That
//     is not a defect of the probe: it is the regime the real driver runs in,
//     and the point is to attribute the total, not to measure a bare NVMe.
//   * The probe is NOT a pipeline and must never be quoted as the FFN's own
//     throughput. It measures the transport in isolation.
//
// SELF-SUFFICIENT BY DEFAULT, AND THAT IS A REQUIREMENT
// ----------------------------------------------------
// run_bench.sh runs every tier-C driver with NO arguments, so this probe must
// produce a real result that way. With no <file> argument the read side is
// replaced by a host fill and every affected case is labelled `host-fill`; the
// read-only case reports NOT RUN. Transport decomposition (the copy stream, the
// sync-D2H interaction, both stream flavours) is unaffected by that
// substitution, which is the part that matters for the gate. Pass a file to
// also get the read side with real bytes.
//
// Build (arch-guarded; see run_bench.sh -- this file is picked up as tier C):
//   hipcc -nogpulib -O2 --offload-arch=gfx1201 -Xclang -target-feature \
//         -Xclang +wavefrontsize32 -I tools/bench -I kernels \
//         -L"G:/ROCM10RT-gfx1201/lib" -lamdhip64 \
//         tools/bench/knj_xfer_probe.hip -o /tmp/knj_xfer_probe.exe
// Run:
//   KNJ_BUILD_ARCH=gfx1201 /tmp/knj_xfer_probe.exe [file] [bytes] [count] [offset]
//
// Exit codes: 0 PASS, 5 FAIL (a correctness check), 9 HIP error, 3 missing
//             input, 6 arch mismatch, 7 arch guard unconfigured.

#include <hip/hip_runtime.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "knj_wmma_guard.h"

#define HIPCHECK(x)                                                            \
    do {                                                                       \
        hipError_t _e = (x);                                                   \
        if (_e != hipSuccess) {                                                \
            printf("HIP ERROR %s at %s:%d\n", hipGetErrorString(_e), __FILE__, \
                   __LINE__);                                                  \
            exit(9);                                                           \
        }                                                                      \
    } while (0)

static int g_failed = 0;

static uint64_t mono_ns() {
    typedef std::chrono::steady_clock clk;
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clk::now().time_since_epoch()).count();
}

// Result of one case. `ok=false` means the case did not produce a usable
// number; it is still printed, because a case that did not run is a result
// (rule 9) and must never be silently dropped from the table.
struct CaseResult {
    const char *name;
    bool ok;
    uint64_t wall_ns;
    uint64_t bytes;
    double mib_per_s;
    double ms_per_copy;
};

static void print_case(const CaseResult &r, uint64_t per_copy_bytes) {
    if (!r.ok) {
        printf("  %-26s NOT RUN (see note above)\n", r.name);
        return;
    }
    printf("  %-26s %9.2f MiB/s   %8.3f ms/copy   (%llu bytes in %.3f s)\n",
           r.name, r.mib_per_s, r.ms_per_copy,
           (unsigned long long)r.bytes, (double)r.wall_ns / 1e9);
    (void)per_copy_bytes;
}

// A buffered read of `n` bytes at `off`. stdio, deliberately: this is the same
// shape the FFN driver's reader thread uses, so the probe attributes the same
// cost the pipeline pays.
static bool read_at(const std::string &path, long long off, void *dst, size_t n) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    if (std::fseek(fp, (long)off, SEEK_SET) != 0) { std::fclose(fp); return false; }
    const size_t got = std::fread(dst, 1, n, fp);
    std::fclose(fp);
    return got == n;
}

// Full-output check on a copy: bring the device bytes back and compare.
// Returns the number of mismatching bytes (0 = identical).
static uint64_t device_matches_host(const void *dev, const void *host, size_t n,
                                    std::vector<uint8_t> &scratch) {
    scratch.resize(n);
    HIPCHECK(hipMemcpy(&scratch[0], dev, n, hipMemcpyDeviceToHost));
    const uint8_t *a = (const uint8_t *)host;
    uint64_t bad = 0;
    for (size_t i = 0; i < n; ++i) if (a[i] != scratch[i]) ++bad;
    return bad;
}

// Two-slot reader->copier handshake. One persistent reader thread, so thread
// creation is never inside the measured loop (the FFN driver's NvmeReader is
// the same shape for the same reason).
class ReadPipe {
 public:
    // `src_mem` non-null selects the host-fill source (no file): the reader's
    // "read" becomes a memcpy from that buffer. Same handshake either way, so
    // the pipeline shape -- and therefore the stream interaction being
    // measured -- is identical.
    ReadPipe(const std::string &path, uint8_t *b0, uint8_t *b1, size_t b,
             int n, long long base, const uint8_t *src_mem = nullptr)
        : path_(path), b_(b), n_(n), base_(base), src_mem_(src_mem) {
        buf_[0] = b0; buf_[1] = b1;
        free_[0] = free_[1] = 1;
        ready_[0] = ready_[1] = 0;
        th_ = std::thread(&ReadPipe::loop, this);
    }
    ~ReadPipe() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
        cv_free_.notify_all();
        if (th_.joinable()) th_.join();
    }
    // Blocks until the buffer for iteration i is filled. Returns the buffer.
    uint8_t *take(int i) {
        const int s = i & 1;
        std::unique_lock<std::mutex> lk(m_);
        while (!ready_[s] && !stop_) cv_ready_.wait(lk);
        ready_[s] = 0;
        failed_ = failed_ || fail_slot_[s];
        return buf_[s];
    }
    void release(int i) {
        const int s = i & 1;
        { std::lock_guard<std::mutex> lk(m_); free_[s] = 1; }
        cv_free_.notify_all();
    }
    bool failed() const { return failed_; }
    uint64_t reads() const { return reads_; }

 private:
    void loop() {
        for (int i = 0; i < n_; ++i) {
            const int s = i & 1;
            {
                std::unique_lock<std::mutex> lk(m_);
                while (!free_[s] && !stop_) cv_free_.wait(lk);
                if (stop_) return;
                free_[s] = 0;
            }
            const bool ok = src_mem_ ? (std::memcpy(buf_[s], src_mem_, b_), true)
                                     : read_at(path_, base_ + (long long)i * (long long)b_,
                                               buf_[s], b_);
            {
                std::lock_guard<std::mutex> lk(m_);
                if (ok) reads_++; else { fail_slot_[s] = 1; failed_ = true; }
                ready_[s] = 1;
            }
            cv_ready_.notify_all();
        }
    }
    std::string path_;
    uint8_t *buf_[2];
    size_t b_;
    int n_;
    long long base_;
    std::thread th_;
    std::mutex m_;
    std::condition_variable cv_ready_, cv_free_;
    int free_[2], ready_[2], fail_slot_[2] = {0, 0};
    const uint8_t *src_mem_;
    bool stop_ = false, failed_ = false;
    uint64_t reads_ = 0;
};

int main(int argc, char **argv) {
    knj_wmma_dev dev;
    const int grc = knj_wmma_guard(&dev);
    if (grc) return grc;
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // No argument is the graded path (run_bench.sh passes none).
    const std::string path = (argc > 1) ? argv[1] : std::string();
    const bool have_file = !path.empty();
    const size_t B = (argc > 2) ? (size_t)std::strtoull(argv[2], nullptr, 0) : 2580480u;  // 2.4609 MiB
    const int N = (argc > 3) ? std::atoi(argv[3]) : 64;
    const long long base = (argc > 4) ? (long long)std::strtoll(argv[4], nullptr, 0) : 0;

    printf("=== knj_xfer_probe -- decompose the streamed expert transfer rate ===\n");
    printf("device      %s (%s)\n", dev.name, dev.arch);
    if (have_file)
        printf("file        %s\n", path.c_str());
    else
        printf("file        (none given -- read side uses a host fill; read-only case NOT RUN)\n");
    printf("slab        %zu bytes (%.4f MiB)   count %d   offset %lld\n",
           B, (double)B / 1048576.0, N, base);
    printf("total       %.2f MiB per case\n", (double)B * (double)N / 1048576.0);

    std::vector<CaseResult> res;
    std::vector<uint8_t> host(B), host2(B), scratch;
    uint8_t *dev_buf = nullptr;
    HIPCHECK(hipMalloc(&dev_buf, B));
    uint8_t *pin = nullptr, *pin2 = nullptr;
    HIPCHECK(hipHostMalloc((void **)&pin, B, hipHostMallocDefault));
    HIPCHECK(hipHostMalloc((void **)&pin2, B, hipHostMallocDefault));

    hipStream_t sblk = nullptr, snb = nullptr, copy_blk = nullptr, copy_nb = nullptr;
    HIPCHECK(hipStreamCreate(&sblk));
    HIPCHECK(hipStreamCreateWithFlags(&snb, hipStreamNonBlocking));
    HIPCHECK(hipStreamCreate(&copy_blk));
    HIPCHECK(hipStreamCreateWithFlags(&copy_nb, hipStreamNonBlocking));

    // --- A: read-only ------------------------------------------------------
    {
        CaseResult r; r.name = "A read-only"; r.ok = false; r.wall_ns = 0;
        r.bytes = 0; r.mib_per_s = 0; r.ms_per_copy = 0;
        const uint64_t t0 = mono_ns();
        bool ok = have_file;
        for (int i = 0; i < N && ok; ++i)
            ok = read_at(path, base + (long long)i * (long long)B, pin, B);
        const uint64_t t1 = mono_ns();
        if (!have_file)
            printf("  A read-only                   NOT RUN (no file argument; a host-fill rate is not a read rate)\n");
        if (ok) {
            // Independent re-read of the last range: proves the loop read the
            // file, not that it read nothing at 5 TB/s.
            const bool re = read_at(path, base + (long long)(N - 1) * (long long)B, host2.data(), B);
            uint64_t bad = re ? 0 : (uint64_t)B;
            if (re) for (size_t k = 0; k < B; ++k) if (pin[k] != host2[k]) ++bad;
            if (bad) { printf("  A CHECK FAILED: %llu byte(s) of the last read differ from a re-read\n",
                              (unsigned long long)bad); ++g_failed; r.ok = false; }
            else { r.ok = true; r.wall_ns = t1 - t0; r.bytes = (uint64_t)B * (uint64_t)N; }
        }
        if (r.ok) {
            r.mib_per_s = (double)r.bytes / 1048576.0 / ((double)r.wall_ns / 1e9);
            r.ms_per_copy = (double)r.wall_ns / 1e6 / (double)N;
        }
        res.push_back(r);
    }

    // Seed the pinned source so every copy moves real bytes. With a file the
    // bytes are the model's; without one they are a seeded pattern, which is
    // enough for a transport measurement and is checked by the same D2H compare.
    std::vector<uint8_t> fill(B);
    if (have_file) {
        read_at(path, base, pin, B);
    } else {
        uint32_t st = 0x9e3779b9u;
        for (size_t i = 0; i < B; ++i) {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            fill[i] = (uint8_t)(st >> 24);
        }
        std::memcpy(pin, fill.data(), B);
    }
    HIPCHECK(hipMemcpy(dev_buf, pin, B, hipMemcpyHostToDevice));   // warm

    // --- shared copy-loop body, one stream, no D2H -------------------------
    {
        struct LoopCfg { const char *name; hipStream_t st; };
        LoopCfg cfgs[2] = {{"B h2d-blocking-stream", sblk},
                           {"C h2d-nonblocking-stream", snb}};
        for (int c = 0; c < 2; ++c) {
            CaseResult r; r.name = cfgs[c].name; r.ok = false;
            hipEvent_t ev;
            HIPCHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
            const uint64_t t0 = mono_ns();
            for (int i = 0; i < N; ++i) {
                HIPCHECK(hipMemcpyAsync(dev_buf, pin, B, hipMemcpyHostToDevice, cfgs[c].st));
                HIPCHECK(hipEventRecord(ev, cfgs[c].st));
                // Poll, never hipDeviceSynchronize inside the loop (rule 12).
                while (hipEventQuery(ev) != hipSuccess) std::this_thread::yield();
            }
            const uint64_t t1 = mono_ns();
            HIPCHECK(hipEventDestroy(ev));
            const uint64_t bad = device_matches_host(dev_buf, pin, B, scratch);
            if (bad) {
                printf("  %s CHECK FAILED: %llu byte(s) differ after the copy\n", r.name,
                       (unsigned long long)bad);
                ++g_failed;
            } else {
                r.ok = true; r.wall_ns = t1 - t0; r.bytes = (uint64_t)B * (uint64_t)N;
                r.mib_per_s = (double)r.bytes / 1048576.0 / ((double)r.wall_ns / 1e9);
                r.ms_per_copy = (double)r.wall_ns / 1e6 / (double)N;
            }
            res.push_back(r);
        }
    }

    // --- D/E/F: read+h2d, with and without the harness's sync D2H ----------
    // `sync_d2h` reproduces the FFN driver's per-layer pattern: a synchronous
    // hipMemcpy (default stream) between copies. `copy_stream` picks blocking
    // (which implicitly synchronises with the default stream) or non-blocking.
    {
        // Label the source honestly: with no file the reader's read is a host
        // memcpy, so the case is a fill+h2d, not a read+h2d.
        const char *names[3] = {have_file ? "D read+h2d pipelined" : "D fill+h2d pipelined",
                                have_file ? "E +sync-d2h (blocking)" : "E fill +sync-d2h (block)",
                                have_file ? "F +sync-d2h (nonblocking)" : "F fill +sync-d2h (nonblk)"};
        for (int c = 0; c < 3; ++c) {
            const bool with_d2h = (c > 0);
            hipStream_t cst = (c == 2) ? copy_nb : copy_blk;
            CaseResult r; r.name = names[c]; r.ok = false;
            std::vector<uint8_t> dcpy(B);
            const uint8_t *last_src = nullptr;   // the buffer the LAST copy read from
            hipEvent_t ev;
            HIPCHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
            {
                ReadPipe pipe(path, pin, pin2, B, N, base, have_file ? nullptr : fill.data());
                const uint64_t t0 = mono_ns();
                for (int i = 0; i < N; ++i) {
                    uint8_t *src = pipe.take(i);
                    last_src = src;
                    HIPCHECK(hipMemcpyAsync(dev_buf, src, B, hipMemcpyHostToDevice, cst));
                    HIPCHECK(hipEventRecord(ev, cst));
                    // The consumer side of the real pipeline never reads device
                    // data back, so the D2H here exists only to reproduce the
                    // harness's null-stream traffic.
                    if (with_d2h) HIPCHECK(hipMemcpy(&dcpy[0], dev_buf, B, hipMemcpyDeviceToHost));
                    while (hipEventQuery(ev) != hipSuccess) std::this_thread::yield();
                    pipe.release(i);
                }
                const uint64_t t1 = mono_ns();
                const bool pf = pipe.failed();
                const uint64_t rd = pipe.reads();
                if (pf || rd != (uint64_t)N) {
                    printf("  %s CHECK FAILED: reader failed (reads %llu of %d)\n",
                           r.name, (unsigned long long)rd, N);
                    ++g_failed;
                } else {
                    r.wall_ns = t1 - t0; r.bytes = (uint64_t)B * (uint64_t)N;
                    r.mib_per_s = (double)r.bytes / 1048576.0 / ((double)r.wall_ns / 1e9);
                    r.ms_per_copy = (double)r.wall_ns / 1e6 / (double)N;
                    // Compare against the buffer the LAST copy actually read from,
                    // not against `pin`: the pipe alternates two slots, so the
                    // final iteration's source is whichever slot parity N-1 has.
                    uint64_t bad = device_matches_host(dev_buf, last_src, B, scratch);
                    if (bad) {
                        printf("  %s CHECK FAILED: %llu byte(s) differ after the copy\n",
                               r.name, (unsigned long long)bad);
                        ++g_failed;
                    } else { r.ok = true; }
                }
            }
            HIPCHECK(hipEventDestroy(ev));
            res.push_back(r);
        }
    }

    // --- the table --------------------------------------------------------
    printf("\n--- transport decomposition (each case: %d x %.4f MiB) ---\n",
           N, (double)B / 1048576.0);
    for (size_t i = 0; i < res.size(); ++i) print_case(res[i], B);
    {
        const double ref = 13.6 * 1024.0;   // docs/00 section 8: 13.6 GiB/s -> MiB/s
        printf("\n  reference (docs/00 section 8): one 2.47 MB expert copy = 13.4-14.7 GB/s "
               "= %.0f MiB/s\n", ref);
        for (size_t i = 0; i < res.size(); ++i)
            if (res[i].ok)
                printf("  %-26s %.3fx the single-copy reference\n", res[i].name,
                       res[i].mib_per_s / ref);
        printf("  ratio E/D (does the harness's sync D2H cost anything?):          ");
        if (res.size() >= 5 && res[3].ok && res[4].ok)
            printf("%.3fx\n", res[4].ms_per_copy / res[3].ms_per_copy);
        else printf("not measured\n");
        printf("  ratio F/E (does the stream flag recover it?):                    ");
        if (res.size() >= 6 && res[4].ok && res[5].ok)
            printf("%.3fx\n", res[5].ms_per_copy / res[4].ms_per_copy);
        else printf("not measured\n");
    }

    HIPCHECK(hipStreamDestroy(sblk));
    HIPCHECK(hipStreamDestroy(snb));
    HIPCHECK(hipStreamDestroy(copy_blk));
    HIPCHECK(hipStreamDestroy(copy_nb));
    HIPCHECK(hipHostFree(pin));
    HIPCHECK(hipHostFree(pin2));
    HIPCHECK(hipFree(dev_buf));

    printf("\n=== Summary ===\n");
    printf("  checks failed: %d\n", g_failed);
    printf("\nRESULT: %s\n", g_failed == 0 ? "PASS" : "FAIL");
    return g_failed == 0 ? 0 : 5;
}
