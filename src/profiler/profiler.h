// src/profiler/profiler.h -- C21 profiling and telemetry.
//
// Owns: the attribution the engine is judged by. `docs/06-profiling.md` is the
// contract, and this file is the subset of it that can be measured honestly
// today. Two rules from that document are load-bearing here:
//
//  1. `dev` and `idle` are never folded together. An op that is 95% of device
//     time is not the same problem as an op that is 95% of wall time and 3% of
//     device time, and one line of the table has to show the difference.
//  2. `--profiling` never changes what the engine computes. It changes what it
//     measures, and it says so in its output header.
//
// THE CLOCK DOMAIN IS DECLARED, NOT ASSUMED
// -----------------------------------------
// `dev us` in the contract is GPU busy time from paired device events. A
// host-only run has no device events, so this profiler records a ClockDomain and
// every report prints it. On ClockDomain::Host, `dev` is the op's own interval
// and `host` is the same interval -- there is no asynchronous device to separate
// them, and pretending otherwise by printing 0.00 would be worse than saying so.
// When a device event backend is wired in, the same begin/end markers carry real
// event intervals and the domain flips to Device. Nothing else in the report
// changes, which is the point of keeping the columns fixed.
//
// SCOPE DISCIPLINE: NESTING IS ALLOWED, AND ROWS ARE SELF TIME
// ------------------------------------------------------------
// `kanjoos-run --profiling` opens a `prefill` scope around the model's forward
// pass, and the model opens a scope per component inside it. Attributing the
// full inclusive interval to both would count the same microseconds twice, so
// every row is **self time**: a parent's children are subtracted from it, and
// the rows therefore partition the wall time of the run instead of overlapping
// it. A container row (`prefill`, `decode`) is consequently small and the real
// work appears on the leaf rows, which is what a component table is for. The
// header states which convention is in force.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace knj {

// ---- configuration (the documented command-line surface) -------------------
struct ProfileConfig {
  bool on = false;
  // off | table | json | csv | no-subtract   (docs/06-profiling.md section 1)
  std::string format = "table";
  // class | layer | kernel
  std::string detail = "class";
  // --profile-dir; empty means "stdout only"
  std::string dir;
  // --profile-floor: empty ops timed through the same begin/end path
  int floor_ops = 256;
  // --profile-warmup: discard the first N steps entirely
  int warmup_steps = 8;
  // --profiling=no-subtract keeps the raw numbers (no floor subtraction)
  bool subtract_floor = true;
};

enum class ClockDomain { None, Host, Device };
const char* clock_domain_name(ClockDomain d);

// Residency and transfer counters. Every field is optional: a path that cannot
// measure one reports it as unmeasured rather than as zero.
struct ResidencyStats {
  bool measured = false;
  double vram_pct = 0.0, ram_pct = 0.0, nvme_pct = 0.0;
  uint64_t slots_used = 0, slots_total = 0, evictions = 0;
  double prefetch_hit = 0.0, stall_ms = 0.0;
};

struct TransferStats {
  bool measured = false;
  uint64_t h2d_bytes = 0, d2h_bytes = 0, nvme_read_bytes = 0, nvme_write_bytes = 0;
  uint64_t peak_pinned_bytes = 0;
};

class Profiler {
 public:
  static Profiler& get();

  void configure(const ProfileConfig& c);
  const ProfileConfig& config() const { return cfg_; }
  bool on() const { return cfg_.on; }

  // Marker pair. `name` may carry a scope suffix after ':' -- `class` grouping
  // strips it, `layer`/`kernel` detail keeps it ("projections:12").
  void begin(const char* name, int stream = 0);
  void end();
  // One model step finished. Feeds --profile-warmup: ops recorded before the
  // warmup count is reached are discarded, because the first steps pay for
  // page faults and thread-pool spin-up, not for the model.
  void step_done();

  void set_domain(ClockDomain d) { domain_ = d; }
  ClockDomain domain() const { return domain_; }

  void note_residency(const ResidencyStats& r) { res_ = r; }
  void note_transfer(const TransferStats& t) { xfer_ = t; }

  // Times `floor_ops` empty marker pairs through the identical begin/end path.
  // The floor is what an op would cost if it did nothing, which is the only
  // honest way to read a sub-microsecond row.
  void measure_floor();
  uint64_t floor_host_ns() const { return floor_host_ns_; }
  uint64_t floor_dev_ns() const { return floor_dev_ns_; }
  int floor_n() const { return floor_n_; }

  uint64_t steps_kept() const { return steps_kept_; }
  uint64_t steps_discarded() const { return steps_discarded_; }
  uint64_t nested_ops() const { return nested_ops_; }   // scopes measured as self time
  uint64_t max_depth() const { return max_depth_; }
  uint64_t op_count() const { return total_ops_; }

  // Header: identity, hardware, budgets, run totals. `device_line` is supplied
  // by the caller because only it knows what it opened.
  void print_header(std::FILE* f, const std::string& version, const std::string& device_line,
                    const std::string& budgets_line) const;

  void report_table(std::FILE* f) const;
  void report_json(std::FILE* f) const;
  void report_csv(std::FILE* f) const;

  // Writes profile.{txt,json,csv} into cfg_.dir when it is set, and prints the
  // paths. Returns the number of files written.
  int write_dir() const;

  // The single call a host is expected to make at exit. Honours cfg_.format.
  void report(std::FILE* f) const;

  void reset();

 private:
  Profiler() = default;
  struct Row {
    std::string group;
    uint64_t ops = 0, dev_ns = 0, idle_ns = 0, host_ns = 0;
  };
  Row* find_or_add(const std::string& group);
  std::string group_for(const std::string& name) const;
  std::vector<Row> rows_sorted() const;
  double subtract(const Row& r) const;
  std::string identity_line() const;

  // One open marker. `child_ns` accumulates the inclusive spans of the scopes
  // opened inside this one, so the parent can report self time on its own end.
  struct Frame {
    std::string group;
    uint64_t t0 = 0;
    uint64_t child_ns = 0;
  };

  ProfileConfig cfg_;
  ClockDomain domain_ = ClockDomain::None;
  std::vector<Row> rows_;
  std::vector<Frame> stack_;
  int cur_stream_ = 0;
  uint64_t idle_pending_ = 0;
  uint64_t stream_end_ns_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint64_t steps_seen_ = 0, steps_kept_ = 0, steps_discarded_ = 0;
  uint64_t nested_ops_ = 0, max_depth_ = 0, total_ops_ = 0;
  uint64_t floor_host_ns_ = 0, floor_dev_ns_ = 0;
  int floor_n_ = 0;
  ResidencyStats res_;
  TransferStats xfer_;
};

// RAII marker. Two clock reads and a pointer when profiling is off, so a hot
// path can carry it unconditionally.
class ProfileOp {
 public:
  explicit ProfileOp(const char* name, int stream = 0) { Profiler::get().begin(name, stream); }
  ~ProfileOp() { Profiler::get().end(); }
  ProfileOp(const ProfileOp&) = delete;
  ProfileOp& operator=(const ProfileOp&) = delete;
};

}  // namespace knj

#define KNJ_PROFILE_CONCAT_(a, b) a##b
#define KNJ_PROFILE_CONCAT(a, b) KNJ_PROFILE_CONCAT_(a, b)
// Unique name per site, so two scopes in one scope cannot collide.
#define KNJ_PROFILE_OP(name) \
  ::knj::ProfileOp KNJ_PROFILE_CONCAT(knj_po_, __COUNTER__)(name)
#define KNJ_PROFILE_OP_STREAM(name, stream) \
  ::knj::ProfileOp KNJ_PROFILE_CONCAT(knj_po_, __COUNTER__)(name, stream)
