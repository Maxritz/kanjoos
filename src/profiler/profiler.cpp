// src/profiler/profiler.cpp -- see profiler.h.
//
// Deliberately small: a marker stack, a per-stream gap walk, and three writers
// over one row set. Anything cleverer would drift from the definitions in
// `docs/06-profiling.md` section 3, and a profiler that disagrees with its own
// specification is worse than no profiler.
#include "src/profiler/profiler.h"

#include "src/platform/memprobe.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>

namespace knj {
namespace {

uint64_t now_ns() {
  using namespace std::chrono;
  return (uint64_t)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\t': o += "\\t"; break;
      default: o += c;
    }
  }
  return o;
}

}  // namespace

const char* clock_domain_name(ClockDomain d) {
  switch (d) {
    case ClockDomain::Host: return "host";
    case ClockDomain::Device: return "device";
    default: return "none";
  }
}

Profiler& Profiler::get() {
  static Profiler p;
  return p;
}

void Profiler::configure(const ProfileConfig& c) {
  cfg_ = c;
  if (cfg_.on && domain_ == ClockDomain::None) domain_ = ClockDomain::Host;
  reset();
}

void Profiler::reset() {
  rows_.clear();
  stack_.clear();
  idle_pending_ = 0;
  for (int i = 0; i < 8; ++i) stream_end_ns_[i] = 0;
  for (int i = 0; i < 8; ++i) dev_pending_[i].clear();
  dev_spans_recorded_ = dev_spans_dropped_ = 0;
  dev_need_fold_ = false;
  steps_seen_ = steps_kept_ = steps_discarded_ = 0;
  nested_ops_ = max_depth_ = total_ops_ = 0;
}

Profiler::Row* Profiler::find_or_add(const std::string& group) {
  for (Row& r : rows_) {
    if (r.group == group) return &r;
  }
  rows_.push_back(Row());
  rows_.back().group = group;
  return &rows_.back();
}

// `name` is "class" or "class:scope". Class grouping strips the suffix; layer
// and kernel detail keep the whole string, which is what lets one instrumentation
// pass serve all three --profile-detail settings.
std::string Profiler::group_for(const std::string& name) const {
  if (cfg_.detail == "class") {
    const size_t p = name.find(':');
    if (p != std::string::npos) return name.substr(0, p);
  }
  return name;
}

void Profiler::begin(const char* name, int stream) {
  if (!cfg_.on) return;
  const uint64_t t = now_ns();
  if (stack_.empty()) {
    // Idle is the gap between the previous op's end on THIS stream and now. Per
    // stream, then summed: two concurrent streams have their own gaps, and a
    // global gap would be a lie about an overlapped engine. Only the outermost
    // op can start after a gap -- a child starts inside its parent by definition.
    if (stream < 0 || stream >= 8) stream = 0;
    cur_stream_ = stream;
    const uint64_t prev_end = stream_end_ns_[stream];
    idle_pending_ = (prev_end != 0 && t > prev_end) ? (t - prev_end) : 0;
  } else {
    ++nested_ops_;
  }
  Frame f;
  f.group = group_for(name ? name : "");
  f.t0 = t;
  stack_.push_back(f);
  if (stack_.size() > max_depth_) max_depth_ = stack_.size();
}

void Profiler::end() {
  if (!cfg_.on || stack_.empty()) return;
  const uint64_t t1 = now_ns();
  Frame f = stack_.back();
  stack_.pop_back();
  const uint64_t span = t1 - f.t0;
  // Self time: a parent does not get to claim what its children spent, so the
  // rows partition the run instead of double-counting it.
  const uint64_t self = span > f.child_ns ? span - f.child_ns : 0;
  const bool outer = stack_.empty();
  if (!outer) stack_.back().child_ns += span;
  else stream_end_ns_[cur_stream_] = t1;
  if (steps_seen_ < (uint64_t)cfg_.warmup_steps) return;   // warmup: discarded
  Row* r = find_or_add(f.group);
  r->ops += 1;
  // With a device backend attached, `dev` belongs to the device events and a
  // host marker has no business writing it -- that is the difference between a
  // declared clock domain and a claimed one.
  if (!device_backend_) {
    r->dev_ns += self;
    if (outer) r->idle_ns += idle_pending_;
  }
  r->host_ns += self;
  ++total_ops_;
}

void Profiler::device_span(const char* name, uint64_t start_ns, uint64_t end_ns, int stream) {
  if (!cfg_.on || !device_backend_) return;
  if (stream < 0 || stream >= 8) stream = 0;
  if (end_ns < start_ns) end_ns = start_ns;
  DeviceSpan sp;
  sp.group = group_for(name ? name : "");
  sp.start = start_ns;
  sp.end = end_ns;
  dev_pending_[stream].push_back(sp);
  find_or_add(sp.group)->dev_spans += 1;
  ++dev_spans_recorded_;
  dev_need_fold_ = true;
}

// The interval walk. Spans arrive per stream and are folded in ascending start
// order, so containment IS the nesting: a span that starts after its parent and
// ends before it is a child, and its microseconds are subtracted from the
// parent's self time exactly as the host marker stack does.
//
// Idle is the gap between OUTERMOST spans on one stream. A gap while a parent is
// still open is not idle -- the compiler is inside a scope the report already
// attributes -- and counting it would invent device idle time that the run never
// had.
//
// `mutable` state and a const method on purpose: the report writers fold on
// demand, so a caller that forgets a finalise step cannot silently lose the last
// span of every stream.
void Profiler::fold_device_spans() const {
  if (!dev_need_fold_) return;
  dev_need_fold_ = false;
  Profiler* self = const_cast<Profiler*>(this);
  for (int st = 0; st < 8; ++st) {
    std::vector<DeviceSpan>& v = dev_pending_[st];
    if (v.empty()) continue;
    std::stable_sort(v.begin(), v.end(),
                     [](const DeviceSpan& a, const DeviceSpan& b) { return a.start < b.start; });
    struct Open {
      std::string group;
      uint64_t start = 0, end = 0, child = 0;
    };
    std::vector<Open> open;
    uint64_t prev_end = 0;
    auto close = [&](const Open& f) {
      const uint64_t dur = f.end - f.start;
      Row* r = self->find_or_add(f.group);
      r->dev_ns += dur > f.child ? dur - f.child : 0;
    };
    for (const DeviceSpan& s : v) {
      while (!open.empty() && s.start >= open.back().end) {
        close(open.back());
        open.pop_back();
      }
      if (open.empty() && prev_end != 0 && s.start > prev_end) {
        Row* r = self->find_or_add(s.group);
        r->idle_ns += s.start - prev_end;
      }
      const uint64_t dur = s.end - s.start;
      if (!open.empty()) open.back().child += dur;
      Open o;
      o.group = s.group;
      o.start = s.start;
      o.end = s.end;
      open.push_back(o);
      if (s.end > prev_end) prev_end = s.end;
    }
    while (!open.empty()) {
      close(open.back());
      open.pop_back();
    }
    v.clear();
  }
}

void Profiler::step_done() {
  if (!cfg_.on) return;
  if (steps_seen_ < (uint64_t)cfg_.warmup_steps) ++steps_discarded_; else ++steps_kept_;
  ++steps_seen_;
}

void Profiler::measure_floor() {
  const int n = cfg_.floor_ops > 0 ? cfg_.floor_ops : 0;
  floor_n_ = n;
  floor_host_ns_ = floor_dev_ns_ = 0;
  if (n == 0) return;
  // The floor must go through the identical begin/end path, or it measures
  // something else. Warm up first: the first pairs pay for cache misses on the
  // row vector and would inflate the floor.
  const bool was_on = cfg_.on;
  const std::string keep_detail = cfg_.detail;
  const int keep_warm = cfg_.warmup_steps;
  const uint64_t keep_steps = steps_seen_;
  const uint64_t keep_stream_end = stream_end_ns_[7];

  cfg_.on = true;
  cfg_.detail = "kernel";        // never grouped with a real component
  cfg_.warmup_steps = 0;
  steps_seen_ = 0;
  for (int i = 0; i < 32; ++i) { begin("__floor", 7); end(); }
  rows_.erase(std::remove_if(rows_.begin(), rows_.end(),
                             [](const Row& r) { return r.group == "__floor"; }),
              rows_.end());
  total_ops_ = 0;
  nested_ops_ = 0;

  const uint64_t t0 = now_ns();
  for (int i = 0; i < n; ++i) { begin("__floor", 7); end(); }
  const uint64_t t1 = now_ns();
  rows_.erase(std::remove_if(rows_.begin(), rows_.end(),
                             [](const Row& r) { return r.group == "__floor"; }),
              rows_.end());

  cfg_.on = was_on;
  cfg_.detail = keep_detail;
  cfg_.warmup_steps = keep_warm;
  steps_seen_ = keep_steps;
  total_ops_ = 0;
  nested_ops_ = 0;
  stream_end_ns_[7] = keep_stream_end;
  floor_host_ns_ = (t1 - t0) / (uint64_t)n;
  floor_dev_ns_ = floor_host_ns_;   // one marker pair, one clock, host domain
}

std::vector<Profiler::Row> Profiler::rows_sorted() const {
  std::vector<Row> v = rows_;
  std::sort(v.begin(), v.end(), [](const Row& a, const Row& b) {
    if (a.dev_ns != b.dev_ns) return a.dev_ns > b.dev_ns;
    return a.group < b.group;
  });
  return v;
}

// A row that did nothing still costs the marker pair. Subtracting the floor is
// what makes a sub-microsecond row readable; --profiling=no-subtract keeps the
// raw numbers instead, and the report says so.
//
// There are two floors because there are two clocks. On the host path they are
// one number (one marker pair, one clock). With a device backend the device
// floor has to be measured through the event path, and until it is, the dev
// rows are printed RAW -- subtracting a host floor from device time would be an
// arithmetic error dressed as a correction.
double Profiler::subtract_host(const Row& r) const {
  if (!cfg_.subtract_floor || floor_host_ns_ == 0 || r.ops == 0) return 0.0;
  const double total = (double)floor_host_ns_ * (double)r.ops;
  // Never subtract more than the row contains. Without this a row whose own time
  // is below its floor renders as NEGATIVE microseconds, which is not a small
  // number -- it is a nonsense one, and it reads as a bug in the engine rather
  // than as the resolution limit of the instrument.
  return total > (double)r.host_ns ? (double)r.host_ns : total;
}

double Profiler::subtract_dev(const Row& r) const {
  if (!cfg_.subtract_floor) return 0.0;
  if (device_backend_) {
    if (!device_floor_measured_) return 0.0;
    // The multiplier is the number of MEASUREMENTS in the row. For a row built
    // from host markers that is `ops`; a device-only row has ops 0 and its
    // measurement count is its span count. Using ops alone would skip the floor
    // for exactly the rows that came from the device clock.
    const uint64_t n = r.ops > 0 ? r.ops : r.dev_spans;
    if (n == 0) return 0.0;
    const double total = (double)dev_floor_ns_ * (double)n;
    return total > (double)r.dev_ns ? (double)r.dev_ns : total;
  }
  if (floor_dev_ns_ == 0 || r.ops == 0) return 0.0;
  const double total = (double)floor_dev_ns_ * (double)r.ops;
  return total > (double)r.dev_ns ? (double)r.dev_ns : total;
}

std::string Profiler::identity_line() const {
  const HostMem hm = host_mem();
  char buf[640];
  std::snprintf(buf, sizeof(buf),
                "clock %s   dev/host are SELF time (child time subtracted; max depth %llu)"
                "   steps kept %llu, discarded %llu (warmup %d)   ops %llu   rss %.0f MiB",
                clock_domain_name(domain_), (unsigned long long)max_depth_,
                (unsigned long long)steps_kept_, (unsigned long long)steps_discarded_,
                cfg_.warmup_steps, (unsigned long long)total_ops_,
                (double)hm.peak_rss_bytes / 1048576.0);
  return std::string(buf);
}

void Profiler::print_header(std::FILE* f, const std::string& version,
                            const std::string& device_line,
                            const std::string& budgets_line) const {
  std::fprintf(f, "\n  kanjoos %s   device %s\n", version.c_str(), device_line.c_str());
  if (!budgets_line.empty()) std::fprintf(f, "                 %s\n", budgets_line.c_str());
  std::fprintf(f, "                 %s\n", identity_line().c_str());
  if (nested_ops_)
    std::fprintf(f, "                 %llu nested scope(s) measured as self time; the rows "
                    "partition the run, they do not overlap it\n",
                 (unsigned long long)nested_ops_);
  std::fprintf(f, "                 --profiling never changes what the engine computes; this "
                  "block reports what it measured and nothing else\n");
}

void Profiler::report_table(std::FILE* f) const {
  if (!cfg_.on) {
    std::fprintf(f, "\n-- profile: off (no --profiling flag) --\n");
    return;
  }
  fold_device_spans();
  std::fprintf(f, "\n     %-20s %5s %7s %12s %12s %12s\n", "component", "ops", "%dev",
               "dev us", "idle us", "host us");
  const std::vector<Row> v = rows_sorted();
  uint64_t total_dev = 0, total_idle = 0, total_host = 0;
  for (const Row& r : v) {
    const double sd = subtract_dev(r), sh = subtract_host(r);
    total_dev += r.dev_ns > (uint64_t)sd ? r.dev_ns - (uint64_t)sd : 0;
    total_idle += r.idle_ns;
    total_host += r.host_ns > (uint64_t)sh ? r.host_ns - (uint64_t)sh : 0;
  }
  for (const Row& r : v) {
    const double sd = subtract_dev(r), sh = subtract_host(r);
    const double dev = ((double)r.dev_ns - sd) / 1e3;
    const double host = ((double)r.host_ns - sh) / 1e3;
    const double pct = total_dev > 0 ? 100.0 * ((double)r.dev_ns - sd) / (double)total_dev : 0.0;
    std::fprintf(f, "     %-20s %5llu %6.1f%% %12.2f %12.2f %12.2f\n", r.group.c_str(),
                 (unsigned long long)r.ops, pct, dev, (double)r.idle_ns / 1e3, host);
  }
  if (floor_n_ > 0) {
    std::fprintf(f,
                 "   instrumentation floor, %d empty ops timed through the same begin/end "
                 "path: %.3f us host, %.3f us device each.\n",
                 floor_n_, (double)floor_host_ns_ / 1e3, (double)floor_dev_ns_ / 1e3);
    if (!cfg_.subtract_floor)
      std::fprintf(f, "   (--profiling=no-subtract: the rows above are raw, the floor is NOT "
                      "removed)\n");
  } else {
    std::fprintf(f, "   instrumentation floor: not measured (--profile-floor 0)\n");
  }
  std::fprintf(f, "   totals: dev %.2f us   idle %.2f us   host %.2f us   over %llu op(s)\n",
               (double)total_dev / 1e3, (double)total_idle / 1e3, (double)total_host / 1e3,
               (unsigned long long)total_ops_);
  std::fprintf(f, "   clock domain: %s%s\n", clock_domain_name(domain_),
               domain_ == ClockDomain::Host
                   ? "  (no device event backend on this path: dev and host are the same "
                     "interval by construction)"
                   : "");
  if (device_backend_) {
    std::fprintf(f,
                 "   device events: %llu span(s) folded into `dev`/%llu dropped "
                 "(ring full)%s\n",
                 (unsigned long long)dev_spans_recorded_,
                 (unsigned long long)dev_spans_dropped_,
                 dev_spans_dropped_ ? "  <-- a stated loss, not a silent one" : "");
    if (device_floor_measured_)
      std::fprintf(f, "   device floor: %d empty event pair(s), %.3f us each -- dev rows "
                      "above are floor-subtracted\n",
                   dev_floor_n_, (double)dev_floor_ns_ / 1e3);
    else
      std::fprintf(f, "   device floor: NOT MEASURED -- the dev rows above are RAW; a host "
                      "floor is deliberately not subtracted from device time\n");
    std::fprintf(f, "   device resolution: elapsed-time is reported in float ms, so a span "
                    "below roughly 0.1 us is inside the timer's noise\n");
  }
  if (res_.measured) {
    std::fprintf(f,
                 "\n  residency   vram %.1f%%   ram %.1f%%   nvme %.1f%%   slots %llu/%llu   "
                 "evictions %llu   prefetch hit %.2f   stall %.2f ms\n",
                 res_.vram_pct, res_.ram_pct, res_.nvme_pct,
                 (unsigned long long)res_.slots_used, (unsigned long long)res_.slots_total,
                 (unsigned long long)res_.evictions, res_.prefetch_hit, res_.stall_ms);
  } else {
    std::fprintf(f, "\n  residency   NOT MEASURED on this path\n");
  }
  if (xfer_.measured) {
    // Peak pinned is printed in MiB, not GiB: it is a STAGING high-water mark and
    // one expert-side bounce buffer is MiB-scale. In GiB a real 5 MiB peak renders
    // as `0.00` -- a true measurement displayed as a false zero.
    std::fprintf(f,
                 "  transfer    h2d %.2f GB   d2h %.2f GB   nvme read %.2f GB   nvme write "
                 "%.2f GB   peak pinned %.2f MiB\n",
                 (double)xfer_.h2d_bytes / 1e9, (double)xfer_.d2h_bytes / 1e9,
                 (double)xfer_.nvme_read_bytes / 1e9, (double)xfer_.nvme_write_bytes / 1e9,
                 (double)xfer_.peak_pinned_bytes / 1048576.0);
  } else {
    std::fprintf(f, "  transfer    NOT MEASURED on this path\n");
  }
}

void Profiler::report_json(std::FILE* f) const {
  fold_device_spans();
  const std::vector<Row> v = rows_sorted();
  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"clock_domain\": \"%s\",\n", clock_domain_name(domain_));
  std::fprintf(f, "  \"detail\": \"%s\",\n", json_escape(cfg_.detail).c_str());
  std::fprintf(f, "  \"subtract_floor\": %s,\n", cfg_.subtract_floor ? "true" : "false");
  std::fprintf(f, "  \"warmup_steps\": %d,\n", cfg_.warmup_steps);
  std::fprintf(f, "  \"steps_kept\": %llu,\n", (unsigned long long)steps_kept_);
  std::fprintf(f, "  \"steps_discarded\": %llu,\n", (unsigned long long)steps_discarded_);
  std::fprintf(f, "  \"max_depth\": %llu,\n", (unsigned long long)max_depth_);
  std::fprintf(f, "  \"nested_scopes_self_time\": %llu,\n", (unsigned long long)nested_ops_);
  std::fprintf(f, "  \"ops\": %llu,\n", (unsigned long long)total_ops_);
  std::fprintf(f, "  \"floor\": {\"n\": %d, \"host_ns\": %llu, \"dev_ns\": %llu},\n", floor_n_,
               (unsigned long long)floor_host_ns_, (unsigned long long)floor_dev_ns_);
  std::fprintf(f, "  \"device_backend\": %s,\n", device_backend_ ? "true" : "false");
  std::fprintf(f, "  \"device_spans\": %llu,\n", (unsigned long long)dev_spans_recorded_);
  std::fprintf(f, "  \"device_spans_dropped\": %llu,\n", (unsigned long long)dev_spans_dropped_);
  std::fprintf(f, "  \"device_floor_measured\": %s,\n", device_floor_measured_ ? "true" : "false");
  std::fprintf(f, "  \"components\": [\n");
  for (size_t i = 0; i < v.size(); ++i) {
    const Row& r = v[i];
    const double sd = subtract_dev(r), sh = subtract_host(r);
    std::fprintf(f,
                 "    {\"component\": \"%s\", \"ops\": %llu, \"dev_ns\": %llu, "
                 "\"dev_ns_self\": %.1f, \"idle_ns\": %llu, \"host_ns\": %llu, "
                 "\"host_ns_self\": %.1f}%s\n",
                 json_escape(r.group).c_str(), (unsigned long long)r.ops,
                 (unsigned long long)r.dev_ns,
                 (double)r.dev_ns - sd > 0.0 ? (double)r.dev_ns - sd : 0.0,
                 (unsigned long long)r.idle_ns, (unsigned long long)r.host_ns,
                 (double)r.host_ns - sh > 0.0 ? (double)r.host_ns - sh : 0.0,
                 i + 1 < v.size() ? "," : "");
  }
  std::fprintf(f, "  ],\n");
  std::fprintf(f, "  \"residency_measured\": %s,\n", res_.measured ? "true" : "false");
  if (res_.measured) {
    std::fprintf(f, "  \"residency\": {\"vram_pct\": %.4f, \"ram_pct\": %.4f, "
                    "\"nvme_pct\": %.4f, \"slots_used\": %llu, \"slots_total\": %llu, "
                    "\"evictions\": %llu, \"prefetch_hit\": %.4f, \"stall_ms\": %.4f},\n",
                 res_.vram_pct, res_.ram_pct, res_.nvme_pct,
                 (unsigned long long)res_.slots_used, (unsigned long long)res_.slots_total,
                 (unsigned long long)res_.evictions, res_.prefetch_hit, res_.stall_ms);
  }
  std::fprintf(f, "  \"transfer_measured\": %s", xfer_.measured ? "true" : "false");
  if (xfer_.measured) {
    std::fprintf(f, ",\n  \"transfer\": {\"h2d_bytes\": %llu, \"d2h_bytes\": %llu, "
                    "\"nvme_read_bytes\": %llu, \"nvme_write_bytes\": %llu, "
                    "\"peak_pinned_bytes\": %llu}\n",
                 (unsigned long long)xfer_.h2d_bytes, (unsigned long long)xfer_.d2h_bytes,
                 (unsigned long long)xfer_.nvme_read_bytes,
                 (unsigned long long)xfer_.nvme_write_bytes,
                 (unsigned long long)xfer_.peak_pinned_bytes);
  } else {
    std::fprintf(f, "\n");
  }
  std::fprintf(f, "}\n");
}

void Profiler::report_csv(std::FILE* f) const {
  fold_device_spans();
  std::fprintf(f, "component,ops,dev_ns,idle_ns,host_ns,clock_domain,detail,floor_dev_ns\n");
  const std::vector<Row> v = rows_sorted();
  for (const Row& r : v)
    std::fprintf(f, "%s,%llu,%llu,%llu,%llu,%s,%s,%llu\n", r.group.c_str(),
                 (unsigned long long)r.ops, (unsigned long long)r.dev_ns,
                 (unsigned long long)r.idle_ns, (unsigned long long)r.host_ns,
                 clock_domain_name(domain_), cfg_.detail.c_str(),
                 (unsigned long long)floor_dev_ns_);
}

int Profiler::write_dir() const {
  if (cfg_.dir.empty()) return 0;
  const std::string dir = cfg_.dir;
  // std::filesystem, not a shell-out: `system("mkdir -p ...")` on Windows runs
  // under cmd.exe, whose `mkdir` takes -p as a DIRECTORY NAME, so the previous
  // version created a junk directory literally called `-p` in the working
  // directory while appearing to succeed. One portable call, no shell.
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec && !std::filesystem::is_directory(dir)) {
    std::fprintf(stderr, "profile-dir: cannot create %s (%s)\n", dir.c_str(), ec.message().c_str());
    return 0;
  }
  int written = 0;
  const char* names[3] = {"profile.txt", "profile.json", "profile.csv"};
  for (int i = 0; i < 3; ++i) {
    const std::string path = dir + "/" + names[i];
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
      std::fprintf(stderr, "profile-dir: cannot write %s\n", path.c_str());
      continue;
    }
    if (i == 0) report_table(f);
    else if (i == 1) report_json(f);
    else report_csv(f);
    std::fclose(f);
    ++written;
  }
  if (written) std::printf("profile files: %s/profile.{txt,json,csv}\n", dir.c_str());
  return written;
}

void Profiler::report(std::FILE* f) const {
  if (!cfg_.on) return;
  if (cfg_.format == "json") report_json(f);
  else if (cfg_.format == "csv") report_csv(f);
  else report_table(f);
}

}  // namespace knj
