// tests/unit/test_c21_profiler.cpp -- C21 profiling and telemetry unit tests.
//
// The device clock domain is the part of C21 that cannot be checked by reading
// it. The contract (docs/06-profiling.md section 3) says `dev` is GPU busy time
// from paired device events and `idle` is the gap between them, and the profiler
// therefore has to be right about three things that a host-only run can never
// exercise: that a host marker stops writing `dev` once a device backend is
// attached, that device spans partition their own time into self time the way
// host markers do, and that a gap is only idle when it sits BETWEEN outermost
// spans rather than inside one.
//
// A GPU is not needed for any of that, because the profiler never talks to HIP:
// the device layer feeds it **offsets from one epoch event per stream**
// (profiler.h, enable_device_backend), and offsets are just numbers. So the
// arithmetic is graded here, on synthetic spans with exact expected values, and
// a wrong answer is a failed assertion rather than a plausible-looking table.
//
// Rows are read back through the **artifact writers** (report_csv/report_json/
// report_table) rather than through an accessor, because those files are what a
// user actually reads and diffs. A test that reads private state would pass
// while the published numbers were wrong.
//
// Exit codes:
//   0  every gate passed
//   1  at least one gate failed
//
// Run:
//   /c/Strawberry/c/bin/g++ -std=c++17 -O2 -Wall -Wextra -I. \
//     src/profiler/profiler.cpp src/platform/memprobe.cpp \
//     tests/unit/test_c21_profiler.cpp -o build/host/test_c21_profiler.exe

#include "src/profiler/profiler.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

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

using knj::ClockDomain;
using knj::ProfileConfig;
using knj::Profiler;

// ---------------------------------------------------------------- reading ---
// One row of the CSV, as written. Field order is fixed by report_csv.
struct CsvRow {
  bool found = false;
  std::string component, clock, detail;
  uint64_t ops = 0, dev_ns = 0, idle_ns = 0, host_ns = 0, floor_dev_ns = 0;
};

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == sep) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  out.push_back(cur);
  return out;
}

std::string slurp(const char* path) {
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return std::string();
  std::string s;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
  std::fclose(f);
  return s;
}

void write_csv(const char* path) {
  std::FILE* f = std::fopen(path, "wb");
  if (!f) {
    std::printf("      FAIL  cannot open %s for writing\n", path);
    ++g_failed;
    return;
  }
  Profiler::get().report_csv(f);
  std::fclose(f);
}

CsvRow csv_row(const char* path, const char* component) {
  const std::string text = slurp(path);
  std::vector<std::string> lines = split(text, '\n');
  CsvRow r;
  for (const std::string& line : lines) {
    if (line.empty()) continue;
    const std::vector<std::string> f = split(line, ',');
    if (f.size() < 8) continue;
    if (f[0] != component) continue;
    r.found = true;
    r.component = f[0];
    r.ops = std::strtoull(f[1].c_str(), nullptr, 10);
    r.dev_ns = std::strtoull(f[2].c_str(), nullptr, 10);
    r.idle_ns = std::strtoull(f[3].c_str(), nullptr, 10);
    r.host_ns = std::strtoull(f[4].c_str(), nullptr, 10);
    r.clock = f[5];
    r.detail = f[6];
    r.floor_dev_ns = std::strtoull(f[7].c_str(), nullptr, 10);
    break;
  }
  return r;
}

// `dev_ns_self` out of the JSON, which is where the floor-subtracted device
// number is published. Returns -1 when the component is absent.
double json_dev_self(const char* path, const char* component) {
  const std::string text = slurp(path);
  const std::string key = std::string("\"component\": \"") + component + "\"";
  const size_t at = text.find(key);
  if (at == std::string::npos) return -1.0;
  const size_t v = text.find("\"dev_ns_self\": ", at);
  if (v == std::string::npos) return -1.0;
  return std::strtod(text.c_str() + v + std::strlen("\"dev_ns_self\": "), nullptr);
}

std::string write_table() {
  std::FILE* f = std::fopen("test_c21_out/test_c21_table.txt", "wb");
  if (!f) return std::string();
  Profiler::get().report_table(f);
  std::fclose(f);
  return slurp("test_c21_out/test_c21_table.txt");
}

// ---------------------------------------------------------------- harness ---
// A profiler configured for exact arithmetic: no floor subtraction, no warmup,
// class detail. `on` is a parameter so a gate can also check the off path.
void arm(bool subtract_floor, int floor_ops) {
  ProfileConfig pc;
  pc.on = true;
  pc.format = "csv";
  pc.detail = "class";
  pc.floor_ops = floor_ops;
  pc.warmup_steps = 0;
  pc.subtract_floor = subtract_floor;
  Profiler::get().configure(pc);
}

}  // namespace

int main() {
  Profiler& p = Profiler::get();

  // The artifacts the gates read are published into one directory, created here:
  // a manual run from the repo root otherwise drops five loose `test_c21_*` files
  // beside the sources, which is how they end up committed by accident.
  std::error_code ec;
  std::filesystem::create_directories("test_c21_out", ec);
  if (ec && !std::filesystem::is_directory("test_c21_out")) {
    std::printf("cannot create test_c21_out (%s)\n", ec.message().c_str());
    return 1;
  }

  // =========================================================================
  // GATE 1 -- the host path: dev and host are the same interval, the domain is
  // declared host, and a gap between outer ops becomes idle on the op that
  // follows it.
  // =========================================================================
  std::printf("  GATE 1  host path: dev == host, declared as the host clock\n");
  const int g1 = g_failed;
  arm(false, 0);
  p.begin("alpha", 0);
  p.end();
  // A real gap, long enough that no clock resolution question can explain it.
  // Guaranteed-minimum sleep, not plain sleep_for: on Windows the OS wait has
  // been observed to return after ~1.45 ms of a 3 ms request, which made this
  // gate flake ~1/6 runs while the profiler measured exactly right (proven by
  // instrumenting both clocks: qpc_gap == idle_ns every run). The 2 ms
  // assertion below is untouched -- the stimulus is fixed, not the check.
  // The spin only ever fills an undershoot and is bounded by a monotonic clock.
  const auto g1_t0 = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  while (std::chrono::steady_clock::now() - g1_t0 < std::chrono::milliseconds(3))
    std::this_thread::yield();
  p.begin("beta", 0);
  p.end();
  p.step_done();

  write_csv("test_c21_out/test_c21_host.csv");
  const CsvRow alpha = csv_row("test_c21_out/test_c21_host.csv", "alpha");
  const CsvRow beta = csv_row("test_c21_out/test_c21_host.csv", "beta");
  CHECK(alpha.found && beta.found);
  CHECK(alpha.ops == 1 && beta.ops == 1);
  CHECK(alpha.dev_ns == alpha.host_ns);   // one clock, one interval, by contract
  CHECK(beta.dev_ns == beta.host_ns);
  CHECK(alpha.dev_ns > 0);
  CHECK(alpha.clock == "host");
  CHECK(alpha.idle_ns == 0);              // nothing before the first op
  CHECK(beta.idle_ns >= 2000000ull);      // the 3 ms sleep really did show up
  std::printf("          alpha dev=host=%lluns  beta idle=%lluns (clock %s)\n",
              (unsigned long long)alpha.dev_ns, (unsigned long long)beta.idle_ns,
              alpha.clock.c_str());
  const bool g1_ok = g_failed == g1;

  // =========================================================================
  // GATE 2 -- attaching a device backend moves `dev` to the device events. A
  // host marker must then stop claiming device time: it cannot know it.
  // =========================================================================
  std::printf("  GATE 2  device backend: a host marker no longer writes `dev`\n");
  const int g2 = g_failed;
  arm(false, 0);
  p.enable_device_backend();
  CHECK(p.device_backend());
  CHECK(p.domain() == ClockDomain::Device);
  p.begin("gamma", 0);
  p.end();
  p.step_done();
  write_csv("test_c21_out/test_c21_dev.csv");
  const CsvRow gamma = csv_row("test_c21_out/test_c21_dev.csv", "gamma");
  CHECK(gamma.found);
  CHECK(gamma.host_ns > 0);      // the host interval is still measured
  CHECK(gamma.dev_ns == 0);      // and it is NOT counted as device time
  CHECK(gamma.idle_ns == 0);     // nor is the host gap a device gap
  CHECK(gamma.clock == "device");
  std::printf("          gamma host=%lluns dev=%lluns (clock %s)\n",
              (unsigned long long)gamma.host_ns, (unsigned long long)gamma.dev_ns,
              gamma.clock.c_str());
  const bool g2_ok = g_failed == g2;

  // =========================================================================
  // GATE 3 -- device spans partition into self time exactly as host markers do,
  // and a gap is idle only when it sits BETWEEN outermost spans on a stream.
  //
  // Stream 0:  gate [1000,5000] with gate-inner [2000,3000] nested inside it,
  //            then up [9000,11000].
  //   gate       self = 4000 - 1000 = 3000
  //   gate-inner self = 1000
  //   gate + gate-inner = 4000 = the whole span, so nothing is counted twice
  //   up         self = 2000, idle = 9000 - 5000 = 4000
  // Stream 1:  other [2000,3000], other2 [4000,6000]  -> idle 1000 on other2.
  //   Two streams have their own gaps: a single global gap would be a lie about
  //   an overlapped engine, so `other` must NOT inherit stream 0's idle.
  // =========================================================================
  std::printf("  GATE 3  device spans: nested self time and per-stream idle\n");
  const int g3 = g_failed;
  p.device_span("gate", 1000, 5000, 0);
  p.device_span("gate-inner", 2000, 3000, 0);
  p.device_span("up", 9000, 11000, 0);
  p.device_span("other", 2000, 3000, 1);
  // The LAST span pushed; it must survive the fold without a finalise call.
  p.device_span("other2", 4000, 6000, 1);
  write_csv("test_c21_out/test_c21_dev.csv");
  const CsvRow gate = csv_row("test_c21_out/test_c21_dev.csv", "gate");
  const CsvRow inner = csv_row("test_c21_out/test_c21_dev.csv", "gate-inner");
  const CsvRow up = csv_row("test_c21_out/test_c21_dev.csv", "up");
  const CsvRow other = csv_row("test_c21_out/test_c21_dev.csv", "other");
  const CsvRow other2 = csv_row("test_c21_out/test_c21_dev.csv", "other2");
  CHECK(gate.found && inner.found && up.found && other.found && other2.found);
  CHECK(gate.dev_ns == 3000);
  CHECK(inner.dev_ns == 1000);
  CHECK(gate.dev_ns + inner.dev_ns == 4000);   // the parent span, exactly once
  CHECK(up.dev_ns == 2000);
  CHECK(up.idle_ns == 4000);                   // 9000 - 5000, on stream 0
  CHECK(other.dev_ns == 1000);                 // 3000 - 2000
  CHECK(other.idle_ns == 0);                   // first span of stream 1: no gap
  CHECK(other2.dev_ns == 2000);                // GATE 4: the last span is kept
  CHECK(other2.idle_ns == 1000);               // 4000 - 3000, on stream 1
  std::printf("          gate self=%llu + inner=%llu == 4000; up idle=%llu; "
              "other2 dev=%llu idle=%llu\n",
              (unsigned long long)gate.dev_ns, (unsigned long long)inner.dev_ns,
              (unsigned long long)up.idle_ns, (unsigned long long)other2.dev_ns,
              (unsigned long long)other2.idle_ns);
  const bool g3_ok = g_failed == g3;

  // =========================================================================
  // GATE 5 -- the device floor. Until one is measured through the event path,
  // dev rows are published RAW: subtracting a HOST floor from DEVICE time would
  // be an arithmetic error wearing a correction's clothes. Once measured, it is
  // subtracted per op, exactly as the host floor is.
  // =========================================================================
  std::printf("  GATE 5  device floor: raw until measured, subtracted after\n");
  const int g5 = g_failed;
  // The floor only exists when subtraction is armed, so this gate re-arms. It
  // re-arms with floor_ops = 0, which leaves the HOST floor unmeasured: the
  // point is that a device floor is a separate measurement, not that it is
  // derived from the host one.
  arm(true, 0);
  // One host marker so the row has an `ops` count for the floor to multiply.
  p.begin("floored", 0);
  p.end();
  p.device_span("floored", 0, 10000, 0);
  write_csv("test_c21_out/test_c21_dev.csv");
  std::FILE* jf = std::fopen("test_c21_out/test_c21_dev.json", "wb");
  CHECK(jf != nullptr);
  if (jf) {
    p.report_json(jf);
    std::fclose(jf);
  }
  const CsvRow floored = csv_row("test_c21_out/test_c21_dev.csv", "floored");
  CHECK(floored.found);
  CHECK(floored.dev_ns == 10000);              // the raw measurement is published
  CHECK(p.device_floor_measured() == false);
  CHECK(json_dev_self("test_c21_out/test_c21_dev.json", "floored") == 10000.0);  // no host floor taken
  p.set_device_floor(2500, 64);
  CHECK(p.device_floor_measured());
  // Re-publish before reading: the reader reads the FILE, so asserting against
  // the pre-floor file would be asserting about the state before the change.
  jf = std::fopen("test_c21_out/test_c21_dev.json", "wb");
  CHECK(jf != nullptr);
  if (jf) {
    p.report_json(jf);
    std::fclose(jf);
  }
  const double subtracted = json_dev_self("test_c21_out/test_c21_dev.json", "floored");
  CHECK(subtracted == 7500.0);                 // 10000 - 2500*1 op
  // The device floor must not have leaked into the host column.
  const CsvRow floored_host = csv_row("test_c21_out/test_c21_dev.csv", "floored");
  CHECK(floored_host.host_ns > 0);
  std::printf("          raw dev=%lluns -> published %.0fns after the device floor "
              "(floor 2500ns x 1 op)\n",
              (unsigned long long)floored.dev_ns, subtracted);
  const bool g5_ok = g_failed == g5;

  // =========================================================================
  // GATE 6 -- a full event ring is a STATED loss. A dropped span must appear in
  // the report, because a profiler that quietly loses measurements reports a
  // smaller total than the run performed, which is indistinguishable from a
  // faster run.
  // =========================================================================
  std::printf("  GATE 6  dropped spans are reported, not swallowed\n");
  const int g6 = g_failed;
  p.set_device_dropped(7);
  const std::string table = write_table();
  CHECK(table.find("7 dropped") != std::string::npos);
  CHECK(table.find("a stated loss, not a silent one") != std::string::npos);
  CHECK(p.device_spans() == 1);               // gate 5 re-armed the profiler
  CHECK(p.device_dropped() == 7);
  std::printf("          %llu span(s) recorded, %llu dropped, both named in the table\n",
              (unsigned long long)p.device_spans(),
              (unsigned long long)p.device_dropped());
  const bool g6_ok = g_failed == g6;

  // =========================================================================
  // GATE 7 -- the off path stays free: no rows, no device accounting, and the
  // report says it is off instead of printing an empty table.
  // =========================================================================
  std::printf("  GATE 7  profiler off: nothing is recorded\n");
  const int g7 = g_failed;
  ProfileConfig off;
  off.on = false;
  p.configure(off);
  p.begin("never", 0);
  p.end();
  p.device_span("never-dev", 0, 5000, 0);
  write_csv("test_c21_out/test_c21_off.csv");
  CHECK(csv_row("test_c21_out/test_c21_off.csv", "never").found == false);
  CHECK(csv_row("test_c21_out/test_c21_off.csv", "never-dev").found == false);
  CHECK(p.on() == false);
  const std::string off_table = write_table();
  CHECK(off_table.find("profile: off") != std::string::npos);
  const bool g7_ok = g_failed == g7;

  std::printf("\n");
  std::printf("C21 profiling: %d check(s), %d failed\n", g_checks, g_failed);
  if (g1_ok && g2_ok && g3_ok && g5_ok && g6_ok && g7_ok && g_failed == 0) {
    std::printf("RESULT: PASS\n");
    return 0;
  }
  std::printf("RESULT: FAIL\n");
  return 1;
}
