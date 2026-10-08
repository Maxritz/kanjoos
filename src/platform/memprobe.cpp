// src/platform/memprobe.cpp -- see memprobe.h.
//
// Two implementations, one on each side of `#ifdef _WIN32`, and nothing above
// this file knows which one ran. A probe that cannot read a field reports 0 for
// it rather than inventing a plausible value: a zero is visibly "not measured",
// a guess is not.
#include "src/platform/memprobe.h"

#include <cstdio>
#include <cstring>

#ifdef _WIN32

#include <windows.h>

namespace knj {
namespace {

// PROCESS_MEMORY_COUNTERS_EX, declared here rather than included from psapi.h,
// and resolved from kernel32's K32GetProcessMemoryInfo at call time. That keeps
// this translation unit free of a psapi.lib link dependency, which matters
// because the bench drivers are built by hand-written hipcc lines and by
// run_bench.sh, and neither of those should have to grow a library for a probe.
struct PmcEx {
  DWORD cb;
  DWORD page_fault_count;
  SIZE_T peak_working_set_size;
  SIZE_T working_set_size;
  SIZE_T quota_peak_paged_pool_usage;
  SIZE_T quota_paged_pool_usage;
  SIZE_T quota_peak_non_paged_pool_usage;
  SIZE_T quota_non_paged_pool_usage;
  SIZE_T pagefile_usage;
  SIZE_T peak_pagefile_usage;
  SIZE_T private_usage;
};

typedef BOOL(WINAPI* GetProcMemFn)(HANDLE, PmcEx*, DWORD);

uint64_t filetime_ns(const FILETIME& ft) {
  ULARGE_INTEGER v;
  v.LowPart = ft.dwLowDateTime;
  v.HighPart = ft.dwHighDateTime;
  return (uint64_t)v.QuadPart * 100ull;   // FILETIME is 100 ns units
}

GetProcMemFn proc_mem_fn() {
  static GetProcMemFn fn = []() -> GetProcMemFn {
    HMODULE k32 = GetModuleHandleA("kernel32.dll");
    if (!k32) return nullptr;
    return (GetProcMemFn)(void*)GetProcAddress(k32, "K32GetProcessMemoryInfo");
  }();
  return fn;
}

}  // namespace

HostMem host_mem() {
  HostMem m;
  if (GetProcMemFn fn = proc_mem_fn()) {
    PmcEx pmc;
    std::memset(&pmc, 0, sizeof(pmc));
    pmc.cb = (DWORD)sizeof(pmc);
    if (fn(GetCurrentProcess(), &pmc, (DWORD)sizeof(pmc))) {
      m.rss_bytes = (uint64_t)pmc.working_set_size;
      m.peak_rss_bytes = (uint64_t)pmc.peak_working_set_size;
      m.private_bytes = (uint64_t)pmc.private_usage;
    }
  }
  MEMORYSTATUSEX ms;
  std::memset(&ms, 0, sizeof(ms));
  ms.dwLength = (DWORD)sizeof(ms);
  if (GlobalMemoryStatusEx(&ms)) {
    m.phys_total_bytes = (uint64_t)ms.ullTotalPhys;
    m.phys_avail_bytes = (uint64_t)ms.ullAvailPhys;
  }
  return m;
}

CpuTime cpu_time() {
  CpuTime c;
  FILETIME create, exit, kernel, user;
  if (GetProcessTimes(GetCurrentProcess(), &create, &exit, &kernel, &user)) {
    c.user_ns = filetime_ns(user);
    c.sys_ns = filetime_ns(kernel);
  }
  return c;
}

SystemCpu system_cpu() {
  SystemCpu s;
  FILETIME idle, kernel, user;
  if (GetSystemTimes(&idle, &kernel, &user)) {
    // GetSystemTimes reports kernel time inclusive of idle on Windows, so busy
    // is (kernel + user) - idle and idle stays idle.
    const uint64_t i = filetime_ns(idle), k = filetime_ns(kernel), u = filetime_ns(user);
    s.idle_ns = i;
    s.busy_ns = (k + u > i) ? (k + u - i) : 0;
  }
  return s;
}

const char* platform_name() { return "win32"; }

int logical_cpus() {
  SYSTEM_INFO si;
  std::memset(&si, 0, sizeof(si));
  GetNativeSystemInfo(&si);
  return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 0;
}

}  // namespace knj

#elif defined(__linux__)  // -------------------------------------------------

#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <unistd.h>

namespace knj {
namespace {

// Reads a `Key:  <value> <unit>` line out of /proc/self/status and returns the
// value in bytes. Returns 0 when the key is absent -- never a default.
uint64_t status_kb(const char* key) {
  FILE* f = std::fopen("/proc/self/status", "r");
  if (!f) return 0;
  char line[256];
  const size_t klen = std::strlen(key);
  uint64_t kb = 0;
  while (std::fgets(line, sizeof(line), f)) {
    if (std::strncmp(line, key, klen) == 0) {
      // "VmRSS:	   12345 kB" -- the key includes its trailing colon.
      unsigned long long v = 0;
      if (std::sscanf(line + klen, " %llu", &v) == 1) kb = (uint64_t)v;
      break;
    }
  }
  std::fclose(f);
  return kb * 1024ull;
}

}  // namespace

HostMem host_mem() {
  HostMem m;
  m.rss_bytes = status_kb("VmRSS:");
  m.peak_rss_bytes = status_kb("VmHWM:");
  m.private_bytes = status_kb("VmData:");
  {
    struct sysinfo si;
    if (sysinfo(&si) == 0)
      m.phys_total_bytes = (uint64_t)si.totalram * (uint64_t)si.mem_unit;
  }
  if (FILE* f = std::fopen("/proc/meminfo", "r")) {
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
      if (std::strncmp(line, "MemAvailable:", 13) == 0) {
        unsigned long long kb = 0;
        if (std::sscanf(line + 13, " %llu", &kb) == 1) m.phys_avail_bytes = kb * 1024ull;
        break;
      }
    }
    std::fclose(f);
  }
  return m;
}

CpuTime cpu_time() {
  CpuTime c;
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
    c.user_ns = (uint64_t)ru.ru_utime.tv_sec * 1000000000ull + (uint64_t)ru.ru_utime.tv_usec * 1000ull;
    c.sys_ns = (uint64_t)ru.ru_stime.tv_sec * 1000000000ull + (uint64_t)ru.ru_stime.tv_usec * 1000ull;
  }
  return c;
}

SystemCpu system_cpu() {
  SystemCpu s;
  FILE* f = std::fopen("/proc/stat", "r");
  if (!f) return s;
  char line[512];
  if (std::fgets(line, sizeof(line), f)) {
    // "cpu  user nice system idle iowait irq softirq steal guest guest_nice"
    unsigned long long user = 0, nice = 0, system = 0, idle = 0, iowait = 0,
                        irq = 0, softirq = 0, steal = 0;
    const int n = std::sscanf(line, "cpu %llu %llu %llu %llu %llu %llu %llu %llu",
                              &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal);
    if (n >= 4) {
      const long hz = sysconf(_SC_CLK_TCK);
      const uint64_t per_tick = hz > 0 ? (uint64_t)(1000000000ull / (uint64_t)hz) : 0;
      s.idle_ns = (idle + iowait) * per_tick;
      s.busy_ns = (user + nice + system + irq + softirq + steal) * per_tick;
    }
  }
  std::fclose(f);
  return s;
}

const char* platform_name() { return "linux"; }

int logical_cpus() {
  const long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (int)n : 0;
}

}  // namespace knj

#else  // unsupported host -- zeros, so "not measured" can never read as a value

namespace knj {

HostMem host_mem() { return HostMem(); }
CpuTime cpu_time() { return CpuTime(); }
SystemCpu system_cpu() { return SystemCpu(); }
const char* platform_name() { return "unknown"; }
int logical_cpus() { return 0; }

}  // namespace knj

#endif
