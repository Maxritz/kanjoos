// src/platform/memprobe.h -- C24 platform facts the engine must not guess.
//
// Owns: the per-OS memory and CPU-time probe. This is one of the two places
// allowed to contain `#ifdef _WIN32` (AGENTS.md section 6, the layering rule):
// every platform difference lives in src/device/, kernels/, src/platform/ or
// cmake/arch.cmake, and nothing above them branches on the OS.
//
// Host-only ISO C++17 -- no HIP headers. The device half of the memory picture
// comes from `hipMemGetInfo` at the call site, because a device query must not
// become a build dependency of a host-only translation unit.
//
// Everything returned here is a MEASURED platform fact, never a table lookup
// (docs/00-verified-facts.md section 7 is where the numbers land).
//
// WHY THIS EXISTS
// ---------------
// `docs/06-profiling.md` specifies the C21 profiler (`--profiling`,
// `--profile-dir`, `--profile-floor`, `--profile-warmup`, `--profile-detail`)
// and `docs/01-architecture.md` section 3.1 requires the VRAM ceiling to be
// *measured* rather than assumed. None of that can be honest without a probe
// that reports what the process actually holds in RAM and what the device
// actually has free. This is that probe, in its smallest useful form: RSS and
// its high-water mark, private commit, the system RAM picture, process CPU time
// and system-wide CPU time. Utilisation is derived by a caller that samples
// twice, so the probe itself holds no state and cannot drift.
#pragma once

#include <cstdint>

namespace knj {

// Host memory, as the OS reports it for THIS process.
struct HostMem {
  uint64_t rss_bytes = 0;        // resident set now
  uint64_t peak_rss_bytes = 0;   // high-water mark (VmHWM / PeakWorkingSetSize)
  uint64_t private_bytes = 0;    // private/anonymous commit
  uint64_t phys_total_bytes = 0; // installed physical RAM
  uint64_t phys_avail_bytes = 0; // available to new allocations (not "free")
};

// Cumulative process CPU time. Sample twice and divide by wall time for the
// utilisation over that window; the probe never averages for you.
struct CpuTime {
  uint64_t user_ns = 0;
  uint64_t sys_ns = 0;
  uint64_t total_ns() const { return user_ns + sys_ns; }
};

// Cumulative system-wide CPU time, same sample-twice contract.
struct SystemCpu {
  uint64_t idle_ns = 0;
  uint64_t busy_ns = 0;
};

HostMem host_mem();
CpuTime cpu_time();
SystemCpu system_cpu();

// "win32" | "linux" | "unknown" -- for the one-line provenance in a transcript
// so a captured record can never be mistaken for a different platform's.
const char* platform_name();

// Logical processors usable by this process.
int logical_cpus();

}  // namespace knj
