// src/util/parallel.h -- a persistent worker pool for the CPU matmuls.
//
// Not std::async and not a fresh std::thread per call: a decode step issues
// hundreds of small matmuls, and thread creation (~20 us each) would cost more
// than the arithmetic. Workers park on a condition variable and are woken once
// per parallel region.
//
// The pool deliberately has no work-stealing and no task graph. The only
// parallelism the engine needs today is "split these independent output tiles
// across N workers and wait", which is exactly `for_each`.
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace knj {

class ThreadPool {
 public:
  // `n` = 0 means one worker per hardware thread.
  explicit ThreadPool(int n = 0);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // Spawned worker threads. The calling thread participates too, so the
  // number of threads a parallel region actually uses is total().
  int size() const { return (int)workers_.size(); }
  int total() const { return (int)workers_.size() + 1; }

  // Runs fn(i) for every i in [0, count). The calling thread participates, so
  // the pool is usable even at size 1. Returns once every index is done.
  void for_each(int count, const std::function<void(int)>& fn);

 private:
  void worker_loop(int index);

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  const std::function<void(int)>* job_ = nullptr;
  int  count_ = 0;
  int  next_ = 0;             // shared work index for dynamic balancing
  int  remaining_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

}  // namespace knj
