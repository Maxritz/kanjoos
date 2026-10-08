// src/util/parallel.cpp -- persistent worker pool.
#include "src/util/parallel.h"

namespace knj {

ThreadPool::ThreadPool(int n) {
  if (n <= 0) n = (int)std::thread::hardware_concurrency();
  if (n <= 0) n = 1;
  // n includes the calling thread.
  workers_.reserve(n > 1 ? n - 1 : 0);
  for (int i = 1; i < n; ++i) workers_.emplace_back([this, i] { worker_loop(i); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
    ++generation_;
  }
  cv_start_.notify_all();
  for (std::thread& t : workers_) t.join();
}

void ThreadPool::worker_loop(int index) {
  (void)index;
  uint64_t seen = 0;
  for (;;) {
    const std::function<void(int)>* job = nullptr;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_start_.wait(lock, [this, &seen] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
      job = job_;
    }
    if (job) {
      for (;;) {
        int i;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (next_ >= count_) break;
          i = next_++;
        }
        (*job)(i);
      }
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (--remaining_ == 0) cv_done_.notify_all();
    }
  }
}

void ThreadPool::for_each(int count, const std::function<void(int)>& fn) {
  if (count <= 0) return;
  if (workers_.empty()) {
    for (int i = 0; i < count; ++i) fn(i);
    return;
  }
  {
    std::unique_lock<std::mutex> lock(mutex_);
    job_ = &fn;
    count_ = count;
    next_ = 0;
    remaining_ = (int)workers_.size();
    ++generation_;
  }
  cv_start_.notify_all();
  // The calling thread pulls from the same queue, so the pool uses every core.
  for (;;) {
    int i;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (next_ >= count_) break;
      i = next_++;
    }
    fn(i);
  }
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [this] { return remaining_ == 0; });
    job_ = nullptr;
  }
}

}  // namespace knj
