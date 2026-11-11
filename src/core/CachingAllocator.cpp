#include "micrograd/CachingAllocator.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace micrograd {

bool CachingAllocator::BlockLess::operator()(const Block *a,
                                             const Block *b) const {
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return reinterpret_cast<std::uintptr_t>(a->ptr) <
         reinterpret_cast<std::uintptr_t>(b->ptr);
}

CachingAllocator::CachingAllocator(RawAlloc raw_alloc, RawFree raw_free)
    : raw_alloc_(std::move(raw_alloc)),
      raw_free_(std::move(raw_free)),
      owner_(std::this_thread::get_id()) {}

CachingAllocator::~CachingAllocator() { releaseAll(); }

size_t CachingAllocator::RoundSize(size_t bytes) {
  if (bytes < kMinBlockSize) {
    return kMinBlockSize;
  }
  return kMinBlockSize * ((bytes + kMinBlockSize - 1) / kMinBlockSize);
}

size_t CachingAllocator::SegmentSize(size_t rounded) {
  if (rounded <= kSmallSize) {
    return kSmallSegment;
  }
  if (rounded < kMinLargeAlloc) {
    return kLargeSegment;
  }
  return kRoundLarge * ((rounded + kRoundLarge - 1) / kRoundLarge);
}

void CachingAllocator::CheckOwnerThread() const {
#ifndef NDEBUG
  if (owner_ != std::this_thread::get_id()) {
    throw std::logic_error(
        "CachingAllocator is single-threaded, use it only from the thread "
        "that created it");
  }
#endif
}

CachingAllocator::Pool &CachingAllocator::PoolFor(const Block &block) {
  return block.small ? small_free_ : large_free_;
}

CachingAllocator::Block *CachingAllocator::NewBlock(const Block &value) {
  auto owned = std::make_unique<Block>(value);
  Block *block = owned.get();
  blocks_.emplace(block, std::move(owned));
  return block;
}

void CachingAllocator::DeleteBlock(Block *block) { blocks_.erase(block); }

void *CachingAllocator::NewSegment(size_t size) {
  void *ptr = raw_alloc_(size);
  if (ptr != nullptr) {
    return ptr;
  }
  stats_.alloc_retries++;
  releaseCached();
  return raw_alloc_(size);
}

void *CachingAllocator::allocate(size_t bytes) {
  CheckOwnerThread();
  const size_t size = RoundSize(bytes);
  const bool small = size <= kSmallSize;
  Pool &pool = small ? small_free_ : large_free_;

  Block key;
  key.size = size;
  Block *block = nullptr;
  auto it = pool.lower_bound(&key);
  if (it != pool.end()) {
    block = *it;
    pool.erase(it);
  } else {
    const size_t segment = SegmentSize(size);
    void *ptr = NewSegment(segment);
    if (ptr == nullptr) {
      stats_.out_of_memory++;
      return nullptr;
    }
    try {
      block = NewBlock({.ptr = ptr, .size = segment, .small = small});
    } catch (...) {
      raw_free_(ptr);
      throw;
    }
    stats_.reserved_bytes += segment;
    stats_.peak_reserved_bytes =
        std::max(stats_.peak_reserved_bytes, stats_.reserved_bytes);
    stats_.segments++;
  }

  const size_t remaining = block->size - size;
  const bool split =
      small ? remaining >= kMinBlockSize : remaining > kSmallSize;
  if (split) {
    Block *rest = NewBlock({.ptr = static_cast<char *>(block->ptr) + size,
                            .size = remaining,
                            .small = small,
                            .prev = block,
                            .next = block->next});
    if (block->next != nullptr) {
      block->next->prev = rest;
    }
    block->next = rest;
    block->size = size;
    pool.insert(rest);
  }

  block->allocated = true;
  active_[block->ptr] = block;
  stats_.allocated_bytes += block->size;
  stats_.peak_allocated_bytes =
      std::max(stats_.peak_allocated_bytes, stats_.allocated_bytes);
  return block->ptr;
}

void CachingAllocator::deallocate(void *ptr) {
  CheckOwnerThread();
  if (ptr == nullptr) {
    return;
  }
  auto it = active_.find(ptr);
  if (it == active_.end()) {
    if (retired_.erase(ptr) > 0) {
      return;
    }
    throw std::invalid_argument(
        "CachingAllocator::deallocate got a pointer it does not own or that "
        "was already freed");
  }
  Block *block = it->second;
  active_.erase(it);
  stats_.allocated_bytes -= block->size;
  block->allocated = false;

  Pool &pool = PoolFor(*block);
  if (block->prev != nullptr && !block->prev->allocated) {
    Block *prev = block->prev;
    pool.erase(prev);
    prev->size += block->size;
    prev->next = block->next;
    if (block->next != nullptr) {
      block->next->prev = prev;
    }
    DeleteBlock(block);
    block = prev;
  }
  if (block->next != nullptr && !block->next->allocated) {
    Block *next = block->next;
    pool.erase(next);
    block->size += next->size;
    block->next = next->next;
    if (next->next != nullptr) {
      next->next->prev = block;
    }
    DeleteBlock(next);
  }
  pool.insert(block);
}

void CachingAllocator::ReleasePool(Pool &pool) {
  for (auto it = pool.begin(); it != pool.end();) {
    Block *block = *it;
    if (block->prev == nullptr && block->next == nullptr) {
      raw_free_(block->ptr);
      stats_.reserved_bytes -= block->size;
      stats_.segments--;
      it = pool.erase(it);
      DeleteBlock(block);
    } else {
      ++it;
    }
  }
}

void CachingAllocator::releaseCached() {
  CheckOwnerThread();
  ReleasePool(small_free_);
  ReleasePool(large_free_);
}

void CachingAllocator::releaseAll() {
  CheckOwnerThread();
  for (const auto &[block, owned] : blocks_) {
    if (block->prev == nullptr) {
      raw_free_(block->ptr);
    }
  }
  for (const auto &[ptr, block] : active_) {
    retired_.insert(ptr);
  }
  small_free_.clear();
  large_free_.clear();
  active_.clear();
  blocks_.clear();
  stats_.allocated_bytes = 0;
  stats_.reserved_bytes = 0;
  stats_.segments = 0;
}

CachingAllocator::Stats CachingAllocator::stats() const { return stats_; }

std::string CachingAllocator::summary() const {
  const auto mib = [](size_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(size_t{1} << 20);
  };
  char line[256];
  std::snprintf(line, sizeof(line),
                "allocated %.1f MiB (peak %.1f), reserved %.1f MiB (peak "
                "%.1f) in %zu segments, %zu retries, %zu out-of-memory",
                mib(stats_.allocated_bytes), mib(stats_.peak_allocated_bytes),
                mib(stats_.reserved_bytes), mib(stats_.peak_reserved_bytes),
                stats_.segments, stats_.alloc_retries, stats_.out_of_memory);
  return line;
}

}  // namespace micrograd
