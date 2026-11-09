#include "micrograd/CachingAllocator.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

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
    : raw_alloc_(std::move(raw_alloc)), raw_free_(std::move(raw_free)) {}

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

CachingAllocator::Pool &CachingAllocator::PoolFor(const Block &block) {
  return block.small ? small_free_ : large_free_;
}

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
    block = new Block{.ptr = ptr, .size = segment, .small = small};
    stats_.reserved_bytes += segment;
    stats_.peak_reserved_bytes =
        std::max(stats_.peak_reserved_bytes, stats_.reserved_bytes);
    stats_.segments++;
  }

  const size_t remaining = block->size - size;
  const bool split =
      small ? remaining >= kMinBlockSize : remaining > kSmallSize;
  if (split) {
    auto *rest = new Block{.ptr = static_cast<char *>(block->ptr) + size,
                           .size = remaining,
                           .small = small,
                           .prev = block,
                           .next = block->next};
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
  if (ptr == nullptr) {
    return;
  }
  auto it = active_.find(ptr);
  if (it == active_.end()) {
    return;
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
    delete block;
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
    delete next;
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
      delete block;
    } else {
      ++it;
    }
  }
}

void CachingAllocator::releaseCached() {
  ReleasePool(small_free_);
  ReleasePool(large_free_);
}

void CachingAllocator::releaseAll() {
  std::vector<Block *> heads;
  const auto collect = [&heads](Block *block) {
    if (block->prev == nullptr) {
      heads.push_back(block);
    }
  };
  for (Block *block : small_free_) {
    collect(block);
  }
  for (Block *block : large_free_) {
    collect(block);
  }
  for (const auto &[ptr, block] : active_) {
    collect(block);
  }

  for (Block *head : heads) {
    raw_free_(head->ptr);
    for (Block *block = head; block != nullptr;) {
      Block *next = block->next;
      delete block;
      block = next;
    }
  }
  small_free_.clear();
  large_free_.clear();
  active_.clear();
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
