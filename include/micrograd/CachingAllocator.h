#pragma once

#include <cstddef>
#include <functional>
#include <set>
#include <string>
#include <unordered_map>

namespace micrograd {

class CachingAllocator {
 public:
  using RawAlloc = std::function<void *(size_t bytes)>;
  using RawFree = std::function<void(void *ptr)>;

  struct Stats {
    size_t allocated_bytes = 0;
    size_t reserved_bytes = 0;
    size_t peak_allocated_bytes = 0;
    size_t peak_reserved_bytes = 0;
    size_t segments = 0;
    size_t alloc_retries = 0;
    size_t out_of_memory = 0;
  };

  static constexpr size_t kMinBlockSize = 512;
  static constexpr size_t kSmallSize = size_t{1} << 20;
  static constexpr size_t kSmallSegment = size_t{2} << 20;
  static constexpr size_t kMinLargeAlloc = size_t{10} << 20;
  static constexpr size_t kLargeSegment = size_t{20} << 20;
  static constexpr size_t kRoundLarge = size_t{2} << 20;

  CachingAllocator(RawAlloc raw_alloc, RawFree raw_free);
  ~CachingAllocator();

  CachingAllocator(const CachingAllocator &) = delete;
  CachingAllocator &operator=(const CachingAllocator &) = delete;

  void *allocate(size_t bytes);
  void deallocate(void *ptr);
  void releaseCached();
  void releaseAll();

  Stats stats() const;
  std::string summary() const;

  static size_t RoundSize(size_t bytes);
  static size_t SegmentSize(size_t rounded);

 private:
  struct Block {
    void *ptr = nullptr;
    size_t size = 0;
    bool small = false;
    bool allocated = false;
    Block *prev = nullptr;
    Block *next = nullptr;
  };

  struct BlockLess {
    bool operator()(const Block *a, const Block *b) const;
  };

  using Pool = std::set<Block *, BlockLess>;

  Pool &PoolFor(const Block &block);
  void *NewSegment(size_t size);
  void ReleasePool(Pool &pool);

  RawAlloc raw_alloc_;
  RawFree raw_free_;
  Pool small_free_;
  Pool large_free_;
  std::unordered_map<void *, Block *> active_;
  Stats stats_;
};

}  // namespace micrograd
