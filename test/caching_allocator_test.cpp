#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "micrograd/CachingAllocator.h"

using micrograd::CachingAllocator;

namespace {

constexpr size_t kMiB = size_t{1} << 20;

struct FakeDevice {
  explicit FakeDevice(size_t capacity_bytes) : capacity(capacity_bytes) {}

  void *Allocate(size_t bytes) {
    if (used + bytes > capacity) {
      return nullptr;
    }
    const std::uintptr_t address = next;
    next += bytes + 4096;
    used += bytes;
    live[address] = bytes;
    allocations++;
    return reinterpret_cast<void *>(address);
  }

  void Free(void *ptr) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    used -= live.at(address);
    live.erase(address);
    frees++;
  }

  size_t capacity;
  size_t used = 0;
  std::uintptr_t next = 0x100000;
  std::unordered_map<std::uintptr_t, size_t> live;
  size_t allocations = 0;
  size_t frees = 0;
};

std::unique_ptr<CachingAllocator> MakeAllocator(FakeDevice &device) {
  return std::make_unique<CachingAllocator>(
      [&device](size_t bytes) { return device.Allocate(bytes); },
      [&device](void *ptr) { device.Free(ptr); });
}

char *Bytes(void *ptr) { return static_cast<char *>(ptr); }

}  // namespace

TEST(CachingAllocator, RoundsSizesLikePyTorch) {
  EXPECT_EQ(CachingAllocator::RoundSize(1), 512u);
  EXPECT_EQ(CachingAllocator::RoundSize(512), 512u);
  EXPECT_EQ(CachingAllocator::RoundSize(513), 1024u);
  EXPECT_EQ(CachingAllocator::SegmentSize(512), 2 * kMiB);
  EXPECT_EQ(CachingAllocator::SegmentSize(1 * kMiB), 2 * kMiB);
  EXPECT_EQ(CachingAllocator::SegmentSize(5 * kMiB), 20 * kMiB);
  EXPECT_EQ(CachingAllocator::SegmentSize(11 * kMiB), 12 * kMiB);
}

TEST(CachingAllocator, SmallRequestsShareOneSegment) {
  FakeDevice device(64 * kMiB);
  auto allocator = MakeAllocator(device);

  void *a = allocator->allocate(100);
  void *b = allocator->allocate(700);

  EXPECT_EQ(device.allocations, 1u);
  EXPECT_EQ(Bytes(b), Bytes(a) + 512);
  EXPECT_EQ(allocator->stats().allocated_bytes, 512u + 1024u);
  EXPECT_EQ(allocator->stats().reserved_bytes, 2 * kMiB);
}

TEST(CachingAllocator, ReusesFreedBlockWithoutNewSegment) {
  FakeDevice device(64 * kMiB);
  auto allocator = MakeAllocator(device);

  void *a = allocator->allocate(3 * kMiB);
  allocator->deallocate(a);
  void *b = allocator->allocate(3 * kMiB);

  EXPECT_EQ(a, b);
  EXPECT_EQ(device.allocations, 1u);
}

TEST(CachingAllocator, FreedNeighboursMergeBackIntoOneSegment) {
  FakeDevice device(64 * kMiB);
  auto allocator = MakeAllocator(device);

  void *a = allocator->allocate(4 * kMiB);
  void *b = allocator->allocate(4 * kMiB);
  void *c = allocator->allocate(4 * kMiB);
  EXPECT_EQ(device.allocations, 1u);

  allocator->deallocate(a);
  allocator->deallocate(c);
  allocator->deallocate(b);
  allocator->releaseCached();

  EXPECT_EQ(device.frees, 1u);
  EXPECT_EQ(device.used, 0u);
  EXPECT_EQ(allocator->stats().reserved_bytes, 0u);
}

TEST(CachingAllocator, LargeBlockServesSmallerLargeRequestBySplitting) {
  FakeDevice device(64 * kMiB);
  auto allocator = MakeAllocator(device);

  void *big = allocator->allocate(16 * kMiB);
  allocator->deallocate(big);
  void *a = allocator->allocate(6 * kMiB);
  void *b = allocator->allocate(6 * kMiB);

  EXPECT_EQ(device.allocations, 1u);
  EXPECT_EQ(a, big);
  EXPECT_EQ(Bytes(b), Bytes(a) + (6 * kMiB));
}

TEST(CachingAllocator, FreesCacheAndRetriesWhenDeviceIsFull) {
  FakeDevice device(11 * kMiB);
  auto allocator = MakeAllocator(device);

  void *small = allocator->allocate(1024);
  allocator->deallocate(small);
  EXPECT_EQ(device.used, 2 * kMiB);

  void *large = allocator->allocate(10 * kMiB);

  EXPECT_NE(large, nullptr);
  EXPECT_EQ(allocator->stats().alloc_retries, 1u);
  EXPECT_EQ(device.used, 10 * kMiB);
}

TEST(CachingAllocator, ReturnsNullWhenRetryAlsoFails) {
  FakeDevice device(8 * kMiB);
  auto allocator = MakeAllocator(device);

  void *ptr = allocator->allocate(10 * kMiB);

  EXPECT_EQ(ptr, nullptr);
  EXPECT_EQ(allocator->stats().out_of_memory, 1u);
}

TEST(CachingAllocator, TracksPeakAllocatedBytes) {
  FakeDevice device(64 * kMiB);
  auto allocator = MakeAllocator(device);

  void *a = allocator->allocate(12 * kMiB);
  void *b = allocator->allocate(12 * kMiB);
  allocator->deallocate(a);
  allocator->deallocate(b);

  EXPECT_EQ(allocator->stats().allocated_bytes, 0u);
  EXPECT_EQ(allocator->stats().peak_allocated_bytes, 24 * kMiB);
}

TEST(CachingAllocator, ReleaseAllReturnsEverySegment) {
  FakeDevice device(64 * kMiB);
  auto allocator = MakeAllocator(device);

  allocator->allocate(100);
  allocator->allocate(12 * kMiB);
  void *freed = allocator->allocate(3 * kMiB);
  allocator->deallocate(freed);
  allocator->releaseAll();

  EXPECT_EQ(device.used, 0u);
  EXPECT_EQ(device.frees, device.allocations);
}

TEST(CachingAllocator, DoubleFreeThrows) {
  FakeDevice device(64 * kMiB);
  auto allocator = MakeAllocator(device);

  void *a = allocator->allocate(100);
  allocator->deallocate(a);

  EXPECT_THROW(allocator->deallocate(a), std::invalid_argument);
}

TEST(CachingAllocator, UnknownPointerThrows) {
  FakeDevice device(64 * kMiB);
  auto allocator = MakeAllocator(device);
  void *a = allocator->allocate(100);
  int local = 0;

  EXPECT_THROW(allocator->deallocate(&local), std::invalid_argument);
  EXPECT_THROW(allocator->deallocate(Bytes(a) + 1), std::invalid_argument);
  EXPECT_EQ(allocator->stats().allocated_bytes,
            CachingAllocator::RoundSize(100));
}

TEST(CachingAllocator, FreeOfABlockLiveAtReleaseAllIsAcceptedOnce) {
  FakeDevice device(64 * kMiB);
  auto allocator = MakeAllocator(device);
  void *a = allocator->allocate(100);
  allocator->releaseAll();

  EXPECT_NO_THROW(allocator->deallocate(a));
  EXPECT_THROW(allocator->deallocate(a), std::invalid_argument);
  EXPECT_EQ(device.used, 0u);
}

TEST(CachingAllocator, UseFromAnotherThreadThrowsInDebugBuilds) {
#ifdef NDEBUG
  GTEST_SKIP() << "the owner-thread check is compiled out with NDEBUG";
#else
  FakeDevice device(64 * kMiB);
  auto allocator = MakeAllocator(device);
  bool threw = false;
  std::thread worker([&threw, &allocator] {
    try {
      allocator->allocate(100);
    } catch (const std::logic_error &) {
      threw = true;
    }
  });
  worker.join();

  EXPECT_TRUE(threw);
  EXPECT_EQ(allocator->stats().allocated_bytes, 0u);
#endif
}
