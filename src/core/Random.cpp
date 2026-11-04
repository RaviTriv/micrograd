#include "micrograd/Random.h"

namespace micrograd {

namespace {
constexpr uint64_t kDefaultSeed = 42;
}

std::mt19937_64 &global_rng() {
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  static thread_local std::mt19937_64 rng(kDefaultSeed);
  return rng;
}

void manual_seed(uint64_t seed) { global_rng().seed(seed); }

}  // namespace micrograd
