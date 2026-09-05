#include "micrograd/Random.h"

namespace micrograd {

std::mt19937_64 &global_rng() {
  static thread_local std::mt19937_64 rng(std::random_device{}());
  return rng;
}

void manual_seed(uint64_t seed) { global_rng().seed(seed); }

}  // namespace micrograd
