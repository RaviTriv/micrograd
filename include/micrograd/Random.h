#pragma once

#include <cstdint>
#include <random>

namespace micrograd {

std::mt19937_64 &global_rng();
void manual_seed(uint64_t seed);

}  // namespace micrograd
