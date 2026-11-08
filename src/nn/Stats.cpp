#include "micrograd/nn/Stats.h"

#include <cstddef>
#include <cstdint>

namespace micrograd {

size_t param_count(const nn::Module &module) {
  size_t total = 0;
  for (const auto &param : module.parameters()) {
    total += param->size();
  }
  return total;
}

size_t param_bytes(const nn::Module &module) {
  size_t total = 0;
  for (const auto &param : module.parameters()) {
    total += param->data_storage().bytes();
  }
  return total;
}

uint64_t linear_flops(size_t batch, size_t in_features, size_t out_features,
                      bool bias) {
  const auto b = static_cast<uint64_t>(batch);
  const auto out = static_cast<uint64_t>(out_features);
  const uint64_t matmul = 2 * b * static_cast<uint64_t>(in_features) * out;
  return bias ? matmul + b * out : matmul;
}

}  // namespace micrograd
