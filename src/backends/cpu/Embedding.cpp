#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cpu/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::ops::cpu {
namespace {

void EmbeddingLookup(const OpArgs &args) {
  size_t dim = args.lhs->shape()[1];
  size_t count = args.rhs->size();
  auto weight = args.lhs->data();
  auto index_values = args.rhs->data();
  auto out_values = args.out->data();

  for (size_t p = 0; p < count; p++) {
    auto index = static_cast<size_t>(std::lround(index_values[p]));
    std::copy_n(&weight[index * dim], dim, &out_values[p * dim]);
  }
}

std::function<void()> EmbeddingLookupBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, indices = args.rhs]() {
    size_t dim = lhs->shape()[1];
    size_t count = indices->size();
    auto gradient = lhs->grad();
    auto incoming = out->grad();
    auto index_values = indices->data();

    for (size_t p = 0; p < count; p++) {
      auto index = static_cast<size_t>(std::lround(index_values[p]));
      for (size_t d = 0; d < dim; d++) {
        gradient[(index * dim) + d] += incoming[(p * dim) + d];
      }
    }
  };
}

}  // namespace

void RegisterEmbeddingOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kEmbeddingLookup, Device::CPU, EmbeddingLookup);
  registry.RegisterBackward(OpId::kEmbeddingLookup, Device::CPU,
                            EmbeddingLookupBackward);
}

}  // namespace micrograd::ops::cpu
