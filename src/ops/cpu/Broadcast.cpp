#include "micrograd/Broadcast.h"

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"
#include "micrograd/ops/cpu/Ops.h"

namespace micrograd::ops::cpu {
namespace {

void ReduceBroadcastGradient(std::span<const scalar_t> gradient,
                             const std::vector<size_t> &shape,
                             std::span<scalar_t> reduced,
                             const std::vector<size_t> &target) {
  std::vector<size_t> strides =
      BroadcastStrides(target, ContiguousStrides(target), shape);

  std::vector<size_t> index(shape.size(), 0);
  for (scalar_t value : gradient) {
    size_t reduced_index = 0;
    for (size_t d = 0; d < shape.size(); d++) {
      reduced_index += index[d] * strides[d];
    }
    reduced[reduced_index] += value;
    for (size_t d = shape.size(); d > 0; d--) {
      if (++index[d - 1] < shape[d - 1]) {
        break;
      }
      index[d - 1] = 0;
    }
  }
}

void BroadcastTo(const OpArgs &args) {
  const auto &shape = args.out->shape();
  std::vector<size_t> strides = BroadcastStrides(
      args.lhs->shape(), ContiguousStrides(args.lhs->shape()), shape);

  auto source = args.lhs->data();
  auto values = args.out->data();
  std::vector<size_t> index(shape.size(), 0);
  for (scalar_t &value : values) {
    size_t source_index = 0;
    for (size_t d = 0; d < shape.size(); d++) {
      source_index += index[d] * strides[d];
    }
    value = source[source_index];
    for (size_t d = shape.size(); d > 0; d--) {
      if (++index[d - 1] < shape[d - 1]) {
        break;
      }
      index[d - 1] = 0;
    }
  }
}

std::function<void()> BroadcastToBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto source_grad = lhs->grad();
    ReduceBroadcastGradient(out->grad(), out->shape(), source_grad,
                            lhs->shape());
  };
}

}  // namespace

void RegisterBroadcastOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kBroadcastTo, Device::CPU, BroadcastTo);
  registry.RegisterBackward(OpId::kBroadcastTo, Device::CPU,
                            BroadcastToBackward);
}

}  // namespace micrograd::ops::cpu
