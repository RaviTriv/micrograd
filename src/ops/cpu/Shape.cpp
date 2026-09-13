#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"
#include "micrograd/ops/cpu/Ops.h"

namespace micrograd::ops::cpu {
namespace {

template <typename Visit>
void ForEachStridedIndex(const std::vector<size_t> &shape,
                         std::span<const size_t> strides, Visit visit) {
  size_t total = 1;
  for (size_t dim : shape) {
    total *= dim;
  }

  std::vector<size_t> index(shape.size(), 0);
  for (size_t linear = 0; linear < total; linear++) {
    size_t offset = 0;
    for (size_t d = 0; d < shape.size(); d++) {
      offset += index[d] * strides[d];
    }
    visit(linear, offset);
    for (size_t d = shape.size(); d > 0; d--) {
      if (++index[d - 1] < shape[d - 1]) {
        break;
      }
      index[d - 1] = 0;
    }
  }
}

std::function<void()> ReshapeBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto gradient = lhs->grad();
    auto incoming = out->grad();
    for (size_t i = 0; i < gradient.size(); i++) {
      gradient[i] += incoming[i];
    }
  };
}

void StridedCopy(const OpArgs &args) {
  auto source = args.lhs->data();
  auto values = args.out->data();
  ForEachStridedIndex(
      args.out->shape(), args.strides,
      [&](size_t linear, size_t offset) { values[linear] = source[offset]; });
}

std::function<void()> StridedCopyBackward(const GradArgs &args) {
  auto strides = std::make_shared<const std::vector<size_t>>(
      args.strides.begin(), args.strides.end());
  return [out = args.out, lhs = args.lhs, strides]() {
    auto gradient = lhs->grad();
    auto incoming = out->grad();
    ForEachStridedIndex(out->shape(), *strides,
                        [&](size_t linear, size_t offset) {
                          gradient[offset] += incoming[linear];
                        });
  };
}

}  // namespace

void RegisterShapeOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kStridedCopy, Device::CPU, StridedCopy);
  registry.RegisterBackward(OpId::kReshape, Device::CPU, ReshapeBackward);
  registry.RegisterBackward(OpId::kStridedCopy, Device::CPU,
                            StridedCopyBackward);
}

}  // namespace micrograd::ops::cpu
