#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cpu/Ops.h"
#include "micrograd/ops/Dispatch.h"

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

void Cast(const OpArgs &args) {
  const size_t n = args.out->size();
  if (args.out->dtype() == DType::kBFloat16) {
    const auto *source =
        static_cast<const scalar_t *>(args.lhs->data_storage().data());
    auto *target = static_cast<uint16_t *>(args.out->data_storage().data());
    for (size_t i = 0; i < n; i++) {
      target[i] = float_to_bfloat16_bits(source[i]);
    }
    return;
  }
  const auto *source =
      static_cast<const uint16_t *>(args.lhs->data_storage().data());
  auto *target = static_cast<scalar_t *>(args.out->data_storage().data());
  for (size_t i = 0; i < n; i++) {
    target[i] = bfloat16_bits_to_float(source[i]);
  }
}

std::function<void()> CastBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    const size_t n = lhs->size();
    if (lhs->dtype() == DType::kFloat32) {
      auto gradient = lhs->grad();
      const auto *incoming =
          static_cast<const uint16_t *>(out->grad_storage().data());
      for (size_t i = 0; i < n; i++) {
        gradient[i] += bfloat16_bits_to_float(incoming[i]);
      }
      return;
    }
    auto *gradient = static_cast<uint16_t *>(lhs->grad_storage().data());
    auto incoming = out->grad();
    for (size_t i = 0; i < n; i++) {
      gradient[i] = float_to_bfloat16_bits(bfloat16_bits_to_float(gradient[i]) +
                                           incoming[i]);
    }
  };
}

}  // namespace

void RegisterShapeOps() {
  OpRegistry &registry = OpRegistry::Instance();
  for (DType dtype : {DType::kFloat32, DType::kBFloat16}) {
    registry.Register(OpId::kCast, Device::CPU, Cast, dtype);
    registry.RegisterBackward(OpId::kCast, Device::CPU, CastBackward, dtype);
  }
  registry.Register(OpId::kStridedCopy, Device::CPU, StridedCopy);
  registry.RegisterBackward(OpId::kReshape, Device::CPU, ReshapeBackward);
  registry.RegisterBackward(OpId::kStridedCopy, Device::CPU,
                            StridedCopyBackward);
}

}  // namespace micrograd::ops::cpu
