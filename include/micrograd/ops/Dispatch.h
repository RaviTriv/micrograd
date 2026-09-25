#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include "micrograd/DType.h"
#include "micrograd/Device.h"
#include "micrograd/Scalar.h"

namespace micrograd {

class Tensor;

enum class OpId {
  kAdd,
  kSub,
  kMul,
  kDiv,
  kAddScalar,
  kSubScalar,
  kMulScalar,
  kDivScalar,
  kPow,
  kSum,
  kMatmul,
  kRelu,
  kSigmoid,
  kTanh,
  kExp,
  kLog,
  kSqrt,
  kNeg,
  kSoftmax,
  kLogSoftmax,
  kSumDim,
  kMean,
  kMax,
  kArgmax,
  kReshape,
  kStridedCopy,
  kBroadcastTo,
};

inline constexpr size_t kOpCount = static_cast<size_t>(OpId::kBroadcastTo) + 1;
inline constexpr size_t kDeviceCount = static_cast<size_t>(Device::CUDA) + 1;

struct OpArgs {
  const Tensor *lhs = nullptr;
  const Tensor *rhs = nullptr;
  Tensor *out = nullptr;
  scalar_t scalar = 0;
  int64_t dim = 0;
  bool keepdim = false;
  std::span<const size_t> strides = {};
};

using OpFn = void (*)(const OpArgs &);

struct GradArgs {
  std::shared_ptr<Tensor> lhs = nullptr;
  std::shared_ptr<Tensor> rhs = nullptr;
  Tensor *out = nullptr;
  scalar_t scalar = 0;
  int64_t dim = 0;
  std::span<const size_t> strides = {};
};

using GradFn = std::function<void()> (*)(const GradArgs &);

class OpRegistry {
 public:
  static OpRegistry &Instance() {
    static OpRegistry registry;
    return registry;
  }

  void Register(OpId op, Device device, OpFn fn,
                DType dtype = DType::kFloat32) {
    Slot(op, device, dtype) = fn;
  }

  void RegisterBackward(OpId op, Device device, GradFn fn,
                        DType dtype = DType::kFloat32) {
    BackwardSlot(op, device, dtype) = fn;
  }

  OpFn Lookup(OpId op, Device device, DType dtype = DType::kFloat32) const {
    OpFn fn = Slot(op, device, dtype);
    if (fn == nullptr) {
      throw std::runtime_error(
          "No kernel registered for op " +
          std::to_string(static_cast<size_t>(op)) + " on device " +
          std::to_string(static_cast<size_t>(device)) + " dtype " +
          std::to_string(static_cast<size_t>(dtype)));
    }
    return fn;
  }

  GradFn LookupBackward(OpId op, Device device,
                        DType dtype = DType::kFloat32) const {
    GradFn fn = BackwardSlot(op, device, dtype);
    if (fn == nullptr) {
      throw std::runtime_error(
          "No backward kernel registered for op " +
          std::to_string(static_cast<size_t>(op)) + " on device " +
          std::to_string(static_cast<size_t>(device)) + " dtype " +
          std::to_string(static_cast<size_t>(dtype)));
    }
    return fn;
  }

 private:
  OpRegistry() = default;

  static size_t Index(OpId op, Device device, DType dtype) {
    return (static_cast<size_t>(op) * kDeviceCount +
            static_cast<size_t>(device)) *
               kDTypeCount +
           static_cast<size_t>(dtype);
  }

  OpFn &Slot(OpId op, Device device, DType dtype) {
    return table_[Index(op, device, dtype)];
  }
  const OpFn &Slot(OpId op, Device device, DType dtype) const {
    return table_[Index(op, device, dtype)];
  }

  GradFn &BackwardSlot(OpId op, Device device, DType dtype) {
    return backward_table_[Index(op, device, dtype)];
  }
  const GradFn &BackwardSlot(OpId op, Device device, DType dtype) const {
    return backward_table_[Index(op, device, dtype)];
  }

  std::array<OpFn, kOpCount * kDeviceCount * kDTypeCount> table_{};
  std::array<GradFn, kOpCount * kDeviceCount * kDTypeCount> backward_table_{};
};

void DispatchOp(OpId op, Device device, const OpArgs &args);

std::function<void()> MakeBackward(OpId op, Device device,
                                   const GradArgs &args);

}  // namespace micrograd
