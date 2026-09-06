#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

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
};

inline constexpr size_t kOpCount = static_cast<size_t>(OpId::kTanh) + 1;
inline constexpr size_t kDeviceCount = static_cast<size_t>(Device::CUDA) + 1;

struct OpArgs {
  const Tensor *lhs = nullptr;
  const Tensor *rhs = nullptr;
  Tensor *out = nullptr;
  scalar_t scalar = 0;
};

using OpFn = void (*)(const OpArgs &);

class OpRegistry {
 public:
  static OpRegistry &Instance() {
    static OpRegistry registry;
    return registry;
  }

  void Register(OpId op, Device device, OpFn fn) { Slot(op, device) = fn; }

  OpFn Find(OpId op, Device device) const { return Slot(op, device); }

  OpFn Lookup(OpId op, Device device) const {
    OpFn fn = Slot(op, device);
    if (fn == nullptr) {
      throw std::runtime_error("No kernel registered for op " +
                               std::to_string(static_cast<size_t>(op)) +
                               " on device " +
                               std::to_string(static_cast<size_t>(device)));
    }
    return fn;
  }

 private:
  OpRegistry() = default;

  static size_t Index(OpId op, Device device) {
    return static_cast<size_t>(op) * kDeviceCount + static_cast<size_t>(device);
  }

  OpFn &Slot(OpId op, Device device) { return table_[Index(op, device)]; }
  const OpFn &Slot(OpId op, Device device) const {
    return table_[Index(op, device)];
  }

  std::array<OpFn, kOpCount * kDeviceCount> table_{};
};

class OpRegistrar {
 public:
  OpRegistrar(OpId op, Device device, OpFn fn) {
    OpRegistry::Instance().Register(op, device, fn);
  }
};

}  // namespace micrograd

#define MICROGRAD_OP_REGISTRAR_CONCAT_(a, b) a##b
#define MICROGRAD_OP_REGISTRAR_NAME_(a, b) MICROGRAD_OP_REGISTRAR_CONCAT_(a, b)

#define REGISTER_OP(op, device, fn)                                   \
  static const ::micrograd::OpRegistrar MICROGRAD_OP_REGISTRAR_NAME_( \
      micrograd_op_registrar_, __LINE__)(::micrograd::OpId::op,       \
                                         ::micrograd::Device::device, fn)
