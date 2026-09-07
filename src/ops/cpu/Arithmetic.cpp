#include <cmath>
#include <functional>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"
#include "micrograd/ops/cpu/Ops.h"

namespace micrograd::ops::cpu {
namespace {

void Add(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto rhs = args.rhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] + rhs[i];
  }
}

void Sub(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto rhs = args.rhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] - rhs[i];
  }
}

void Mul(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto rhs = args.rhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] * rhs[i];
  }
}

void Div(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto rhs = args.rhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] / rhs[i];
  }
}

void AddScalar(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] + args.scalar;
  }
}

void SubScalar(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] - args.scalar;
  }
}

void MulScalar(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] * args.scalar;
  }
}

void DivScalar(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] / args.scalar;
  }
}

void Pow(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = std::pow(lhs[i], args.scalar);
  }
}

std::function<void()> AddBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    auto a_grad = lhs->grad();
    auto b_grad = rhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
      b_grad[i] += out_grad[i];
    }
  };
}

std::function<void()> SubBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    auto a_grad = lhs->grad();
    auto b_grad = rhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
      b_grad[i] -= out_grad[i];
    }
  };
}

std::function<void()> MulBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    auto a_data = lhs->data();
    auto b_data = rhs->data();
    auto a_grad = lhs->grad();
    auto b_grad = rhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] * b_data[i];
      b_grad[i] += out_grad[i] * a_data[i];
    }
  };
}

std::function<void()> DivBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    auto a_data = lhs->data();
    auto b_data = rhs->data();
    auto a_grad = lhs->grad();
    auto b_grad = rhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] / b_data[i];
      b_grad[i] -= out_grad[i] * a_data[i] / b_data[i] / b_data[i];
    }
  };
}

std::function<void()> AddScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
    }
  };
}

std::function<void()> SubScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
    }
  };
}

std::function<void()> MulScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, scalar = args.scalar]() {
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] * scalar;
    }
  };
}

std::function<void()> DivScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, scalar = args.scalar]() {
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] / scalar;
    }
  };
}

std::function<void()> PowBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, exponent = args.scalar]() {
    auto a_data = lhs->data();
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] +=
          out_grad[i] * exponent * std::pow(a_data[i], exponent - 1.0f);
    }
  };
}

}  // namespace

void RegisterArithmeticOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kAdd, Device::CPU, Add);
  registry.Register(OpId::kSub, Device::CPU, Sub);
  registry.Register(OpId::kMul, Device::CPU, Mul);
  registry.Register(OpId::kDiv, Device::CPU, Div);
  registry.Register(OpId::kAddScalar, Device::CPU, AddScalar);
  registry.Register(OpId::kSubScalar, Device::CPU, SubScalar);
  registry.Register(OpId::kMulScalar, Device::CPU, MulScalar);
  registry.Register(OpId::kDivScalar, Device::CPU, DivScalar);
  registry.Register(OpId::kPow, Device::CPU, Pow);
  registry.RegisterBackward(OpId::kAdd, Device::CPU, AddBackward);
  registry.RegisterBackward(OpId::kSub, Device::CPU, SubBackward);
  registry.RegisterBackward(OpId::kMul, Device::CPU, MulBackward);
  registry.RegisterBackward(OpId::kDiv, Device::CPU, DivBackward);
  registry.RegisterBackward(OpId::kAddScalar, Device::CPU, AddScalarBackward);
  registry.RegisterBackward(OpId::kSubScalar, Device::CPU, SubScalarBackward);
  registry.RegisterBackward(OpId::kMulScalar, Device::CPU, MulScalarBackward);
  registry.RegisterBackward(OpId::kDivScalar, Device::CPU, DivScalarBackward);
  registry.RegisterBackward(OpId::kPow, Device::CPU, PowBackward);
}

}  // namespace micrograd::ops::cpu
