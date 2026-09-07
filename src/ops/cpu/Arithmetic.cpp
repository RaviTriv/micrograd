#include <cmath>

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
}

}  // namespace micrograd::ops::cpu
