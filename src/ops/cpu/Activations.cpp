#include <cmath>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"
#include "micrograd/ops/cpu/Ops.h"

namespace micrograd::ops::cpu {
namespace {

void Relu(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] > 0 ? lhs[i] : 0.0f;
  }
}

void Sigmoid(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = 1.0f / (1.0f + std::exp(-lhs[i]));
  }
}

void Tanh(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = std::tanh(lhs[i]);
  }
}

}  // namespace

void RegisterActivationOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kRelu, Device::CPU, Relu);
  registry.Register(OpId::kSigmoid, Device::CPU, Sigmoid);
  registry.Register(OpId::kTanh, Device::CPU, Tanh);
}

}  // namespace micrograd::ops::cpu
