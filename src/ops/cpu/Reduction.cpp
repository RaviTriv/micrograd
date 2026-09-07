#include <functional>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"
#include "micrograd/ops/cpu/Ops.h"

namespace micrograd::ops::cpu {
namespace {

void Sum(const OpArgs &args) {
  scalar_t total = 0.0f;
  for (scalar_t value : args.lhs->data()) {
    total += value;
  }
  args.out->data()[0] = total;
}

std::function<void()> SumBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    const scalar_t out_grad = out->grad()[0];
    for (scalar_t &g : lhs->grad()) {
      g += out_grad;
    }
  };
}

}  // namespace

void RegisterReductionOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kSum, Device::CPU, Sum);
  registry.RegisterBackward(OpId::kSum, Device::CPU, SumBackward);
}

}  // namespace micrograd::ops::cpu
