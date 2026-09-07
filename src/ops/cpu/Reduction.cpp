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

}  // namespace

void RegisterReductionOps() {
  OpRegistry::Instance().Register(OpId::kSum, Device::CPU, Sum);
}

}  // namespace micrograd::ops::cpu
