#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"
#include "micrograd/ops/cpu/Ops.h"

namespace micrograd::ops::cpu {
namespace {

void Matmul(const OpArgs &args) {
  const size_t m = args.lhs->shape()[0];
  const size_t k = args.lhs->shape()[1];
  const size_t n = args.rhs->shape()[1];

  auto lhs = args.lhs->data();
  auto rhs = args.rhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      scalar_t sum = 0.0f;
      for (size_t p = 0; p < k; p++) {
        sum += lhs[i * k + p] * rhs[p * n + j];
      }
      out[i * n + j] = sum;
    }
  }
}

}  // namespace

void RegisterMatmulOps() {
  OpRegistry::Instance().Register(OpId::kMatmul, Device::CPU, Matmul);
}

}  // namespace micrograd::ops::cpu
