#include <functional>

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

std::function<void()> MatmulBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    const size_t m = lhs->shape()[0];
    const size_t k = lhs->shape()[1];
    const size_t n = rhs->shape()[1];

    auto a_data = lhs->data();
    auto b_data = rhs->data();
    auto a_grad = lhs->grad();
    auto b_grad = rhs->grad();
    auto out_grad = out->grad();

    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        for (size_t p = 0; p < n; p++) {
          a_grad[i * k + j] += out_grad[i * n + p] * b_data[j * n + p];
        }
      }
    }

    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        for (size_t p = 0; p < m; p++) {
          b_grad[i * n + j] += a_data[p * k + i] * out_grad[p * n + j];
        }
      }
    }
  };
}

}  // namespace

void RegisterMatmulOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kMatmul, Device::CPU, Matmul);
  registry.RegisterBackward(OpId::kMatmul, Device::CPU, MatmulBackward);
}

}  // namespace micrograd::ops::cpu
