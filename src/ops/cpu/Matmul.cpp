#include <functional>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"
#include "micrograd/ops/cpu/Ops.h"

namespace micrograd::ops::cpu {
namespace {

void Matmul(const OpArgs &args) {
  const auto &lhs_shape = args.lhs->shape();
  const auto &rhs_shape = args.rhs->shape();
  const size_t rank = lhs_shape.size();
  const size_t batch = rank == 3 ? lhs_shape[0] : 1;
  const size_t m = lhs_shape[rank - 2];
  const size_t k = lhs_shape[rank - 1];
  const size_t n = rhs_shape[rank - 1];

  auto lhs = args.lhs->data();
  auto rhs = args.rhs->data();
  auto out = args.out->data();
  for (size_t bidx = 0; bidx < batch; bidx++) {
    const scalar_t *lhs_b = lhs.data() + bidx * m * k;
    const scalar_t *rhs_b = rhs.data() + bidx * k * n;
    scalar_t *out_b = out.data() + bidx * m * n;
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        scalar_t sum = 0.0f;
        for (size_t p = 0; p < k; p++) {
          sum += lhs_b[i * k + p] * rhs_b[p * n + j];
        }
        out_b[i * n + j] = sum;
      }
    }
  }
}

std::function<void()> MatmulBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    const auto &lhs_shape = lhs->shape();
    const auto &rhs_shape = rhs->shape();
    const size_t rank = lhs_shape.size();
    const size_t batch = rank == 3 ? lhs_shape[0] : 1;
    const size_t m = lhs_shape[rank - 2];
    const size_t k = lhs_shape[rank - 1];
    const size_t n = rhs_shape[rank - 1];

    auto a_data = lhs->data();
    auto b_data = rhs->data();
    auto a_grad = lhs->grad();
    auto b_grad = rhs->grad();
    auto out_grad = out->grad();

    for (size_t bidx = 0; bidx < batch; bidx++) {
      const scalar_t *a_data_b = a_data.data() + bidx * m * k;
      const scalar_t *b_data_b = b_data.data() + bidx * k * n;
      scalar_t *a_grad_b = a_grad.data() + bidx * m * k;
      scalar_t *b_grad_b = b_grad.data() + bidx * k * n;
      const scalar_t *out_grad_b = out_grad.data() + bidx * m * n;

      for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
          for (size_t p = 0; p < n; p++) {
            a_grad_b[i * k + j] += out_grad_b[i * n + p] * b_data_b[j * n + p];
          }
        }
      }

      for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < n; j++) {
          for (size_t p = 0; p < m; p++) {
            b_grad_b[i * n + j] += a_data_b[p * k + i] * out_grad_b[p * n + j];
          }
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
