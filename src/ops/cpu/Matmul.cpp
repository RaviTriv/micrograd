#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"
#include "micrograd/ops/cpu/Ops.h"

namespace micrograd::ops::cpu {
namespace {

constexpr size_t kBlock = 64;

void MatmulRows(const scalar_t *lhs_b, const scalar_t *rhs_b, scalar_t *out_b,
                size_t k, size_t n, size_t row_begin, size_t row_end) {
  for (size_t i = row_begin; i < row_end; i++) {
    for (size_t j = 0; j < n; j++) {
      out_b[i * n + j] = 0.0f;
    }
  }
  for (size_t ii = row_begin; ii < row_end; ii += kBlock) {
    const size_t i_max = std::min(ii + kBlock, row_end);
    for (size_t kk = 0; kk < k; kk += kBlock) {
      const size_t k_max = std::min(kk + kBlock, k);
      for (size_t jj = 0; jj < n; jj += kBlock) {
        const size_t j_max = std::min(jj + kBlock, n);
        for (size_t i = ii; i < i_max; i++) {
          for (size_t p = kk; p < k_max; p++) {
            const scalar_t a = lhs_b[i * k + p];
            for (size_t j = jj; j < j_max; j++) {
              out_b[i * n + j] += a * rhs_b[p * n + j];
            }
          }
        }
      }
    }
  }
}

void MatmulBatch(const scalar_t *lhs_b, const scalar_t *rhs_b, scalar_t *out_b,
                 size_t m, size_t k, size_t n) {
  const size_t threads =
      std::max<size_t>(1, std::thread::hardware_concurrency());
  const size_t rows_per_thread = (m + threads - 1) / threads;
  if (threads == 1 || rows_per_thread == 0) {
    MatmulRows(lhs_b, rhs_b, out_b, k, n, 0, m);
    return;
  }

  std::vector<std::jthread> pool;
  for (size_t t = 0; t < threads; t++) {
    const size_t row_begin = t * rows_per_thread;
    if (row_begin >= m) {
      break;
    }
    const size_t row_end = std::min(row_begin + rows_per_thread, m);
    pool.emplace_back(MatmulRows, lhs_b, rhs_b, out_b, k, n, row_begin,
                      row_end);
  }
}

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
    MatmulBatch(lhs_b, rhs_b, out_b, m, k, n);
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
