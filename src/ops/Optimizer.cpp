#include "micrograd/ops/Optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "micrograd/Tensor.h"

#ifdef MICROGRAD_CUDA_ENABLED
#include "micrograd/backends/cuda/ops/Ops.h"
#endif

namespace micrograd::ops {
namespace {

scalar_t CpuGradNormSquared(const Tensor &param) {
  double sum_sq = 0.0;
  for (scalar_t g : param.grad()) {
    sum_sq += static_cast<double>(g) * static_cast<double>(g);
  }
  return static_cast<scalar_t>(sum_sq);
}

void CpuScaleGrad(Tensor &param, scalar_t scale) {
  for (scalar_t &g : param.grad()) {
    g *= scale;
  }
}

void CpuAdamWStep(Tensor &param, Tensor &m, Tensor &v, scalar_t lr,
                  scalar_t beta1, scalar_t beta2, scalar_t eps,
                  scalar_t weight_decay, bool decay, scalar_t bias_correction1,
                  scalar_t bias_correction2) {
  auto data = param.data();
  auto grad = param.grad();
  auto m_data = m.data();
  auto v_data = v.data();
  for (size_t j = 0; j < data.size(); j++) {
    if (decay) {
      data[j] -= lr * weight_decay * data[j];
    }
    m_data[j] = (beta1 * m_data[j]) + ((1.0f - beta1) * grad[j]);
    v_data[j] = (beta2 * v_data[j]) + ((1.0f - beta2) * grad[j] * grad[j]);
    scalar_t m_hat = m_data[j] / bias_correction1;
    scalar_t v_hat = v_data[j] / bias_correction2;
    data[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
  }
}

std::vector<scalar_t> transpose_matrix(const std::vector<scalar_t> &m,
                                       size_t rows, size_t cols) {
  std::vector<scalar_t> result(rows * cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      result[j * rows + i] = m[i * cols + j];
    }
  }
  return result;
}

void matmul_transpose_b(const std::vector<scalar_t> &a, size_t a_rows,
                        size_t a_cols, const std::vector<scalar_t> &b,
                        size_t b_rows, std::vector<scalar_t> &out) {
  for (size_t i = 0; i < a_rows; i++) {
    for (size_t j = 0; j < b_rows; j++) {
      scalar_t sum = 0;
      for (size_t k = 0; k < a_cols; k++) {
        sum += a[i * a_cols + k] * b[j * a_cols + k];
      }
      out[i * b_rows + j] = sum;
    }
  }
}

void matmul(const std::vector<scalar_t> &a, size_t a_rows, size_t a_cols,
            const std::vector<scalar_t> &b, size_t b_cols,
            std::vector<scalar_t> &out) {
  for (size_t i = 0; i < a_rows; i++) {
    for (size_t j = 0; j < b_cols; j++) {
      scalar_t sum = 0;
      for (size_t k = 0; k < a_cols; k++) {
        sum += a[i * a_cols + k] * b[k * b_cols + j];
      }
      out[i * b_cols + j] = sum;
    }
  }
}

std::vector<scalar_t> zeropower_via_newton_schulz5(
    const std::vector<scalar_t> &g, size_t rows, size_t cols, size_t steps) {
  constexpr scalar_t a = 3.4445f;
  constexpr scalar_t b = -4.7750f;
  constexpr scalar_t c = 2.0315f;

  scalar_t norm = 0;
  for (auto v : g) {
    norm += v * v;
  }
  norm = std::sqrt(norm) + static_cast<scalar_t>(1e-7);

  bool wide = rows <= cols;
  size_t r = wide ? rows : cols;
  size_t cc = wide ? cols : rows;
  std::vector<scalar_t> x = wide ? g : transpose_matrix(g, rows, cols);
  for (auto &v : x) {
    v /= norm;
  }

  std::vector<scalar_t> a_mat(r * r);
  std::vector<scalar_t> a2(r * r);
  std::vector<scalar_t> b_mat(r * r);
  std::vector<scalar_t> bx(r * cc);
  for (size_t step = 0; step < steps; step++) {
    matmul_transpose_b(x, r, cc, x, r, a_mat);
    matmul(a_mat, r, r, a_mat, r, a2);
    for (size_t i = 0; i < r * r; i++) {
      b_mat[i] = b * a_mat[i] + c * a2[i];
    }
    matmul(b_mat, r, r, x, cc, bx);
    for (size_t i = 0; i < r * cc; i++) {
      x[i] = a * x[i] + bx[i];
    }
  }
  return wide ? x : transpose_matrix(x, r, cc);
}

void CpuMuonStep(Tensor &param, Tensor &momentum, scalar_t lr, scalar_t beta,
                 scalar_t weight_decay, bool nesterov, size_t ns_steps,
                 scalar_t update_scale) {
  const size_t rows = param.shape()[0];
  const size_t cols = param.shape()[1];
  auto data = param.data();
  auto grad = param.grad();
  auto buf = momentum.data();
  std::vector<scalar_t> update(data.size());
  for (size_t j = 0; j < data.size(); j++) {
    buf[j] = (beta * buf[j]) + ((1.0f - beta) * grad[j]);
    update[j] = nesterov ? (beta * buf[j]) + ((1.0f - beta) * grad[j]) : buf[j];
  }
  update = zeropower_via_newton_schulz5(update, rows, cols, ns_steps);
  for (size_t j = 0; j < data.size(); j++) {
    data[j] -= lr * weight_decay * data[j];
    data[j] -= lr * update_scale * update[j];
  }
}

}  // namespace

scalar_t GradNormSquared(const Tensor &param) {
  if (param.backend() == Device::CUDA) {
#ifdef MICROGRAD_CUDA_ENABLED
    return cuda::ops::GradNormSquared(param);
#else
    throw std::runtime_error("CUDA support is not compiled in");
#endif
  }
  return CpuGradNormSquared(param);
}

scalar_t ClipGradNorm(const std::vector<std::shared_ptr<Tensor>> &params,
                      scalar_t max_norm) {
  if (params.empty()) {
    return 0.0f;
  }

  const bool all_cuda =
      std::ranges::all_of(params, [](const std::shared_ptr<Tensor> &p) {
        return p->backend() == Device::CUDA;
      });
  if (all_cuda) {
#ifdef MICROGRAD_CUDA_ENABLED
    std::vector<Tensor *> raw;
    raw.reserve(params.size());
    for (const auto &p : params) {
      raw.push_back(p.get());
    }
    return cuda::ops::ClipGradNorm(raw, max_norm);
#else
    throw std::runtime_error("CUDA support is not compiled in");
#endif
  }

  double sum_sq = 0.0;
  for (const auto &p : params) {
    sum_sq += static_cast<double>(GradNormSquared(*p));
  }
  const auto norm = static_cast<scalar_t>(std::sqrt(sum_sq));
  if (max_norm <= 0.0f || norm <= max_norm) {
    return norm;
  }
  const scalar_t scale = max_norm / (norm + 1e-6f);
  for (const auto &p : params) {
    ScaleGrad(*p, scale);
  }
  return norm;
}

void ScaleGrad(Tensor &param, scalar_t scale) {
  if (param.backend() == Device::CUDA) {
#ifdef MICROGRAD_CUDA_ENABLED
    cuda::ops::ScaleGrad(param, scale);
    return;
#else
    throw std::runtime_error("CUDA support is not compiled in");
#endif
  }
  CpuScaleGrad(param, scale);
}

void AdamWStep(Tensor &param, Tensor &m, Tensor &v, scalar_t lr, scalar_t beta1,
               scalar_t beta2, scalar_t eps, scalar_t weight_decay, bool decay,
               scalar_t bias_correction1, scalar_t bias_correction2) {
  if (param.backend() == Device::CUDA) {
#ifdef MICROGRAD_CUDA_ENABLED
    cuda::ops::AdamWStep(param, m, v, lr, beta1, beta2, eps, weight_decay,
                         decay, bias_correction1, bias_correction2);
    return;
#else
    throw std::runtime_error("CUDA support is not compiled in");
#endif
  }
  CpuAdamWStep(param, m, v, lr, beta1, beta2, eps, weight_decay, decay,
               bias_correction1, bias_correction2);
}

void MuonStep(Tensor &param, Tensor &momentum, scalar_t lr, scalar_t beta,
              scalar_t weight_decay, bool nesterov, size_t ns_steps,
              scalar_t update_scale) {
  if (param.shape().size() != 2 || momentum.shape() != param.shape()) {
    throw std::invalid_argument(
        "MuonStep needs a 2-D parameter and a matching momentum buffer");
  }
  if (param.backend() == Device::CUDA) {
#ifdef MICROGRAD_CUDA_ENABLED
    cuda::ops::MuonStep(param, momentum, lr, beta, weight_decay, nesterov,
                        ns_steps, update_scale);
    return;
#else
    throw std::runtime_error("CUDA support is not compiled in");
#endif
  }
  CpuMuonStep(param, momentum, lr, beta, weight_decay, nesterov, ns_steps,
              update_scale);
}

}  // namespace micrograd::ops
