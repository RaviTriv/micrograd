#include "micrograd/ops/Optimizer.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

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

}  // namespace micrograd::ops
