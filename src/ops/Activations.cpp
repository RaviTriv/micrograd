#include <cmath>

#include "micrograd/Tensor.h"

#ifdef MICROGRAD_METAL_ENABLED
#include "micrograd/metal/MetalContext.h"
#endif

namespace micrograd {

std::shared_ptr<Tensor> Tensor::relu() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend() == Backend::Metal) {
    return relu_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  auto lhs = data();
  auto out = result->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] > 0 ? lhs[i] : 0.0f;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    auto a_data = self_ptr->data();
    auto a_grad = self_ptr->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] * (a_data[i] > 0 ? 1.0f : 0.0f);
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend() == Backend::Metal) {
    return sigmoid_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  auto lhs = data();
  auto out = result->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = 1.0f / (1.0f + std::exp(-lhs[i]));
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    auto out_data = result->data();
    auto a_grad = self_ptr->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      scalar_t sigmoid_val = out_data[i];
      a_grad[i] += out_grad[i] * sigmoid_val * (1.0f - sigmoid_val);
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::tanh() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend() == Backend::Metal) {
    return tanh_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  auto lhs = data();
  auto out = result->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = std::tanh(lhs[i]);
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    auto out_data = result->data();
    auto a_grad = self_ptr->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      scalar_t tanh_val = out_data[i];
      a_grad[i] += out_grad[i] * (1.0f - tanh_val * tanh_val);
    }
  };

  return result;
}

}  // namespace micrograd
