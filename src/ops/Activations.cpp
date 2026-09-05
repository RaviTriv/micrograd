#include <cmath>

#include "micrograd/Tensor.h"

#ifdef MICROGRAD_METAL_ENABLED
#include "micrograd/metal/MetalContext.h"
#endif

namespace micrograd {

std::shared_ptr<Tensor> Tensor::relu() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == Backend::Metal) {
    return relu_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] > 0 ? data_[i] : 0.0f;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] +=
          result->grad_[i] * (self_ptr->data_[i] > 0 ? 1.0f : 0.0f);
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == Backend::Metal) {
    return sigmoid_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = 1.0f / (1.0f + std::exp(-data_[i]));
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      scalar_t sigmoid_val = result->data_[i];
      self_ptr->grad_[i] +=
          result->grad_[i] * sigmoid_val * (1.0f - sigmoid_val);
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::tanh() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == Backend::Metal) {
    return tanh_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = std::tanh(data_[i]);
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      scalar_t tanh_val = result->data_[i];
      self_ptr->grad_[i] += result->grad_[i] * (1.0f - tanh_val * tanh_val);
    }
  };

  return result;
}

}  // namespace micrograd
