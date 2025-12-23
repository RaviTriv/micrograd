#include "micrograd/Tensor.h"
#include <cmath>

#ifdef MICROGRAD_METAL_ENABLED
#include "micrograd/metal/MetalContext.h"
#endif

std::shared_ptr<Tensor> Tensor::relu() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal) {
    return relu_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "relu";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] > 0 ? data_[i] : 0;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] +=
          result->grad_[i] * (self_ptr->data_[i] > 0 ? 1.0 : 0.0);
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal) {
    return sigmoid_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "sigmoid";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = 1.0 / (1.0 + std::exp(-data_[i]));
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      double sigmoid_val = result->data_[i];
      self_ptr->grad_[i] +=
          result->grad_[i] * sigmoid_val * (1.0 - sigmoid_val);
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::tanh() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal) {
    return tanh_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "tanh";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = std::tanh(data_[i]);
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      double tanh_val = result->data_[i];
      self_ptr->grad_[i] += result->grad_[i] * (1.0 - tanh_val * tanh_val);
    }
  };

  return result;
}
