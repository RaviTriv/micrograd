#include <cmath>

#include "micrograd/Tensor.h"

#ifdef MICROGRAD_METAL_ENABLED
#include "micrograd/metal/MetalContext.h"
#endif

namespace micrograd {

std::shared_ptr<Tensor> Tensor::add(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }

#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == Backend::Metal && b->backend_ == Backend::Metal) {
    return add_metal(b);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] + b->data_[i];
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
      b->grad_[i] += result->grad_[i];
    }
  };
  return result;
}

std::shared_ptr<Tensor> Tensor::sub(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }

#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == Backend::Metal && b->backend_ == Backend::Metal) {
    return sub_metal(b);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] - b->data_[i];
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
      b->grad_[i] -= result->grad_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::mul(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }

#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == Backend::Metal && b->backend_ == Backend::Metal) {
    return mul_metal(b);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] * b->data_[i];
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] * b->data_[i];
      b->grad_[i] += result->grad_[i] * self_ptr->data_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::div(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }

#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == Backend::Metal && b->backend_ == Backend::Metal) {
    return div_metal(b);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] / b->data_[i];
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] / b->data_[i];
      b->grad_[i] -=
          result->grad_[i] * self_ptr->data_[i] / b->data_[i] / b->data_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::add(scalar_t scalar) {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == Backend::Metal) {
    return add_scalar_metal(scalar);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] + scalar;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sub(scalar_t scalar) {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == Backend::Metal) {
    return sub_scalar_metal(scalar);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] - scalar;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::mul(scalar_t scalar) {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == Backend::Metal) {
    return mul_scalar_metal(scalar);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] * scalar;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr, scalar]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] * scalar;
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::div(scalar_t scalar) {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == Backend::Metal) {
    return div_scalar_metal(scalar);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] / scalar;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr, scalar]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] / scalar;
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::pow(scalar_t exponent) {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == Backend::Metal) {
    return pow_metal(exponent);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = std::pow(data_[i], exponent);
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr, exponent]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] * exponent *
                            std::pow(self_ptr->data_[i], exponent - 1.0f);
    }
  };

  return result;
}

}  // namespace micrograd
