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
  if (backend() == Backend::Metal && b->backend() == Backend::Metal) {
    return add_metal(b);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  auto lhs = data();
  auto rhs = b->data();
  auto out = result->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] + rhs[i];
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    auto a_grad = self_ptr->grad();
    auto b_grad = b->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
      b_grad[i] += out_grad[i];
    }
  };
  return result;
}

std::shared_ptr<Tensor> Tensor::sub(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }

#ifdef MICROGRAD_METAL_ENABLED
  if (backend() == Backend::Metal && b->backend() == Backend::Metal) {
    return sub_metal(b);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  auto lhs = data();
  auto rhs = b->data();
  auto out = result->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] - rhs[i];
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    auto a_grad = self_ptr->grad();
    auto b_grad = b->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
      b_grad[i] -= out_grad[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::mul(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }

#ifdef MICROGRAD_METAL_ENABLED
  if (backend() == Backend::Metal && b->backend() == Backend::Metal) {
    return mul_metal(b);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  auto lhs = data();
  auto rhs = b->data();
  auto out = result->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] * rhs[i];
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    auto a_data = self_ptr->data();
    auto b_data = b->data();
    auto a_grad = self_ptr->grad();
    auto b_grad = b->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] * b_data[i];
      b_grad[i] += out_grad[i] * a_data[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::div(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }

#ifdef MICROGRAD_METAL_ENABLED
  if (backend() == Backend::Metal && b->backend() == Backend::Metal) {
    return div_metal(b);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  auto lhs = data();
  auto rhs = b->data();
  auto out = result->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] / rhs[i];
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b]() {
    auto a_data = self_ptr->data();
    auto b_data = b->data();
    auto a_grad = self_ptr->grad();
    auto b_grad = b->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] / b_data[i];
      b_grad[i] -= out_grad[i] * a_data[i] / b_data[i] / b_data[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::add(scalar_t scalar) {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend() == Backend::Metal) {
    return add_scalar_metal(scalar);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  auto lhs = data();
  auto out = result->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] + scalar;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    auto a_grad = self_ptr->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sub(scalar_t scalar) {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend() == Backend::Metal) {
    return sub_scalar_metal(scalar);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  auto lhs = data();
  auto out = result->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] - scalar;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    auto a_grad = self_ptr->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::mul(scalar_t scalar) {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend() == Backend::Metal) {
    return mul_scalar_metal(scalar);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  auto lhs = data();
  auto out = result->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] * scalar;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr, scalar]() {
    auto a_grad = self_ptr->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] * scalar;
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::div(scalar_t scalar) {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend() == Backend::Metal) {
    return div_scalar_metal(scalar);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  auto lhs = data();
  auto out = result->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] / scalar;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr, scalar]() {
    auto a_grad = self_ptr->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] / scalar;
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::pow(scalar_t exponent) {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend() == Backend::Metal) {
    return pow_metal(exponent);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);

  auto lhs = data();
  auto out = result->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = std::pow(lhs[i], exponent);
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr, exponent]() {
    auto a_data = self_ptr->data();
    auto a_grad = self_ptr->grad();
    auto out_grad = result->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] +=
          out_grad[i] * exponent * std::pow(a_data[i], exponent - 1.0f);
    }
  };

  return result;
}

}  // namespace micrograd
