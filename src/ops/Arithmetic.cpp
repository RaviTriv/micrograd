#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::add(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }
  if (backend() != b->backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }

  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kAdd, backend(),
             {.lhs = this, .rhs = b.get(), .out = result.get()});

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};
  result->backward_fn_ = MakeBackward(
      OpId::kAdd, backend(), {.lhs = self_ptr, .rhs = b, .out = result.get()});

  return result;
}

std::shared_ptr<Tensor> Tensor::sub(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }
  if (backend() != b->backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }

  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kSub, backend(),
             {.lhs = this, .rhs = b.get(), .out = result.get()});

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};
  result->backward_fn_ = MakeBackward(
      OpId::kSub, backend(), {.lhs = self_ptr, .rhs = b, .out = result.get()});

  return result;
}

std::shared_ptr<Tensor> Tensor::mul(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }
  if (backend() != b->backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }

  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kMul, backend(),
             {.lhs = this, .rhs = b.get(), .out = result.get()});

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};
  result->backward_fn_ = MakeBackward(
      OpId::kMul, backend(), {.lhs = self_ptr, .rhs = b, .out = result.get()});

  return result;
}

std::shared_ptr<Tensor> Tensor::div(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }
  if (backend() != b->backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }

  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kDiv, backend(),
             {.lhs = this, .rhs = b.get(), .out = result.get()});

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};
  result->backward_fn_ = MakeBackward(
      OpId::kDiv, backend(), {.lhs = self_ptr, .rhs = b, .out = result.get()});

  return result;
}

std::shared_ptr<Tensor> Tensor::add(scalar_t scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kAddScalar, backend(),
             {.lhs = this, .out = result.get(), .scalar = scalar});

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};
  result->backward_fn_ =
      MakeBackward(OpId::kAddScalar, backend(),
                   {.lhs = self_ptr, .out = result.get(), .scalar = scalar});

  return result;
}

std::shared_ptr<Tensor> Tensor::sub(scalar_t scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kSubScalar, backend(),
             {.lhs = this, .out = result.get(), .scalar = scalar});

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};
  result->backward_fn_ =
      MakeBackward(OpId::kSubScalar, backend(),
                   {.lhs = self_ptr, .out = result.get(), .scalar = scalar});

  return result;
}

std::shared_ptr<Tensor> Tensor::mul(scalar_t scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kMulScalar, backend(),
             {.lhs = this, .out = result.get(), .scalar = scalar});

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};
  result->backward_fn_ =
      MakeBackward(OpId::kMulScalar, backend(),
                   {.lhs = self_ptr, .out = result.get(), .scalar = scalar});

  return result;
}

std::shared_ptr<Tensor> Tensor::div(scalar_t scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kDivScalar, backend(),
             {.lhs = this, .out = result.get(), .scalar = scalar});

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};
  result->backward_fn_ =
      MakeBackward(OpId::kDivScalar, backend(),
                   {.lhs = self_ptr, .out = result.get(), .scalar = scalar});

  return result;
}

std::shared_ptr<Tensor> Tensor::pow(scalar_t exponent) {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kPow, backend(),
             {.lhs = this, .out = result.get(), .scalar = exponent});

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};
  result->backward_fn_ =
      MakeBackward(OpId::kPow, backend(),
                   {.lhs = self_ptr, .out = result.get(), .scalar = exponent});

  return result;
}

}  // namespace micrograd
