#include "micrograd/Autograd.h"
#include "micrograd/Broadcast.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::add(const std::shared_ptr<Tensor> &b) {
  if (backend() != b->backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }

  std::vector<size_t> out_shape = BroadcastShapes(shape_, b->shape_);
  bool needs_grad = GradEnabled() && (requires_grad_ || b->requires_grad_);

  auto lhs = broadcast_to(out_shape);
  auto rhs = b->broadcast_to(out_shape);

  auto result = std::make_shared<Tensor>(out_shape);
  DispatchOp(OpId::kAdd, backend(),
             {.lhs = lhs.get(), .rhs = rhs.get(), .out = result.get()});

  result->requires_grad_ = needs_grad;
  if (result->requires_grad_) {
    result->children_ = {lhs, rhs};
    result->backward_fn_ = MakeBackward(
        OpId::kAdd, backend(), {.lhs = lhs, .rhs = rhs, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::sub(const std::shared_ptr<Tensor> &b) {
  if (backend() != b->backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }

  std::vector<size_t> out_shape = BroadcastShapes(shape_, b->shape_);
  bool needs_grad = GradEnabled() && (requires_grad_ || b->requires_grad_);

  auto lhs = broadcast_to(out_shape);
  auto rhs = b->broadcast_to(out_shape);

  auto result = std::make_shared<Tensor>(out_shape);
  DispatchOp(OpId::kSub, backend(),
             {.lhs = lhs.get(), .rhs = rhs.get(), .out = result.get()});

  result->requires_grad_ = needs_grad;
  if (result->requires_grad_) {
    result->children_ = {lhs, rhs};
    result->backward_fn_ = MakeBackward(
        OpId::kSub, backend(), {.lhs = lhs, .rhs = rhs, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::mul(const std::shared_ptr<Tensor> &b) {
  if (backend() != b->backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }

  std::vector<size_t> out_shape = BroadcastShapes(shape_, b->shape_);
  bool needs_grad = GradEnabled() && (requires_grad_ || b->requires_grad_);

  auto lhs = broadcast_to(out_shape);
  auto rhs = b->broadcast_to(out_shape);

  auto result = std::make_shared<Tensor>(out_shape);
  DispatchOp(OpId::kMul, backend(),
             {.lhs = lhs.get(), .rhs = rhs.get(), .out = result.get()});

  result->requires_grad_ = needs_grad;
  if (result->requires_grad_) {
    result->children_ = {lhs, rhs};
    result->backward_fn_ = MakeBackward(
        OpId::kMul, backend(), {.lhs = lhs, .rhs = rhs, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::div(const std::shared_ptr<Tensor> &b) {
  if (backend() != b->backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }

  std::vector<size_t> out_shape = BroadcastShapes(shape_, b->shape_);
  bool needs_grad = GradEnabled() && (requires_grad_ || b->requires_grad_);

  auto lhs = broadcast_to(out_shape);
  auto rhs = b->broadcast_to(out_shape);

  auto result = std::make_shared<Tensor>(out_shape);
  DispatchOp(OpId::kDiv, backend(),
             {.lhs = lhs.get(), .rhs = rhs.get(), .out = result.get()});

  result->requires_grad_ = needs_grad;
  if (result->requires_grad_) {
    result->children_ = {lhs, rhs};
    result->backward_fn_ = MakeBackward(
        OpId::kDiv, backend(), {.lhs = lhs, .rhs = rhs, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::add(scalar_t scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kAddScalar, backend(),
             {.lhs = this, .out = result.get(), .scalar = scalar});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ =
        MakeBackward(OpId::kAddScalar, backend(),
                     {.lhs = self_ptr, .out = result.get(), .scalar = scalar});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::sub(scalar_t scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kSubScalar, backend(),
             {.lhs = this, .out = result.get(), .scalar = scalar});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ =
        MakeBackward(OpId::kSubScalar, backend(),
                     {.lhs = self_ptr, .out = result.get(), .scalar = scalar});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::mul(scalar_t scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kMulScalar, backend(),
             {.lhs = this, .out = result.get(), .scalar = scalar});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ =
        MakeBackward(OpId::kMulScalar, backend(),
                     {.lhs = self_ptr, .out = result.get(), .scalar = scalar});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::div(scalar_t scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kDivScalar, backend(),
             {.lhs = this, .out = result.get(), .scalar = scalar});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ =
        MakeBackward(OpId::kDivScalar, backend(),
                     {.lhs = self_ptr, .out = result.get(), .scalar = scalar});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::pow(scalar_t exponent) {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kPow, backend(),
             {.lhs = this, .out = result.get(), .scalar = exponent});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(
        OpId::kPow, backend(),
        {.lhs = self_ptr, .out = result.get(), .scalar = exponent});
  }

  return result;
}

}  // namespace micrograd
