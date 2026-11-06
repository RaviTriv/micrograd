#include <stdexcept>
#include <utility>
#include <vector>

#include "micrograd/Broadcast.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::apply_binary(OpId op,
                                             const std::shared_ptr<Tensor> &b) {
  if (backend() != b->backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }

  std::vector<size_t> out_shape = BroadcastShapes(shape_, b->shape_);
  auto lhs = broadcast_to(out_shape);
  auto rhs = b->broadcast_to(out_shape);

  return apply(op, std::move(out_shape),
               {.lhs = std::move(lhs), .rhs = std::move(rhs)});
}

std::shared_ptr<Tensor> Tensor::add(const std::shared_ptr<Tensor> &b) {
  return apply_binary(OpId::kAdd, b);
}

std::shared_ptr<Tensor> Tensor::sub(const std::shared_ptr<Tensor> &b) {
  return apply_binary(OpId::kSub, b);
}

std::shared_ptr<Tensor> Tensor::mul(const std::shared_ptr<Tensor> &b) {
  return apply_binary(OpId::kMul, b);
}

std::shared_ptr<Tensor> Tensor::div(const std::shared_ptr<Tensor> &b) {
  return apply_binary(OpId::kDiv, b);
}

std::shared_ptr<Tensor> Tensor::add(scalar_t scalar) {
  return apply(OpId::kAddScalar, shape_,
               {.lhs = shared_from_this(), .scalar = scalar});
}

std::shared_ptr<Tensor> Tensor::sub(scalar_t scalar) {
  return apply(OpId::kSubScalar, shape_,
               {.lhs = shared_from_this(), .scalar = scalar});
}

std::shared_ptr<Tensor> Tensor::mul(scalar_t scalar) {
  return apply(OpId::kMulScalar, shape_,
               {.lhs = shared_from_this(), .scalar = scalar});
}

std::shared_ptr<Tensor> Tensor::div(scalar_t scalar) {
  return apply(OpId::kDivScalar, shape_,
               {.lhs = shared_from_this(), .scalar = scalar});
}

std::shared_ptr<Tensor> Tensor::pow(scalar_t exponent) {
  return apply(OpId::kPow, shape_,
               {.lhs = shared_from_this(), .scalar = exponent});
}

}  // namespace micrograd
