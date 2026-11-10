#include <stdexcept>
#include <utility>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor> &b) {
  if (backend() != b->backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }

  if (dtype() != b->dtype()) {
    throw std::invalid_argument("Tensor dtypes do not match for matmul");
  }

  size_t lhs_rank = shape_.size();
  size_t rhs_rank = b->shape_.size();
  if (lhs_rank < 2 || lhs_rank > 3 || rhs_rank < 2 || rhs_rank > 3) {
    throw std::invalid_argument("Tensors must be 2D or 3D for matmul");
  }

  size_t m = shape_[lhs_rank - 2];
  size_t k = shape_[lhs_rank - 1];
  size_t n = b->shape_[rhs_rank - 1];
  if (k != b->shape_[rhs_rank - 2]) {
    throw std::invalid_argument("Inner dimensions must match for matmul");
  }
  if (lhs_rank == 3 && rhs_rank == 3 && shape_[0] != b->shape_[0]) {
    throw std::invalid_argument("Batch dimensions must match for matmul");
  }

  std::shared_ptr<Tensor> lhs = shared_from_this();
  std::shared_ptr<Tensor> rhs = b;
  std::vector<size_t> out_shape = {m, n};

  bool fold_rows = lhs_rank == 3 && rhs_rank == 2;
  if (fold_rows) {
    out_shape = {shape_[0], m, n};
  } else if (lhs_rank == 3 || rhs_rank == 3) {
    size_t batch = lhs_rank == 3 ? shape_[0] : b->shape_[0];
    lhs = broadcast_to({batch, m, k});
    rhs = b->broadcast_to({batch, k, n});
    out_shape = {batch, m, n};
  }

  return apply(OpId::kMatmul, std::move(out_shape),
               {.lhs = std::move(lhs), .rhs = std::move(rhs)});
}

std::shared_ptr<Tensor> Tensor::matmul_nt(const std::shared_ptr<Tensor> &b) {
  if (backend() != b->backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }
  const bool master_weight =
      b->dtype() == DType::kFloat32 && dtype() != DType::kFloat32;
  if (dtype() != b->dtype() && !master_weight) {
    throw std::invalid_argument("Tensor dtypes do not match for matmul_nt");
  }
  if (shape_.size() < 2 || shape_.size() > 3 || b->shape_.size() != 2) {
    throw std::invalid_argument(
        "matmul_nt expects a 2D or 3D lhs and a 2D rhs");
  }
  if (shape_.back() != b->shape_[1]) {
    throw std::invalid_argument("Inner dimensions must match for matmul_nt");
  }

  std::vector<size_t> out_shape = shape_;
  out_shape.back() = b->shape_[0];
  std::shared_ptr<Tensor> copy;
  if (master_weight) {
    const NoGradGuard no_grad;
    copy = b->cast(dtype());
  }
  return apply(OpId::kMatmulNT, std::move(out_shape),
               {.lhs = shared_from_this(), .rhs = b, .aux = std::move(copy)});
}

std::shared_ptr<Tensor> Tensor::linear(const std::shared_ptr<Tensor> &weight,
                                       const std::shared_ptr<Tensor> &bias) {
  if (backend() != weight->backend() || backend() != bias->backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }
  const bool master_weight = weight->dtype() == DType::kFloat32;
  if ((dtype() != weight->dtype() && !master_weight) ||
      bias->dtype() != DType::kFloat32) {
    throw std::invalid_argument(
        "linear expects a weight of the input dtype or float32, and a float32 "
        "bias");
  }
  if (shape_.size() < 2 || shape_.size() > 3 || weight->shape_.size() != 2 ||
      shape_.back() != weight->shape_[0] || bias->shape_.size() != 1 ||
      bias->shape_[0] != weight->shape_[1]) {
    throw std::invalid_argument(
        "linear expects (..., in) input, (in, out) weight and (out) bias");
  }

  std::vector<size_t> out_shape = shape_;
  out_shape.back() = weight->shape_[1];
  std::shared_ptr<Tensor> copy;
  if (master_weight && dtype() != DType::kFloat32) {
    const NoGradGuard no_grad;
    copy = weight->cast(dtype());
  }
  return apply(OpId::kLinear, std::move(out_shape),
               {.lhs = shared_from_this(),
                .rhs = weight,
                .extra = bias,
                .aux = std::move(copy)});
}

}  // namespace micrograd
