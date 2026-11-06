#include <stdexcept>
#include <utility>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor> &b) {
  if (backend() != b->backend()) {
    throw std::invalid_argument("Tensor devices do not match");
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

  bool fold_rows = lhs_rank == 3 && rhs_rank == 2 && backend() != Device::Metal;
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

}  // namespace micrograd
