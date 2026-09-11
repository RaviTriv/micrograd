#include "micrograd/Autograd.h"
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

  bool needs_grad = GradEnabled() && (requires_grad_ || b->requires_grad_);

  std::shared_ptr<Tensor> lhs = shared_from_this();
  std::shared_ptr<Tensor> rhs = b;
  std::vector<size_t> out_shape = {m, n};

  if (lhs_rank == 3 || rhs_rank == 3) {
    size_t batch = lhs_rank == 3 ? shape_[0] : b->shape_[0];
    lhs = broadcast_to({batch, m, k});
    rhs = b->broadcast_to({batch, k, n});
    out_shape = {batch, m, n};
  }

  auto result = std::make_shared<Tensor>(out_shape);
  DispatchOp(OpId::kMatmul, backend(),
             {.lhs = lhs.get(), .rhs = rhs.get(), .out = result.get()});

  result->requires_grad_ = needs_grad;
  if (result->requires_grad_) {
    result->children_ = {lhs, rhs};
    result->backward_fn_ =
        MakeBackward(OpId::kMatmul, backend(),
                     {.lhs = lhs, .rhs = rhs, .out = result.get()});
  }

  return result;
}

}  // namespace micrograd
