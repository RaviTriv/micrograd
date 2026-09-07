#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor> &b) {
  if (shape_.size() != 2 || b->shape_.size() != 2) {
    throw std::invalid_argument("Tensors must be 2D for matmul");
  }

  if (shape_[1] != b->shape_[0]) {
    throw std::invalid_argument("Inner dimensions must match for matmul");
  }

  if (backend() != b->backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }

  auto result =
      std::make_shared<Tensor>(std::vector<size_t>{shape_[0], b->shape_[1]});
  DispatchOp(OpId::kMatmul, backend(),
             {.lhs = this, .rhs = b.get(), .out = result.get()});

  result->requires_grad_ =
      GradEnabled() && (requires_grad_ || b->requires_grad_);
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr, b};
    result->backward_fn_ =
        MakeBackward(OpId::kMatmul, backend(),
                     {.lhs = self_ptr, .rhs = b, .out = result.get()});
  }

  return result;
}

}  // namespace micrograd
