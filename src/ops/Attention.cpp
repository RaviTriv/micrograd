#include <cstddef>
#include <memory>
#include <stdexcept>

#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::flash_attention(
    const std::shared_ptr<Tensor> &key, const std::shared_ptr<Tensor> &value,
    scalar_t scale) {
  if (shape_.size() != 3) {
    throw std::invalid_argument(
        "flash_attention expects a (batch_heads, seq, head_dim) input");
  }
  if (key->shape() != shape_ || value->shape() != shape_) {
    throw std::invalid_argument(
        "flash_attention expects query, key, and value to share a shape");
  }

  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kFlashAttention, backend(),
             {.lhs = this,
              .rhs = key.get(),
              .extra = value.get(),
              .out = result.get(),
              .scalar = scale});

  result->requires_grad_ =
      GradEnabled() &&
      (requires_grad_ || key->requires_grad() || value->requires_grad());
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr, key, value};
    result->backward_fn_ = MakeBackward(OpId::kFlashAttention, backend(),
                                        {.lhs = self_ptr,
                                         .rhs = key,
                                         .extra = value,
                                         .out = result.get(),
                                         .scalar = scale});
  }

  return result;
}

}  // namespace micrograd
