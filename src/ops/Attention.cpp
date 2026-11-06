#include <cstddef>
#include <memory>
#include <stdexcept>

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

  return apply(
      OpId::kFlashAttention, shape_,
      {.lhs = shared_from_this(), .rhs = key, .extra = value, .scalar = scale});
}

}  // namespace micrograd
