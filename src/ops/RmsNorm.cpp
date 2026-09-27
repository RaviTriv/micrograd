#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::rms_norm(
    const std::vector<size_t> &normalized_shape,
    const std::shared_ptr<Tensor> &gain, scalar_t eps) {
  if (normalized_shape.size() > shape_.size()) {
    throw std::invalid_argument(
        "rms_norm normalized_shape has more dimensions than the input");
  }

  size_t leading = shape_.size() - normalized_shape.size();
  for (size_t i = 0; i < normalized_shape.size(); i++) {
    if (shape_[leading + i] != normalized_shape[i]) {
      throw std::invalid_argument(
          "rms_norm input shape does not end with normalized_shape");
    }
  }

  size_t n = 1;
  for (auto dim : normalized_shape) {
    n *= dim;
  }
  if (gain->size() != n) {
    throw std::invalid_argument("rms_norm gain must match normalized_shape");
  }

  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(
      OpId::kRmsNorm, backend(),
      {.lhs = this, .rhs = gain.get(), .out = result.get(), .scalar = eps});

  result->requires_grad_ =
      GradEnabled() && (requires_grad_ || gain->requires_grad());
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr, gain};
    result->backward_fn_ = MakeBackward(
        OpId::kRmsNorm, backend(),
        {.lhs = self_ptr, .rhs = gain, .out = result.get(), .scalar = eps});
  }

  return result;
}

}  // namespace micrograd
