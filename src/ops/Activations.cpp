#include <cstdint>
#include <stdexcept>

#include "micrograd/Broadcast.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::relu() {
  return apply(OpId::kRelu, shape_, {.lhs = shared_from_this()});
}

std::shared_ptr<Tensor> Tensor::sigmoid() {
  return apply(OpId::kSigmoid, shape_, {.lhs = shared_from_this()});
}

std::shared_ptr<Tensor> Tensor::tanh() {
  return apply(OpId::kTanh, shape_, {.lhs = shared_from_this()});
}

std::shared_ptr<Tensor> Tensor::gelu() {
  return apply(OpId::kGelu, shape_, {.lhs = shared_from_this()});
}

std::shared_ptr<Tensor> Tensor::exp() {
  return apply(OpId::kExp, shape_, {.lhs = shared_from_this()});
}

std::shared_ptr<Tensor> Tensor::log() {
  return apply(OpId::kLog, shape_, {.lhs = shared_from_this()});
}

std::shared_ptr<Tensor> Tensor::sqrt() {
  return apply(OpId::kSqrt, shape_, {.lhs = shared_from_this()});
}

std::shared_ptr<Tensor> Tensor::neg() {
  return apply(OpId::kNeg, shape_, {.lhs = shared_from_this()});
}

std::shared_ptr<Tensor> Tensor::softmax(int64_t dim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  if (shape_[axis] == 0) {
    throw std::invalid_argument("softmax over a zero-length dimension");
  }
  return apply(OpId::kSoftmax, shape_,
               {.lhs = shared_from_this(), .dim = static_cast<int64_t>(axis)});
}

std::shared_ptr<Tensor> Tensor::log_softmax(int64_t dim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  if (shape_[axis] == 0) {
    throw std::invalid_argument("log_softmax over a zero-length dimension");
  }
  return apply(OpId::kLogSoftmax, shape_,
               {.lhs = shared_from_this(), .dim = static_cast<int64_t>(axis)});
}

}  // namespace micrograd
