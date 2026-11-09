#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "micrograd/Broadcast.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {
namespace {

std::vector<size_t> ReducedShape(const std::vector<size_t> &shape, size_t axis,
                                 bool keepdim) {
  std::vector<size_t> reduced;
  for (size_t i = 0; i < shape.size(); i++) {
    if (i != axis) {
      reduced.push_back(shape[i]);
    } else if (keepdim) {
      reduced.push_back(1);
    }
  }
  return reduced;
}

}  // namespace

std::shared_ptr<Tensor> Tensor::sum() {
  return apply(OpId::kSum, {}, {.lhs = shared_from_this()});
}

std::shared_ptr<Tensor> Tensor::sum(int64_t dim, bool keepdim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  return apply(OpId::kSumDim, ReducedShape(shape_, axis, keepdim),
               {.lhs = shared_from_this(), .dim = static_cast<int64_t>(axis)});
}

std::shared_ptr<Tensor> Tensor::mean(int64_t dim, bool keepdim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  return sum(dim, keepdim)->div(static_cast<scalar_t>(shape_[axis]));
}

std::shared_ptr<Tensor> Tensor::max(int64_t dim, bool keepdim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  return apply(OpId::kMax, ReducedShape(shape_, axis, keepdim),
               {.lhs = shared_from_this(), .dim = static_cast<int64_t>(axis)});
}

std::shared_ptr<Tensor> Tensor::argmax(int64_t dim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  auto result = Tensor::zeros(ReducedShape(shape_, axis, false), backend());
  DispatchOp(
      OpId::kArgmax, backend(),
      {.lhs = this, .out = result.get(), .dim = static_cast<int64_t>(axis)});

  return result;
}

}  // namespace micrograd
