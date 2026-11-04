#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "micrograd/Autograd.h"
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
  if (reduced.empty()) {
    reduced.push_back(1);
  }
  return reduced;
}

}  // namespace

std::shared_ptr<Tensor> Tensor::sum() {
  auto result = std::make_shared<Tensor>(std::vector<size_t>{1});
  DispatchOp(OpId::kSum, backend(), {.lhs = this, .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kSum, backend(),
                                        {.lhs = self_ptr, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::sum(int64_t dim, bool keepdim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  auto result = std::make_shared<Tensor>(ReducedShape(shape_, axis, keepdim));
  DispatchOp(
      OpId::kSumDim, backend(),
      {.lhs = this, .out = result.get(), .dim = static_cast<int64_t>(axis)});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kSumDim, backend(),
                                        {.lhs = self_ptr,
                                         .out = result.get(),
                                         .dim = static_cast<int64_t>(axis)});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::mean(int64_t dim, bool keepdim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  return sum(dim, keepdim)->div(static_cast<scalar_t>(shape_[axis]));
}

std::shared_ptr<Tensor> Tensor::max(int64_t dim, bool keepdim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  auto result = std::make_shared<Tensor>(ReducedShape(shape_, axis, keepdim));
  DispatchOp(
      OpId::kMax, backend(),
      {.lhs = this, .out = result.get(), .dim = static_cast<int64_t>(axis)});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kMax, backend(),
                                        {.lhs = self_ptr,
                                         .out = result.get(),
                                         .dim = static_cast<int64_t>(axis)});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::argmax(int64_t dim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  auto result = std::make_shared<Tensor>(ReducedShape(shape_, axis, false));
  DispatchOp(
      OpId::kArgmax, backend(),
      {.lhs = this, .out = result.get(), .dim = static_cast<int64_t>(axis)});

  return result;
}

}  // namespace micrograd
