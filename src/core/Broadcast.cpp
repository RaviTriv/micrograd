#include "micrograd/Broadcast.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {
namespace {

size_t DimensionAt(const std::vector<size_t> &shape, size_t rank,
                   size_t position) {
  size_t offset = rank - shape.size();
  return position < offset ? 1 : shape[position - offset];
}

}  // namespace

std::vector<size_t> ContiguousStrides(const std::vector<size_t> &shape) {
  std::vector<size_t> strides(shape.size());
  size_t stride = 1;
  for (size_t i = shape.size(); i > 0; i--) {
    strides[i - 1] = stride;
    stride *= shape[i - 1];
  }
  return strides;
}

size_t NormalizeDim(int64_t dim, size_t rank) {
  auto limit = static_cast<int64_t>(rank);
  int64_t resolved = dim < 0 ? dim + limit : dim;
  if (resolved < 0 || resolved >= limit) {
    throw std::out_of_range("Dimension out of range");
  }
  return static_cast<size_t>(resolved);
}

std::vector<size_t> BroadcastShapes(const std::vector<size_t> &a,
                                    const std::vector<size_t> &b) {
  size_t rank = std::max(a.size(), b.size());
  std::vector<size_t> result(rank);
  for (size_t i = 0; i < rank; i++) {
    size_t dim_a = DimensionAt(a, rank, i);
    size_t dim_b = DimensionAt(b, rank, i);
    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
      throw std::invalid_argument("Tensor shapes are not broadcastable");
    }
    result[i] = dim_a == 1 ? dim_b : dim_a;
  }
  return result;
}

std::vector<size_t> BroadcastStrides(const std::vector<size_t> &shape,
                                     const std::vector<size_t> &strides,
                                     const std::vector<size_t> &target) {
  if (shape.size() != strides.size()) {
    throw std::invalid_argument("Shape and strides rank mismatch");
  }
  if (shape.size() > target.size()) {
    throw std::invalid_argument("Cannot broadcast to a lower rank");
  }

  size_t offset = target.size() - shape.size();
  std::vector<size_t> result(target.size(), 0);
  for (size_t i = offset; i < target.size(); i++) {
    size_t dim = shape[i - offset];
    if (dim == target[i]) {
      result[i] = strides[i - offset];
    } else if (dim != 1) {
      throw std::invalid_argument("Tensor shapes are not broadcastable");
    }
  }
  return result;
}

std::shared_ptr<Tensor> Tensor::broadcast_to(const std::vector<size_t> &shape) {
  auto self_ptr = shared_from_this();
  if (shape_ == shape) {
    return self_ptr;
  }

  auto result = std::make_shared<Tensor>(shape);
  DispatchOp(OpId::kBroadcastTo, backend(), {.lhs = this, .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kBroadcastTo, backend(),
                                        {.lhs = self_ptr, .out = result.get()});
  }

  return result;
}

}  // namespace micrograd
