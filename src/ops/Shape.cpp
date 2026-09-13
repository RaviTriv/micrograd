#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Broadcast.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {
namespace {

std::vector<size_t> ResolveShape(const std::vector<int64_t> &shape,
                                 size_t total) {
  std::vector<size_t> resolved(shape.size());
  size_t inferred_count = 0;
  size_t inferred_position = 0;
  size_t known = 1;

  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] == -1) {
      inferred_count++;
      inferred_position = i;
      continue;
    }
    if (shape[i] < 0) {
      throw std::invalid_argument("Reshape dimensions must be -1 or positive");
    }
    resolved[i] = static_cast<size_t>(shape[i]);
    known *= resolved[i];
  }

  if (inferred_count > 1) {
    throw std::invalid_argument(
        "Reshape allows at most one inferred dimension");
  }

  if (inferred_count == 1) {
    if (known == 0 || total % known != 0) {
      throw std::invalid_argument("Cannot infer reshape dimension");
    }
    resolved[inferred_position] = total / known;
    known = total;
  }

  if (known != total) {
    throw std::invalid_argument("Reshape element count mismatch");
  }

  return resolved;
}

std::vector<size_t> NormalizeDims(const std::vector<int64_t> &dims,
                                  size_t rank) {
  if (dims.size() != rank) {
    throw std::invalid_argument("Permute needs one entry per dimension");
  }

  std::vector<size_t> resolved(rank);
  std::vector<bool> seen(rank, false);
  for (size_t i = 0; i < rank; i++) {
    resolved[i] = NormalizeDim(dims[i], rank);
    if (seen[resolved[i]]) {
      throw std::invalid_argument("Permute dimensions must be unique");
    }
    seen[resolved[i]] = true;
  }
  return resolved;
}

}  // namespace

std::shared_ptr<Tensor> Tensor::reshape(const std::vector<int64_t> &shape) {
  std::vector<size_t> resolved = ResolveShape(shape, size());

  auto result = std::make_shared<Tensor>(resolved);
  result->data_ = data_.copy_to(backend());
  result->grad_ = Storage(grad_.bytes(), backend());
  result->zero_grad();

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kReshape, backend(),
                                        {.lhs = self_ptr, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::view(const std::vector<int64_t> &shape) {
  return reshape(shape);
}

std::shared_ptr<Tensor> Tensor::strided_copy(
    const std::vector<size_t> &shape, const std::vector<size_t> &strides) {
  auto result = std::make_shared<Tensor>(shape);
  DispatchOp(OpId::kStridedCopy, backend(),
             {.lhs = this, .out = result.get(), .strides = strides});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(
        OpId::kStridedCopy, backend(),
        {.lhs = self_ptr, .out = result.get(), .strides = strides});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::permute(const std::vector<int64_t> &dims) {
  std::vector<size_t> order = NormalizeDims(dims, shape_.size());

  std::vector<size_t> shape(order.size());
  std::vector<size_t> strides(order.size());
  for (size_t i = 0; i < order.size(); i++) {
    shape[i] = shape_[order[i]];
    strides[i] = strides_[order[i]];
  }

  return strided_copy(shape, strides);
}

std::shared_ptr<Tensor> Tensor::transpose(int64_t dim0, int64_t dim1) {
  size_t rank = shape_.size();
  size_t first = NormalizeDim(dim0, rank);
  size_t second = NormalizeDim(dim1, rank);

  std::vector<int64_t> dims(rank);
  for (size_t i = 0; i < rank; i++) {
    dims[i] = static_cast<int64_t>(i);
  }
  std::swap(dims[first], dims[second]);

  return permute(dims);
}

std::shared_ptr<Tensor> Tensor::contiguous() {
  if (strides_ == ContiguousStrides(shape_)) {
    return shared_from_this();
  }
  return strided_copy(shape_, strides_);
}

}  // namespace micrograd
