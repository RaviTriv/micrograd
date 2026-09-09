#include <algorithm>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Broadcast.h"
#include "micrograd/Tensor.h"

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

template <typename Visit>
void ForEachStridedIndex(const std::vector<size_t> &shape,
                         const std::vector<size_t> &strides, Visit visit) {
  size_t total = 1;
  for (size_t dim : shape) {
    total *= dim;
  }

  std::vector<size_t> index(shape.size(), 0);
  for (size_t linear = 0; linear < total; linear++) {
    size_t offset = 0;
    for (size_t d = 0; d < shape.size(); d++) {
      offset += index[d] * strides[d];
    }
    visit(linear, offset);
    for (size_t d = shape.size(); d > 0; d--) {
      if (++index[d - 1] < shape[d - 1]) {
        break;
      }
      index[d - 1] = 0;
    }
  }
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
    result->backward_fn_ = [source = self_ptr, out = result.get()]() {
      source->to(Backend::CPU);
      out->to(Backend::CPU);
      std::ranges::transform(source->grad(), out->grad(),
                             source->grad().begin(), std::plus<>{});
    };
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::view(const std::vector<int64_t> &shape) {
  return reshape(shape);
}

std::shared_ptr<Tensor> Tensor::strided_copy(
    const std::vector<size_t> &shape, const std::vector<size_t> &strides) {
  const auto *source = static_cast<const scalar_t *>(data_.host_pointer());

  auto result = std::make_shared<Tensor>(shape);
  std::span<scalar_t> values = result->data();
  ForEachStridedIndex(shape, strides, [&](size_t linear, size_t offset) {
    values[linear] = source[offset];
  });
  result->to(backend());

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    auto source_strides = std::make_shared<const std::vector<size_t>>(strides);
    result->children_ = {self_ptr};
    result->backward_fn_ = [source_tensor = self_ptr, out = result.get(),
                            source_strides]() {
      source_tensor->to(Backend::CPU);
      out->to(Backend::CPU);
      std::span<scalar_t> gradient = source_tensor->grad();
      std::span<const scalar_t> incoming = out->grad();
      ForEachStridedIndex(out->shape(), *source_strides,
                          [&](size_t linear, size_t offset) {
                            gradient[offset] += incoming[linear];
                          });
    };
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
