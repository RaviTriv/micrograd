#include "micrograd/Broadcast.h"

#include <algorithm>
#include <stdexcept>

#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"

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

void ReduceBroadcastGradient(std::span<const scalar_t> gradient,
                             const std::vector<size_t> &shape,
                             std::span<scalar_t> reduced,
                             const std::vector<size_t> &target) {
  std::vector<size_t> strides =
      BroadcastStrides(target, ContiguousStrides(target), shape);

  std::vector<size_t> index(shape.size(), 0);
  for (scalar_t value : gradient) {
    size_t reduced_index = 0;
    for (size_t d = 0; d < shape.size(); d++) {
      reduced_index += index[d] * strides[d];
    }
    reduced[reduced_index] += value;
    for (size_t d = shape.size(); d > 0; d--) {
      if (++index[d - 1] < shape[d - 1]) {
        break;
      }
      index[d - 1] = 0;
    }
  }
}

std::shared_ptr<Tensor> BroadcastTo(const std::shared_ptr<Tensor> &tensor,
                                    const std::vector<size_t> &shape) {
  if (tensor->shape() == shape) {
    return tensor;
  }

  std::vector<size_t> strides = BroadcastStrides(
      tensor->shape(), ContiguousStrides(tensor->shape()), shape);
  const auto *source =
      static_cast<const scalar_t *>(tensor->data_storage().host_pointer());

  auto result = std::make_shared<Tensor>(shape);
  std::vector<size_t> index(shape.size(), 0);
  for (scalar_t &value : result->data()) {
    size_t source_index = 0;
    for (size_t d = 0; d < shape.size(); d++) {
      source_index += index[d] * strides[d];
    }
    value = source[source_index];
    for (size_t d = shape.size(); d > 0; d--) {
      if (++index[d - 1] < shape[d - 1]) {
        break;
      }
      index[d - 1] = 0;
    }
  }

  result->to(tensor->backend());
  return result;
}

std::shared_ptr<Tensor> Tensor::broadcast_to(const std::vector<size_t> &shape) {
  auto self_ptr = shared_from_this();
  if (shape_ == shape) {
    return self_ptr;
  }

  auto result = BroadcastTo(self_ptr, shape);
  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    result->children_ = {self_ptr};
    result->backward_fn_ = [source = self_ptr, out = result.get()]() {
      source->to(Backend::CPU);
      out->to(Backend::CPU);
      ReduceBroadcastGradient(out->grad(), out->shape(), source->grad(),
                              source->shape());
    };
  }

  return result;
}

}  // namespace micrograd
