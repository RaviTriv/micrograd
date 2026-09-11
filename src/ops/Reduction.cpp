#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Broadcast.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {
namespace {

struct AxisLayout {
  size_t outer = 1;
  size_t reduced = 1;
  size_t inner = 1;
};

AxisLayout LayoutFor(const std::vector<size_t> &shape, size_t axis) {
  AxisLayout layout;
  for (size_t i = 0; i < axis; i++) {
    layout.outer *= shape[i];
  }
  layout.reduced = shape[axis];
  for (size_t i = axis + 1; i < shape.size(); i++) {
    layout.inner *= shape[i];
  }
  return layout;
}

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

std::vector<size_t> ArgmaxIndices(const scalar_t *source,
                                  const AxisLayout &layout) {
  std::vector<size_t> indices(layout.outer * layout.inner, 0);
  for (size_t o = 0; o < layout.outer; o++) {
    for (size_t k = 1; k < layout.reduced; k++) {
      for (size_t i = 0; i < layout.inner; i++) {
        size_t slot = (o * layout.inner) + i;
        size_t best =
            (((o * layout.reduced) + indices[slot]) * layout.inner) + i;
        size_t candidate = (((o * layout.reduced) + k) * layout.inner) + i;
        if (source[candidate] > source[best]) {
          indices[slot] = k;
        }
      }
    }
  }
  return indices;
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
  AxisLayout layout = LayoutFor(shape_, axis);

  auto result = std::make_shared<Tensor>(ReducedShape(shape_, axis, keepdim));
  const auto *source = static_cast<const scalar_t *>(data_.host_pointer());
  std::span<scalar_t> values = result->data();
  for (size_t o = 0; o < layout.outer; o++) {
    for (size_t k = 0; k < layout.reduced; k++) {
      for (size_t i = 0; i < layout.inner; i++) {
        values[(o * layout.inner) + i] +=
            source[(((o * layout.reduced) + k) * layout.inner) + i];
      }
    }
  }
  result->to(backend());

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = [source_tensor = self_ptr, out = result.get(),
                            layout]() {
      out->to(Backend::CPU);
      std::span<scalar_t> gradient(
          static_cast<scalar_t *>(source_tensor->grad_storage().host_pointer()),
          source_tensor->size());
      std::span<const scalar_t> incoming = out->grad();
      for (size_t o = 0; o < layout.outer; o++) {
        for (size_t k = 0; k < layout.reduced; k++) {
          for (size_t i = 0; i < layout.inner; i++) {
            gradient[(((o * layout.reduced) + k) * layout.inner) + i] +=
                incoming[(o * layout.inner) + i];
          }
        }
      }
    };
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::mean(int64_t dim, bool keepdim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  return sum(dim, keepdim)->div(static_cast<scalar_t>(shape_[axis]));
}

std::shared_ptr<Tensor> Tensor::max(int64_t dim, bool keepdim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  AxisLayout layout = LayoutFor(shape_, axis);
  const auto *source = static_cast<const scalar_t *>(data_.host_pointer());
  std::vector<size_t> indices = ArgmaxIndices(source, layout);

  auto result = std::make_shared<Tensor>(ReducedShape(shape_, axis, keepdim));
  std::span<scalar_t> values = result->data();
  for (size_t o = 0; o < layout.outer; o++) {
    for (size_t i = 0; i < layout.inner; i++) {
      size_t slot = (o * layout.inner) + i;
      values[slot] =
          source[(((o * layout.reduced) + indices[slot]) * layout.inner) + i];
    }
  }
  result->to(backend());

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = [source_tensor = self_ptr, out = result.get(),
                            layout, positions = std::move(indices)]() {
      out->to(Backend::CPU);
      std::span<scalar_t> gradient(
          static_cast<scalar_t *>(source_tensor->grad_storage().host_pointer()),
          source_tensor->size());
      std::span<const scalar_t> incoming = out->grad();
      for (size_t o = 0; o < layout.outer; o++) {
        for (size_t i = 0; i < layout.inner; i++) {
          size_t slot = (o * layout.inner) + i;
          gradient[(((o * layout.reduced) + positions[slot]) * layout.inner) +
                   i] += incoming[slot];
        }
      }
    };
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::argmax(int64_t dim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  AxisLayout layout = LayoutFor(shape_, axis);
  const auto *source = static_cast<const scalar_t *>(data_.host_pointer());
  std::vector<size_t> indices = ArgmaxIndices(source, layout);

  auto result = std::make_shared<Tensor>(ReducedShape(shape_, axis, false));
  std::span<scalar_t> values = result->data();
  for (size_t slot = 0; slot < indices.size(); slot++) {
    values[slot] = static_cast<scalar_t>(indices[slot]);
  }
  result->to(backend());

  return result;
}

}  // namespace micrograd
