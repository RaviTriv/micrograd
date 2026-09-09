#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
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
      source_tensor->to(Backend::CPU);
      out->to(Backend::CPU);
      std::span<scalar_t> gradient = source_tensor->grad();
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

}  // namespace micrograd
