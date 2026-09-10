#include <cmath>
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

size_t AxisOffset(const AxisLayout &layout, size_t outer, size_t position,
                  size_t inner) {
  return (((outer * layout.reduced) + position) * layout.inner) + inner;
}

scalar_t SliceMax(const scalar_t *values, const AxisLayout &layout,
                  size_t outer, size_t inner) {
  scalar_t largest = values[AxisOffset(layout, outer, 0, inner)];
  for (size_t k = 1; k < layout.reduced; k++) {
    scalar_t candidate = values[AxisOffset(layout, outer, k, inner)];
    if (candidate > largest) {
      largest = candidate;
    }
  }
  return largest;
}

}  // namespace

std::shared_ptr<Tensor> Tensor::relu() {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kRelu, backend(), {.lhs = this, .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kRelu, backend(),
                                        {.lhs = self_ptr, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid() {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kSigmoid, backend(), {.lhs = this, .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kSigmoid, backend(),
                                        {.lhs = self_ptr, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::tanh() {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kTanh, backend(), {.lhs = this, .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kTanh, backend(),
                                        {.lhs = self_ptr, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::exp() {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kExp, backend(), {.lhs = this, .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kExp, backend(),
                                        {.lhs = self_ptr, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::log() {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kLog, backend(), {.lhs = this, .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kLog, backend(),
                                        {.lhs = self_ptr, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::sqrt() {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kSqrt, backend(), {.lhs = this, .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kSqrt, backend(),
                                        {.lhs = self_ptr, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::neg() {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kNeg, backend(), {.lhs = this, .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kNeg, backend(),
                                        {.lhs = self_ptr, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::softmax(int64_t dim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  AxisLayout layout = LayoutFor(shape_, axis);

  auto result = std::make_shared<Tensor>(shape_);
  const auto *source = static_cast<const scalar_t *>(data_.host_pointer());
  std::span<scalar_t> values = result->data();
  for (size_t o = 0; o < layout.outer; o++) {
    for (size_t i = 0; i < layout.inner; i++) {
      scalar_t largest = SliceMax(source, layout, o, i);
      scalar_t total = 0;
      for (size_t k = 0; k < layout.reduced; k++) {
        size_t offset = AxisOffset(layout, o, k, i);
        values[offset] = std::exp(source[offset] - largest);
        total += values[offset];
      }
      for (size_t k = 0; k < layout.reduced; k++) {
        values[AxisOffset(layout, o, k, i)] /= total;
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
      std::span<const scalar_t> outputs = out->data();
      std::span<const scalar_t> incoming = out->grad();
      for (size_t o = 0; o < layout.outer; o++) {
        for (size_t i = 0; i < layout.inner; i++) {
          scalar_t weighted = 0;
          for (size_t k = 0; k < layout.reduced; k++) {
            size_t offset = AxisOffset(layout, o, k, i);
            weighted += incoming[offset] * outputs[offset];
          }
          for (size_t k = 0; k < layout.reduced; k++) {
            size_t offset = AxisOffset(layout, o, k, i);
            gradient[offset] += outputs[offset] * (incoming[offset] - weighted);
          }
        }
      }
    };
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::log_softmax(int64_t dim) {
  size_t axis = NormalizeDim(dim, shape_.size());
  AxisLayout layout = LayoutFor(shape_, axis);

  auto result = std::make_shared<Tensor>(shape_);
  const auto *source = static_cast<const scalar_t *>(data_.host_pointer());
  std::span<scalar_t> values = result->data();
  for (size_t o = 0; o < layout.outer; o++) {
    for (size_t i = 0; i < layout.inner; i++) {
      scalar_t largest = SliceMax(source, layout, o, i);
      scalar_t total = 0;
      for (size_t k = 0; k < layout.reduced; k++) {
        total += std::exp(source[AxisOffset(layout, o, k, i)] - largest);
      }
      scalar_t shift = largest + std::log(total);
      for (size_t k = 0; k < layout.reduced; k++) {
        size_t offset = AxisOffset(layout, o, k, i);
        values[offset] = source[offset] - shift;
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
      std::span<const scalar_t> outputs = out->data();
      std::span<const scalar_t> incoming = out->grad();
      for (size_t o = 0; o < layout.outer; o++) {
        for (size_t i = 0; i < layout.inner; i++) {
          scalar_t total = 0;
          for (size_t k = 0; k < layout.reduced; k++) {
            total += incoming[AxisOffset(layout, o, k, i)];
          }
          for (size_t k = 0; k < layout.reduced; k++) {
            size_t offset = AxisOffset(layout, o, k, i);
            gradient[offset] +=
                incoming[offset] - (std::exp(outputs[offset]) * total);
          }
        }
      }
    };
  }

  return result;
}

}  // namespace micrograd
