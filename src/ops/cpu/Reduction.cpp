#include <cstddef>
#include <functional>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"
#include "micrograd/ops/cpu/Ops.h"

namespace micrograd::ops::cpu {
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

void Sum(const OpArgs &args) {
  scalar_t total = 0.0f;
  for (scalar_t value : args.lhs->data()) {
    total += value;
  }
  args.out->data()[0] = total;
}

std::function<void()> SumBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    const scalar_t out_grad = out->grad()[0];
    for (scalar_t &g : lhs->grad()) {
      g += out_grad;
    }
  };
}

void SumDim(const OpArgs &args) {
  auto axis = static_cast<size_t>(args.dim);
  AxisLayout layout = LayoutFor(args.lhs->shape(), axis);
  auto source = args.lhs->data();
  auto values = args.out->data();
  for (size_t o = 0; o < layout.outer; o++) {
    for (size_t k = 0; k < layout.reduced; k++) {
      for (size_t i = 0; i < layout.inner; i++) {
        values[(o * layout.inner) + i] +=
            source[(((o * layout.reduced) + k) * layout.inner) + i];
      }
    }
  }
}

std::function<void()> SumDimBackward(const GradArgs &args) {
  auto axis = static_cast<size_t>(args.dim);
  return [out = args.out, lhs = args.lhs, axis]() {
    AxisLayout layout = LayoutFor(lhs->shape(), axis);
    auto gradient = lhs->grad();
    auto incoming = out->grad();
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

void Max(const OpArgs &args) {
  auto axis = static_cast<size_t>(args.dim);
  AxisLayout layout = LayoutFor(args.lhs->shape(), axis);
  auto source = args.lhs->data();
  std::vector<size_t> indices = ArgmaxIndices(source.data(), layout);

  auto values = args.out->data();
  for (size_t o = 0; o < layout.outer; o++) {
    for (size_t i = 0; i < layout.inner; i++) {
      size_t slot = (o * layout.inner) + i;
      values[slot] =
          source[(((o * layout.reduced) + indices[slot]) * layout.inner) + i];
    }
  }
}

std::function<void()> MaxBackward(const GradArgs &args) {
  auto axis = static_cast<size_t>(args.dim);
  return [out = args.out, lhs = args.lhs, axis]() {
    AxisLayout layout = LayoutFor(lhs->shape(), axis);
    auto source = lhs->data();
    std::vector<size_t> indices = ArgmaxIndices(source.data(), layout);

    auto gradient = lhs->grad();
    auto incoming = out->grad();
    for (size_t o = 0; o < layout.outer; o++) {
      for (size_t i = 0; i < layout.inner; i++) {
        size_t slot = (o * layout.inner) + i;
        gradient[(((o * layout.reduced) + indices[slot]) * layout.inner) + i] +=
            incoming[slot];
      }
    }
  };
}

void Argmax(const OpArgs &args) {
  auto axis = static_cast<size_t>(args.dim);
  AxisLayout layout = LayoutFor(args.lhs->shape(), axis);
  auto source = args.lhs->data();
  std::vector<size_t> indices = ArgmaxIndices(source.data(), layout);

  auto values = args.out->data();
  for (size_t slot = 0; slot < indices.size(); slot++) {
    values[slot] = static_cast<scalar_t>(indices[slot]);
  }
}

}  // namespace

void RegisterReductionOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kSum, Device::CPU, Sum);
  registry.Register(OpId::kSumDim, Device::CPU, SumDim);
  registry.Register(OpId::kMax, Device::CPU, Max);
  registry.Register(OpId::kArgmax, Device::CPU, Argmax);
  registry.RegisterBackward(OpId::kSum, Device::CPU, SumBackward);
  registry.RegisterBackward(OpId::kSumDim, Device::CPU, SumDimBackward);
  registry.RegisterBackward(OpId::kMax, Device::CPU, MaxBackward);
}

}  // namespace micrograd::ops::cpu
