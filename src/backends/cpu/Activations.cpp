#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cpu/Ops.h"
#include "micrograd/ops/Dispatch.h"

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

void Relu(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] <= 0 ? 0.0f : lhs[i];
  }
}

void ReluSquared(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    const scalar_t r = lhs[i] <= 0 ? 0.0f : lhs[i];
    out[i] = r * r;
  }
}

void Sigmoid(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    scalar_t z = std::exp(-std::fabs(lhs[i]));
    out[i] = lhs[i] >= 0 ? 1.0f / (1.0f + z) : z / (1.0f + z);
  }
}

void Tanh(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = std::tanh(lhs[i]);
  }
}

constexpr scalar_t kGeluSqrt2OverPi = 0.7978845608028654f;
constexpr scalar_t kGeluCubicCoeff = 0.044715f;

void Gelu(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    const scalar_t x = lhs[i];
    const scalar_t t =
        std::tanh(kGeluSqrt2OverPi * (x + (kGeluCubicCoeff * x * x * x)));
    out[i] = 0.5f * x * (1.0f + t);
  }
}

void Exp(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = std::exp(lhs[i]);
  }
}

void Log(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = std::log(lhs[i]);
  }
}

void Sqrt(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = std::sqrt(lhs[i]);
  }
}

void Neg(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = -lhs[i];
  }
}

std::function<void()> ReluBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto a_data = lhs->data();
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] * (a_data[i] > 0 ? 1.0f : 0.0f);
    }
  };
}

std::function<void()> ReluSquaredBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto a_data = lhs->data();
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] * (a_data[i] > 0 ? 2.0f * a_data[i] : 0.0f);
    }
  };
}

std::function<void()> SigmoidBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto out_data = out->data();
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      scalar_t sigmoid_val = out_data[i];
      a_grad[i] += out_grad[i] * sigmoid_val * (1.0f - sigmoid_val);
    }
  };
}

std::function<void()> TanhBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto out_data = out->data();
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      scalar_t tanh_val = out_data[i];
      a_grad[i] += out_grad[i] * (1.0f - tanh_val * tanh_val);
    }
  };
}

std::function<void()> GeluBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto in_data = lhs->data();
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      const scalar_t x = in_data[i];
      const scalar_t t =
          std::tanh(kGeluSqrt2OverPi * (x + (kGeluCubicCoeff * x * x * x)));
      const scalar_t du_dx =
          kGeluSqrt2OverPi * (1.0f + (3.0f * kGeluCubicCoeff * x * x));
      const scalar_t dy_dx =
          (0.5f * (1.0f + t)) + (0.5f * x * (1.0f - (t * t)) * du_dx);
      a_grad[i] += out_grad[i] * dy_dx;
    }
  };
}

std::function<void()> ExpBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto out_data = out->data();
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] * out_data[i];
    }
  };
}

std::function<void()> LogBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto a_data = lhs->data();
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] / a_data[i];
    }
  };
}

std::function<void()> SqrtBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto out_data = out->data();
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] * 0.5f / out_data[i];
    }
  };
}

std::function<void()> NegBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] -= out_grad[i];
    }
  };
}

void Softmax(const OpArgs &args) {
  auto axis = static_cast<size_t>(args.dim);
  AxisLayout layout = LayoutFor(args.lhs->shape(), axis);
  auto source = args.lhs->data();
  auto values = args.out->data();
  for (size_t o = 0; o < layout.outer; o++) {
    for (size_t i = 0; i < layout.inner; i++) {
      scalar_t largest = SliceMax(source.data(), layout, o, i);
      if (std::isinf(largest) && largest < 0) {
        largest = 0;
      }
      scalar_t total = 0;
      for (size_t k = 0; k < layout.reduced; k++) {
        size_t offset = AxisOffset(layout, o, k, i);
        values[offset] = std::exp(source[offset] - largest);
        total += values[offset];
      }
      if (total == 0) {
        total = 1;
      }
      for (size_t k = 0; k < layout.reduced; k++) {
        values[AxisOffset(layout, o, k, i)] /= total;
      }
    }
  }
}

std::function<void()> SoftmaxBackward(const GradArgs &args) {
  auto axis = static_cast<size_t>(args.dim);
  return [out = args.out, lhs = args.lhs, axis]() {
    AxisLayout layout = LayoutFor(lhs->shape(), axis);
    auto gradient = lhs->grad();
    auto outputs = out->data();
    auto incoming = out->grad();
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

void LogSoftmax(const OpArgs &args) {
  auto axis = static_cast<size_t>(args.dim);
  AxisLayout layout = LayoutFor(args.lhs->shape(), axis);
  auto source = args.lhs->data();
  auto values = args.out->data();
  for (size_t o = 0; o < layout.outer; o++) {
    for (size_t i = 0; i < layout.inner; i++) {
      scalar_t largest = SliceMax(source.data(), layout, o, i);
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
}

std::function<void()> LogSoftmaxBackward(const GradArgs &args) {
  auto axis = static_cast<size_t>(args.dim);
  return [out = args.out, lhs = args.lhs, axis]() {
    AxisLayout layout = LayoutFor(lhs->shape(), axis);
    auto gradient = lhs->grad();
    auto outputs = out->data();
    auto incoming = out->grad();
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

}  // namespace

void RegisterActivationOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kRelu, Device::CPU, Relu);
  registry.Register(OpId::kReluSquared, Device::CPU, ReluSquared);
  registry.Register(OpId::kSigmoid, Device::CPU, Sigmoid);
  registry.Register(OpId::kTanh, Device::CPU, Tanh);
  registry.Register(OpId::kGelu, Device::CPU, Gelu);
  registry.Register(OpId::kExp, Device::CPU, Exp);
  registry.Register(OpId::kLog, Device::CPU, Log);
  registry.Register(OpId::kSqrt, Device::CPU, Sqrt);
  registry.Register(OpId::kNeg, Device::CPU, Neg);
  registry.Register(OpId::kSoftmax, Device::CPU, Softmax);
  registry.Register(OpId::kLogSoftmax, Device::CPU, LogSoftmax);
  registry.RegisterBackward(OpId::kRelu, Device::CPU, ReluBackward);
  registry.RegisterBackward(OpId::kReluSquared, Device::CPU,
                            ReluSquaredBackward);
  registry.RegisterBackward(OpId::kSigmoid, Device::CPU, SigmoidBackward);
  registry.RegisterBackward(OpId::kTanh, Device::CPU, TanhBackward);
  registry.RegisterBackward(OpId::kGelu, Device::CPU, GeluBackward);
  registry.RegisterBackward(OpId::kExp, Device::CPU, ExpBackward);
  registry.RegisterBackward(OpId::kLog, Device::CPU, LogBackward);
  registry.RegisterBackward(OpId::kSqrt, Device::CPU, SqrtBackward);
  registry.RegisterBackward(OpId::kNeg, Device::CPU, NegBackward);
  registry.RegisterBackward(OpId::kSoftmax, Device::CPU, SoftmaxBackward);
  registry.RegisterBackward(OpId::kLogSoftmax, Device::CPU, LogSoftmaxBackward);
}

}  // namespace micrograd::ops::cpu
