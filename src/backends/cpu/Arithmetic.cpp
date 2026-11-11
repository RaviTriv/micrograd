#include <cmath>
#include <functional>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cpu/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::ops::cpu {
namespace {

template <typename Op>
void Elementwise(const OpArgs &args, Op op) {
  auto lhs = args.lhs->data();
  auto rhs = args.rhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = op(lhs[i], rhs[i]);
  }
}

template <typename Op>
void ElementwiseScalar(const OpArgs &args, Op op) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  const scalar_t scalar = args.scalar;
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = op(lhs[i], scalar);
  }
}

void Add(const OpArgs &args) { Elementwise(args, std::plus<>{}); }

void Sub(const OpArgs &args) { Elementwise(args, std::minus<>{}); }

void Mul(const OpArgs &args) { Elementwise(args, std::multiplies<>{}); }

void Div(const OpArgs &args) { Elementwise(args, std::divides<>{}); }

void AddScalar(const OpArgs &args) { ElementwiseScalar(args, std::plus<>{}); }

void SubScalar(const OpArgs &args) { ElementwiseScalar(args, std::minus<>{}); }

void MulScalar(const OpArgs &args) {
  ElementwiseScalar(args, std::multiplies<>{});
}

void DivScalar(const OpArgs &args) {
  ElementwiseScalar(args, std::divides<>{});
}

void Pow(const OpArgs &args) {
  ElementwiseScalar(args, [](scalar_t x, scalar_t exponent) {
    return std::pow(x, exponent);
  });
}

void ClampMin(const OpArgs &args) {
  ElementwiseScalar(args, [](scalar_t x, scalar_t bound) {
    return x < bound ? bound : x;
  });
}

void ClampMax(const OpArgs &args) {
  ElementwiseScalar(args, [](scalar_t x, scalar_t bound) {
    return x > bound ? bound : x;
  });
}

std::function<void()> AddBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    auto a_grad = lhs->grad();
    auto b_grad = rhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
      b_grad[i] += out_grad[i];
    }
  };
}

void AddBias(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto bias = args.rhs->data();
  auto out = args.out->data();
  const size_t cols = bias.size();
  const size_t rows = cols == 0 ? 0 : lhs.size() / cols;
  for (size_t r = 0; r < rows; r++) {
    const size_t row = r * cols;
    for (size_t c = 0; c < cols; c++) {
      out[row + c] = lhs[row + c] + bias[c];
    }
  }
}

std::function<void()> AddBiasBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    auto a_grad = lhs->grad();
    auto b_grad = rhs->grad();
    auto out_grad = out->grad();
    const size_t cols = b_grad.size();
    const size_t rows = cols == 0 ? 0 : a_grad.size() / cols;
    for (size_t r = 0; r < rows; r++) {
      const size_t row = r * cols;
      for (size_t c = 0; c < cols; c++) {
        a_grad[row + c] += out_grad[row + c];
        b_grad[c] += out_grad[row + c];
      }
    }
  };
}

std::function<void()> SubBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    auto a_grad = lhs->grad();
    auto b_grad = rhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
      b_grad[i] -= out_grad[i];
    }
  };
}

std::function<void()> MulBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    auto a_data = lhs->data();
    auto b_data = rhs->data();
    auto a_grad = lhs->grad();
    auto b_grad = rhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] * b_data[i];
      b_grad[i] += out_grad[i] * a_data[i];
    }
  };
}

std::function<void()> DivBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    auto a_data = lhs->data();
    auto b_data = rhs->data();
    auto a_grad = lhs->grad();
    auto b_grad = rhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] / b_data[i];
      b_grad[i] -= out_grad[i] * a_data[i] / b_data[i] / b_data[i];
    }
  };
}

std::function<void()> AddScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
    }
  };
}

std::function<void()> SubScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
    }
  };
}

std::function<void()> MulScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, scalar = args.scalar]() {
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] * scalar;
    }
  };
}

std::function<void()> DivScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, scalar = args.scalar]() {
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] / scalar;
    }
  };
}

std::function<void()> PowBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, exponent = args.scalar]() {
    if (exponent == 0.0f) {
      return;
    }
    auto a_data = lhs->data();
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] +=
          out_grad[i] * exponent * std::pow(a_data[i], exponent - 1.0f);
    }
  };
}

std::function<void()> ClampMinBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, bound = args.scalar]() {
    auto a_data = lhs->data();
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      if (a_data[i] >= bound) {
        a_grad[i] += out_grad[i];
      }
    }
  };
}

std::function<void()> ClampMaxBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, bound = args.scalar]() {
    auto a_data = lhs->data();
    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      if (a_data[i] <= bound) {
        a_grad[i] += out_grad[i];
      }
    }
  };
}

}  // namespace

void RegisterArithmeticOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kAdd, Device::CPU, Add);
  registry.Register(OpId::kAddBias, Device::CPU, AddBias);
  registry.Register(OpId::kSub, Device::CPU, Sub);
  registry.Register(OpId::kMul, Device::CPU, Mul);
  registry.Register(OpId::kDiv, Device::CPU, Div);
  registry.Register(OpId::kAddScalar, Device::CPU, AddScalar);
  registry.Register(OpId::kSubScalar, Device::CPU, SubScalar);
  registry.Register(OpId::kMulScalar, Device::CPU, MulScalar);
  registry.Register(OpId::kDivScalar, Device::CPU, DivScalar);
  registry.Register(OpId::kPow, Device::CPU, Pow);
  registry.Register(OpId::kClampMin, Device::CPU, ClampMin);
  registry.Register(OpId::kClampMax, Device::CPU, ClampMax);
  registry.RegisterBackward(OpId::kAdd, Device::CPU, AddBackward);
  registry.RegisterBackward(OpId::kAddBias, Device::CPU, AddBiasBackward);
  registry.RegisterBackward(OpId::kSub, Device::CPU, SubBackward);
  registry.RegisterBackward(OpId::kMul, Device::CPU, MulBackward);
  registry.RegisterBackward(OpId::kDiv, Device::CPU, DivBackward);
  registry.RegisterBackward(OpId::kAddScalar, Device::CPU, AddScalarBackward);
  registry.RegisterBackward(OpId::kSubScalar, Device::CPU, SubScalarBackward);
  registry.RegisterBackward(OpId::kMulScalar, Device::CPU, MulScalarBackward);
  registry.RegisterBackward(OpId::kDivScalar, Device::CPU, DivScalarBackward);
  registry.RegisterBackward(OpId::kPow, Device::CPU, PowBackward);
  registry.RegisterBackward(OpId::kClampMin, Device::CPU, ClampMinBackward);
  registry.RegisterBackward(OpId::kClampMax, Device::CPU, ClampMaxBackward);
}

}  // namespace micrograd::ops::cpu
