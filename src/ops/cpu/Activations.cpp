#include <cmath>
#include <functional>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"
#include "micrograd/ops/cpu/Ops.h"

namespace micrograd::ops::cpu {
namespace {

void Relu(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = lhs[i] > 0 ? lhs[i] : 0.0f;
  }
}

void Sigmoid(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = 1.0f / (1.0f + std::exp(-lhs[i]));
  }
}

void Tanh(const OpArgs &args) {
  auto lhs = args.lhs->data();
  auto out = args.out->data();
  for (size_t i = 0; i < lhs.size(); i++) {
    out[i] = std::tanh(lhs[i]);
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

}  // namespace

void RegisterActivationOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kRelu, Device::CPU, Relu);
  registry.Register(OpId::kSigmoid, Device::CPU, Sigmoid);
  registry.Register(OpId::kTanh, Device::CPU, Tanh);
  registry.Register(OpId::kExp, Device::CPU, Exp);
  registry.Register(OpId::kLog, Device::CPU, Log);
  registry.Register(OpId::kSqrt, Device::CPU, Sqrt);
  registry.Register(OpId::kNeg, Device::CPU, Neg);
  registry.RegisterBackward(OpId::kRelu, Device::CPU, ReluBackward);
  registry.RegisterBackward(OpId::kSigmoid, Device::CPU, SigmoidBackward);
  registry.RegisterBackward(OpId::kTanh, Device::CPU, TanhBackward);
  registry.RegisterBackward(OpId::kExp, Device::CPU, ExpBackward);
  registry.RegisterBackward(OpId::kLog, Device::CPU, LogBackward);
  registry.RegisterBackward(OpId::kSqrt, Device::CPU, SqrtBackward);
  registry.RegisterBackward(OpId::kNeg, Device::CPU, NegBackward);
}

}  // namespace micrograd::ops::cpu
