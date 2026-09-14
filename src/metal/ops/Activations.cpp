#ifdef MICROGRAD_METAL_ENABLED

#include <functional>

#include "micrograd/Tensor.h"
#include "micrograd/metal/Dispatch.h"
#include "micrograd/metal/MetalContext.h"
#include "micrograd/metal/ops/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::metal::ops {
namespace {

void LaunchUnary(const char *kernel, const OpArgs &args) {
  args.out->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufSize.set(static_cast<uint32_t>(args.lhs->size()));

  ElementwiseKernelLauncher(ctx, kernel, args.lhs->size())
      .buffer(args.lhs->data_storage().buffer())
      .buffer(args.out->data_storage().buffer())
      .buffer(bufSize)
      .launch();
}

void Relu(const OpArgs &args) { LaunchUnary("relu", args); }
void Sigmoid(const OpArgs &args) { LaunchUnary("sigmoid", args); }
void Tanh(const OpArgs &args) { LaunchUnary("tanh_op", args); }
void Exp(const OpArgs &args) { LaunchUnary("exp_op", args); }
void Log(const OpArgs &args) { LaunchUnary("log_op", args); }
void Sqrt(const OpArgs &args) { LaunchUnary("sqrt_op", args); }
void Neg(const OpArgs &args) { LaunchUnary("neg_op", args); }

void LaunchUnaryBackward(const char *kernel, const Storage &source,
                         const Tensor &out, Tensor &input) {
  auto &ctx = MetalContext::instance();
  size_t n = input.size();

  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufSize.set(static_cast<uint32_t>(n));

  ElementwiseKernelLauncher(ctx, kernel, n)
      .buffer(out.grad_storage().buffer())
      .buffer(source.buffer())
      .buffer(input.grad_storage().buffer())
      .buffer(bufSize)
      .launch();
}

std::function<void()> MakeInputBackward(const char *kernel,
                                        const GradArgs &args) {
  return [kernel, out = args.out, lhs = args.lhs]() {
    LaunchUnaryBackward(kernel, lhs->data_storage(), *out, *lhs);
  };
}

std::function<void()> MakeOutputBackward(const char *kernel,
                                         const GradArgs &args) {
  return [kernel, out = args.out, lhs = args.lhs]() {
    LaunchUnaryBackward(kernel, out->data_storage(), *out, *lhs);
  };
}

std::function<void()> ReluBackward(const GradArgs &args) {
  return MakeInputBackward("relu_backward", args);
}

std::function<void()> SigmoidBackward(const GradArgs &args) {
  return MakeOutputBackward("sigmoid_backward", args);
}

std::function<void()> TanhBackward(const GradArgs &args) {
  return MakeOutputBackward("tanh_backward", args);
}

std::function<void()> ExpBackward(const GradArgs &args) {
  return MakeOutputBackward("exp_backward", args);
}

std::function<void()> LogBackward(const GradArgs &args) {
  return MakeInputBackward("log_backward", args);
}

std::function<void()> SqrtBackward(const GradArgs &args) {
  return MakeOutputBackward("sqrt_backward", args);
}

std::function<void()> NegBackward(const GradArgs &args) {
  return MakeInputBackward("neg_backward", args);
}

}  // namespace

void RegisterActivationOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kRelu, Device::Metal, Relu);
  registry.Register(OpId::kSigmoid, Device::Metal, Sigmoid);
  registry.Register(OpId::kTanh, Device::Metal, Tanh);
  registry.Register(OpId::kExp, Device::Metal, Exp);
  registry.Register(OpId::kLog, Device::Metal, Log);
  registry.Register(OpId::kSqrt, Device::Metal, Sqrt);
  registry.Register(OpId::kNeg, Device::Metal, Neg);
  registry.RegisterBackward(OpId::kRelu, Device::Metal, ReluBackward);
  registry.RegisterBackward(OpId::kSigmoid, Device::Metal, SigmoidBackward);
  registry.RegisterBackward(OpId::kTanh, Device::Metal, TanhBackward);
  registry.RegisterBackward(OpId::kExp, Device::Metal, ExpBackward);
  registry.RegisterBackward(OpId::kLog, Device::Metal, LogBackward);
  registry.RegisterBackward(OpId::kSqrt, Device::Metal, SqrtBackward);
  registry.RegisterBackward(OpId::kNeg, Device::Metal, NegBackward);
}

}  // namespace micrograd::metal::ops

#endif
