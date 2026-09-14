#ifdef MICROGRAD_METAL_ENABLED

#include <functional>

#include "micrograd/Tensor.h"
#include "micrograd/metal/Dispatch.h"
#include "micrograd/metal/MetalContext.h"
#include "micrograd/metal/ops/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::metal::ops {
namespace {

void LaunchBinary(const char *kernel, const OpArgs &args) {
  args.out->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufSize.set(static_cast<uint32_t>(args.lhs->size()));

  ElementwiseKernelLauncher(ctx, kernel, args.lhs->size())
      .buffer(args.lhs->data_storage().buffer())
      .buffer(args.rhs->data_storage().buffer())
      .buffer(args.out->data_storage().buffer())
      .buffer(bufSize)
      .launch();
}

void LaunchScalar(const char *kernel, const OpArgs &args) {
  args.out->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  ScopedBuffer bufScalar(ctx, sizeof(scalar_t));
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufScalar.set(args.scalar);
  bufSize.set(static_cast<uint32_t>(args.lhs->size()));

  ElementwiseKernelLauncher(ctx, kernel, args.lhs->size())
      .buffer(args.lhs->data_storage().buffer())
      .buffer(args.out->data_storage().buffer())
      .buffer(bufScalar)
      .buffer(bufSize)
      .launch();
}

void Add(const OpArgs &args) { LaunchBinary("add", args); }
void Sub(const OpArgs &args) { LaunchBinary("sub", args); }
void Mul(const OpArgs &args) { LaunchBinary("mul", args); }
void Div(const OpArgs &args) { LaunchBinary("div_op", args); }

void AddScalar(const OpArgs &args) { LaunchScalar("add_scalar", args); }
void SubScalar(const OpArgs &args) { LaunchScalar("sub_scalar", args); }
void MulScalar(const OpArgs &args) { LaunchScalar("mul_scalar", args); }
void DivScalar(const OpArgs &args) { LaunchScalar("div_scalar", args); }
void Pow(const OpArgs &args) { LaunchScalar("pow_op", args); }

ScopedBuffer SizeBuffer(MetalContext &ctx, size_t n) {
  ScopedBuffer bufSize(ctx, sizeof(uint32_t));
  bufSize.set(static_cast<uint32_t>(n));
  return bufSize;
}

std::function<void()> MakeBinaryBackward(const char *kernel,
                                         const GradArgs &args) {
  return [kernel, out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    auto &ctx = MetalContext::instance();
    size_t n = lhs->size();

    ElementwiseKernelLauncher(ctx, kernel, n)
        .buffer(out->grad_storage().buffer())
        .buffer(lhs->grad_storage().buffer())
        .buffer(rhs->grad_storage().buffer())
        .buffer(SizeBuffer(ctx, n))
        .launch();
  };
}

std::function<void()> MakeBinaryDataBackward(const char *kernel,
                                             const GradArgs &args) {
  return [kernel, out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    auto &ctx = MetalContext::instance();
    size_t n = lhs->size();

    ElementwiseKernelLauncher(ctx, kernel, n)
        .buffer(out->grad_storage().buffer())
        .buffer(lhs->data_storage().buffer())
        .buffer(rhs->data_storage().buffer())
        .buffer(lhs->grad_storage().buffer())
        .buffer(rhs->grad_storage().buffer())
        .buffer(SizeBuffer(ctx, n))
        .launch();
  };
}

std::function<void()> MakeScaledBackward(scalar_t scale, const GradArgs &args) {
  return [scale, out = args.out, lhs = args.lhs]() {
    auto &ctx = MetalContext::instance();
    size_t n = lhs->size();

    ScopedBuffer bufScale(ctx, sizeof(scalar_t));
    bufScale.set(scale);

    ElementwiseKernelLauncher(ctx, "accumulate_scaled", n)
        .buffer(out->grad_storage().buffer())
        .buffer(lhs->grad_storage().buffer())
        .buffer(bufScale)
        .buffer(SizeBuffer(ctx, n))
        .launch();
  };
}

std::function<void()> AddBackward(const GradArgs &args) {
  return MakeBinaryBackward("add_backward", args);
}

std::function<void()> SubBackward(const GradArgs &args) {
  return MakeBinaryBackward("sub_backward", args);
}

std::function<void()> MulBackward(const GradArgs &args) {
  return MakeBinaryDataBackward("mul_backward", args);
}

std::function<void()> DivBackward(const GradArgs &args) {
  return MakeBinaryDataBackward("div_backward", args);
}

std::function<void()> AddScalarBackward(const GradArgs &args) {
  return MakeScaledBackward(1.0f, args);
}

std::function<void()> SubScalarBackward(const GradArgs &args) {
  return MakeScaledBackward(1.0f, args);
}

std::function<void()> MulScalarBackward(const GradArgs &args) {
  return MakeScaledBackward(args.scalar, args);
}

std::function<void()> DivScalarBackward(const GradArgs &args) {
  return MakeScaledBackward(1.0f / args.scalar, args);
}

std::function<void()> PowBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, exponent = args.scalar]() {
    auto &ctx = MetalContext::instance();
    size_t n = lhs->size();

    ScopedBuffer bufExp(ctx, sizeof(scalar_t));
    bufExp.set(exponent);

    ElementwiseKernelLauncher(ctx, "pow_backward", n)
        .buffer(out->grad_storage().buffer())
        .buffer(lhs->data_storage().buffer())
        .buffer(lhs->grad_storage().buffer())
        .buffer(bufExp)
        .buffer(SizeBuffer(ctx, n))
        .launch();
  };
}

}  // namespace

void RegisterArithmeticOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kAdd, Device::Metal, Add);
  registry.Register(OpId::kSub, Device::Metal, Sub);
  registry.Register(OpId::kMul, Device::Metal, Mul);
  registry.Register(OpId::kDiv, Device::Metal, Div);
  registry.Register(OpId::kAddScalar, Device::Metal, AddScalar);
  registry.Register(OpId::kSubScalar, Device::Metal, SubScalar);
  registry.Register(OpId::kMulScalar, Device::Metal, MulScalar);
  registry.Register(OpId::kDivScalar, Device::Metal, DivScalar);
  registry.Register(OpId::kPow, Device::Metal, Pow);
  registry.RegisterBackward(OpId::kAdd, Device::Metal, AddBackward);
  registry.RegisterBackward(OpId::kSub, Device::Metal, SubBackward);
  registry.RegisterBackward(OpId::kMul, Device::Metal, MulBackward);
  registry.RegisterBackward(OpId::kDiv, Device::Metal, DivBackward);
  registry.RegisterBackward(OpId::kAddScalar, Device::Metal, AddScalarBackward);
  registry.RegisterBackward(OpId::kSubScalar, Device::Metal, SubScalarBackward);
  registry.RegisterBackward(OpId::kMulScalar, Device::Metal, MulScalarBackward);
  registry.RegisterBackward(OpId::kDivScalar, Device::Metal, DivScalarBackward);
  registry.RegisterBackward(OpId::kPow, Device::Metal, PowBackward);
}

}  // namespace micrograd::metal::ops

#endif
