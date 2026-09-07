#ifdef MICROGRAD_METAL_ENABLED

#include <algorithm>
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

std::function<void()> AddBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    lhs->to(Backend::CPU);
    rhs->to(Backend::CPU);
    out->to(Backend::CPU);

    auto a_grad = lhs->grad();
    auto b_grad = rhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
      b_grad[i] += out_grad[i];
    }
  };
}

std::function<void()> SubBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    lhs->to(Backend::CPU);
    rhs->to(Backend::CPU);
    out->to(Backend::CPU);

    auto a_grad = lhs->grad();
    auto b_grad = rhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
      b_grad[i] -= out_grad[i];
    }
  };
}

std::function<void()> MakeBinaryBackward(const char *kernel,
                                         const GradArgs &args) {
  return [kernel, out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    auto &ctx = MetalContext::instance();
    size_t n = lhs->size();

    out->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(out->grad().data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer gradABuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer gradBBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, kernel, n)
        .buffer(gradOutBuf)
        .buffer(lhs->data_storage().buffer())
        .buffer(rhs->data_storage().buffer())
        .buffer(gradABuf)
        .buffer(gradBBuf)
        .buffer(bufSize)
        .launch();

    auto *gradAPtr = static_cast<scalar_t *>(gradABuf.get()->contents());
    auto *gradBPtr = static_cast<scalar_t *>(gradBBuf.get()->contents());
    auto *gpuGradAPtr =
        static_cast<scalar_t *>(lhs->grad_storage().host_pointer());
    auto *gpuGradBPtr =
        static_cast<scalar_t *>(rhs->grad_storage().host_pointer());
    for (size_t i = 0; i < n; i++) {
      gpuGradAPtr[i] += gradAPtr[i];
      gpuGradBPtr[i] += gradBPtr[i];
    }
  };
}

std::function<void()> MulBackward(const GradArgs &args) {
  return MakeBinaryBackward("mul_backward", args);
}

std::function<void()> DivBackward(const GradArgs &args) {
  return MakeBinaryBackward("div_backward", args);
}

std::function<void()> AddScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    lhs->to(Backend::CPU);
    out->to(Backend::CPU);

    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
    }
  };
}

std::function<void()> SubScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    lhs->to(Backend::CPU);
    out->to(Backend::CPU);

    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i];
    }
  };
}

std::function<void()> MulScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, scalar = args.scalar]() {
    lhs->to(Backend::CPU);
    out->to(Backend::CPU);

    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] * scalar;
    }
  };
}

std::function<void()> DivScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, scalar = args.scalar]() {
    lhs->to(Backend::CPU);
    out->to(Backend::CPU);

    auto a_grad = lhs->grad();
    auto out_grad = out->grad();
    for (size_t i = 0; i < a_grad.size(); i++) {
      a_grad[i] += out_grad[i] / scalar;
    }
  };
}

std::function<void()> PowBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, exponent = args.scalar]() {
    auto &ctx = MetalContext::instance();
    size_t n = lhs->size();

    out->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(out->grad().data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer gradXBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufExp(ctx, sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufExp.set(exponent);
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "pow_backward", n)
        .buffer(gradOutBuf)
        .buffer(lhs->data_storage().buffer())
        .buffer(gradXBuf)
        .buffer(bufExp)
        .buffer(bufSize)
        .launch();

    auto *gradXPtr = static_cast<scalar_t *>(gradXBuf.get()->contents());
    auto *gpuGradPtr =
        static_cast<scalar_t *>(lhs->grad_storage().host_pointer());
    for (size_t i = 0; i < n; i++) {
      gpuGradPtr[i] += gradXPtr[i];
    }
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
