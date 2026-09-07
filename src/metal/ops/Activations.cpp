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

std::function<void()> ReluBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto &ctx = MetalContext::instance();
    size_t n = lhs->size();

    out->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(out->grad().data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer gradXBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "relu_backward", n)
        .buffer(gradOutBuf)
        .buffer(lhs->data_storage().buffer())
        .buffer(gradXBuf)
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

std::function<void()> MakeOutputBackward(const char *kernel,
                                         const GradArgs &args) {
  return [kernel, out = args.out, lhs = args.lhs]() {
    auto &ctx = MetalContext::instance();
    size_t n = lhs->size();

    out->to(Backend::CPU);
    ScopedBuffer gradOutBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(out->grad().data(), n,
                static_cast<scalar_t *>(gradOutBuf.get()->contents()));

    ScopedBuffer outBuf(ctx, n * sizeof(scalar_t));
    std::copy_n(out->data().data(), n,
                static_cast<scalar_t *>(outBuf.get()->contents()));

    ScopedBuffer gradXBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, kernel, n)
        .buffer(gradOutBuf)
        .buffer(outBuf)
        .buffer(gradXBuf)
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

std::function<void()> SigmoidBackward(const GradArgs &args) {
  return MakeOutputBackward("sigmoid_backward", args);
}

std::function<void()> TanhBackward(const GradArgs &args) {
  return MakeOutputBackward("tanh_backward", args);
}

}  // namespace

void RegisterActivationOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kRelu, Device::Metal, Relu);
  registry.Register(OpId::kSigmoid, Device::Metal, Sigmoid);
  registry.Register(OpId::kTanh, Device::Metal, Tanh);
  registry.RegisterBackward(OpId::kRelu, Device::Metal, ReluBackward);
  registry.RegisterBackward(OpId::kSigmoid, Device::Metal, SigmoidBackward);
  registry.RegisterBackward(OpId::kTanh, Device::Metal, TanhBackward);
}

}  // namespace micrograd::metal::ops

#endif
