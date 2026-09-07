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

void Matmul(const OpArgs &args) {
  const size_t m = args.lhs->shape()[0];
  const size_t k = args.lhs->shape()[1];
  const size_t n = args.rhs->shape()[1];

  args.out->to(Backend::Metal);

  auto &ctx = MetalContext::instance();

  MatmulKernelLauncher(ctx, "matmul", m, k, n)
      .A(args.lhs->data_storage().buffer())
      .B(args.rhs->data_storage().buffer())
      .C(args.out->data_storage().buffer())
      .launch();
}

std::function<void()> MatmulBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    const size_t m = lhs->shape()[0];
    const size_t k = lhs->shape()[1];
    const size_t n = rhs->shape()[1];

    auto &ctx = MetalContext::instance();

    out->to(Backend::CPU);

    ScopedBuffer gradCBuf(ctx, m * n * sizeof(scalar_t));
    std::copy_n(out->grad().data(), m * n,
                static_cast<scalar_t *>(gradCBuf.get()->contents()));

    ScopedBuffer gradABuf(ctx, m * k * sizeof(scalar_t));
    MatmulKernelLauncher(ctx, "matmul_nt", m, n, k)
        .A(gradCBuf.get())
        .B(rhs->data_storage().buffer())
        .C(gradABuf.get())
        .launch();

    ScopedBuffer gradBBuf(ctx, k * n * sizeof(scalar_t));
    MatmulKernelLauncher(ctx, "matmul_tn", m, k, n, k)
        .A(lhs->data_storage().buffer())
        .B(gradCBuf.get())
        .C(gradBBuf.get())
        .launch();

    auto *gradAPtr = static_cast<scalar_t *>(gradABuf.get()->contents());
    auto *gradBPtr = static_cast<scalar_t *>(gradBBuf.get()->contents());
    auto *gpuGradAPtr =
        static_cast<scalar_t *>(lhs->grad_storage().host_pointer());
    auto *gpuGradBPtr =
        static_cast<scalar_t *>(rhs->grad_storage().host_pointer());

    for (size_t i = 0; i < m * k; i++) {
      gpuGradAPtr[i] += gradAPtr[i];
    }
    for (size_t i = 0; i < k * n; i++) {
      gpuGradBPtr[i] += gradBPtr[i];
    }
  };
}

}  // namespace

void RegisterMatmulOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kMatmul, Device::Metal, Matmul);
  registry.RegisterBackward(OpId::kMatmul, Device::Metal, MatmulBackward);
}

}  // namespace micrograd::metal::ops

#endif
