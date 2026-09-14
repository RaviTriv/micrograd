#ifdef MICROGRAD_METAL_ENABLED

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
    MTL::Buffer *gradC = out->grad_storage().buffer();

    MatmulKernelLauncher(ctx, "matmul_nt", m, n, k)
        .A(gradC)
        .B(rhs->data_storage().buffer())
        .C(lhs->grad_storage().buffer())
        .launch();

    MatmulKernelLauncher(ctx, "matmul_tn", m, k, n, k)
        .A(lhs->data_storage().buffer())
        .B(gradC)
        .C(rhs->grad_storage().buffer())
        .launch();
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
