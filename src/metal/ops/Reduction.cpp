#ifdef MICROGRAD_METAL_ENABLED

#include <functional>

#include "micrograd/Tensor.h"
#include "micrograd/metal/Dispatch.h"
#include "micrograd/metal/MetalContext.h"
#include "micrograd/metal/ops/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::metal::ops {
namespace {

void Sum(const OpArgs &args) {
  auto &ctx = MetalContext::instance();
  auto pipeline = ctx.getPipeline("sum_reduce");

  const uint32_t threadgroupSize = 256;
  auto currentSize = static_cast<uint32_t>(args.lhs->size());
  uint32_t numThreadgroups =
      (currentSize + threadgroupSize - 1) / threadgroupSize;

  MTL::Buffer *sourceBuf = args.lhs->data_storage().buffer();
  MTL::Buffer *inputBuf = sourceBuf;
  MTL::Buffer *outputBuf = ctx.createBuffer(numThreadgroups * sizeof(scalar_t));

  while (currentSize > 1) {
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(currentSize);

    auto cmdBuf = ctx.commandQueue()->commandBuffer();
    auto encoder = cmdBuf->computeCommandEncoder();

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuf, 0, 0);
    encoder->setBuffer(outputBuf, 0, 1);
    encoder->setBuffer(bufSize, 0, 2);

    MTL::Size numGroups(numThreadgroups, 1, 1);
    MTL::Size tgSize(threadgroupSize, 1, 1);
    encoder->dispatchThreadgroups(numGroups, tgSize);

    encoder->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    currentSize = numThreadgroups;
    numThreadgroups = (currentSize + threadgroupSize - 1) / threadgroupSize;

    if (currentSize > 1) {
      if (inputBuf != sourceBuf) {
        ctx.releaseBuffer(inputBuf);
      }
      inputBuf = outputBuf;
      outputBuf = ctx.createBuffer(numThreadgroups * sizeof(scalar_t));
    }
  }

  args.out->data()[0] = *static_cast<scalar_t *>(outputBuf->contents());

  if (inputBuf != sourceBuf) {
    ctx.releaseBuffer(inputBuf);
  }
  ctx.releaseBuffer(outputBuf);
}

std::function<void()> SumBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto &ctx = MetalContext::instance();
    size_t n = lhs->size();

    scalar_t gradScalar = out->grad()[0];

    ScopedBuffer gradXBuf(ctx, n * sizeof(scalar_t));
    ScopedBuffer bufScalar(ctx, sizeof(scalar_t));
    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufScalar.set(gradScalar);
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "broadcast_scalar", n)
        .buffer(gradXBuf)
        .buffer(bufScalar)
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

void RegisterReductionOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kSum, Device::Metal, Sum);
  registry.RegisterBackward(OpId::kSum, Device::Metal, SumBackward);
}

}  // namespace micrograd::metal::ops

#endif
