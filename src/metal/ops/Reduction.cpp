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
  args.out->to(Backend::Metal);

  auto &ctx = MetalContext::instance();
  auto pipeline = ctx.getPipeline("sum_reduce");

  const uint32_t threadgroupSize = 256;
  auto currentSize = static_cast<uint32_t>(args.lhs->size());
  uint32_t numThreadgroups =
      (currentSize + threadgroupSize - 1) / threadgroupSize;

  MTL::Buffer *sourceBuf = args.lhs->data_storage().buffer();
  MTL::Buffer *resultBuf = args.out->data_storage().buffer();
  MTL::Buffer *inputBuf = sourceBuf;
  MTL::Buffer *outputBuf =
      numThreadgroups == 1
          ? resultBuf
          : ctx.createBuffer(numThreadgroups * sizeof(scalar_t));

  // Each pass folds `currentSize` values into one partial sum per threadgroup;
  // the last pass has a single threadgroup and writes straight into the output.
  do {
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

    if (inputBuf != sourceBuf) {
      ctx.releaseBuffer(inputBuf);
    }
    inputBuf = outputBuf;
    currentSize = numThreadgroups;
    numThreadgroups = (currentSize + threadgroupSize - 1) / threadgroupSize;
    outputBuf = numThreadgroups == 1
                    ? resultBuf
                    : ctx.createBuffer(numThreadgroups * sizeof(scalar_t));
  } while (inputBuf != resultBuf);
}

std::function<void()> SumBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    auto &ctx = MetalContext::instance();
    size_t n = lhs->size();

    ScopedBuffer bufSize(ctx, sizeof(uint32_t));
    bufSize.set(static_cast<uint32_t>(n));

    ElementwiseKernelLauncher(ctx, "accumulate_broadcast", n)
        .buffer(out->grad_storage().buffer())
        .buffer(lhs->grad_storage().buffer())
        .buffer(bufSize)
        .launch();
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
