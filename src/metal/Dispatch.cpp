#ifdef MICROGRAD_METAL_ENABLED

#include "micrograd/metal/Dispatch.h"
#include <algorithm>

ElementwiseKernelLauncher::ElementwiseKernelLauncher(MetalContext &ctx,
                                                     const std::string &kernel, size_t size)
    : ctx_(ctx), size_(size) {
  pipeline_ = ctx_.getPipeline(kernel);
  cmdBuf_ = ctx_.commandQueue()->commandBuffer();
  encoder_ = cmdBuf_->computeCommandEncoder();
  encoder_->setComputePipelineState(pipeline_);
}

ElementwiseKernelLauncher &ElementwiseKernelLauncher::buffer(MTL::Buffer *buf) {
  encoder_->setBuffer(buf, 0, bufferIndex_++);
  return *this;
}

ElementwiseKernelLauncher &ElementwiseKernelLauncher::buffer(const ScopedBuffer &buf) {
  encoder_->setBuffer(buf.get(), 0, bufferIndex_++);
  return *this;
}

void ElementwiseKernelLauncher::launch() {
  constexpr size_t kThreadgroupSize = 256;
  MTL::Size gridSize(size_, 1, 1);
  MTL::Size threadGroupSize(std::min(size_, kThreadgroupSize), 1, 1);
  encoder_->dispatchThreads(gridSize, threadGroupSize);

  encoder_->endEncoding();
  cmdBuf_->commit();
  cmdBuf_->waitUntilCompleted();
}


MatmulKernelLauncher::MatmulKernelLauncher(MetalContext &ctx, const std::string &kernel,
                                           size_t m, size_t k, size_t n)
    : ctx_(ctx), m_(m), k_(k), n_(n), bufM_(ctx, sizeof(uint32_t)),
      bufK_(ctx, sizeof(uint32_t)), bufN_(ctx, sizeof(uint32_t)) {
  pipeline_ = ctx_.getPipeline(kernel);
  cmdBuf_ = ctx_.commandQueue()->commandBuffer();
  encoder_ = cmdBuf_->computeCommandEncoder();
  encoder_->setComputePipelineState(pipeline_);

  bufM_.set(static_cast<uint32_t>(m));
  bufK_.set(static_cast<uint32_t>(k));
  bufN_.set(static_cast<uint32_t>(n));
}

MatmulKernelLauncher &MatmulKernelLauncher::A(MTL::Buffer *buf) {
  encoder_->setBuffer(buf, 0, 0);
  return *this;
}

MatmulKernelLauncher &MatmulKernelLauncher::B(MTL::Buffer *buf) {
  encoder_->setBuffer(buf, 0, 1);
  return *this;
}

MatmulKernelLauncher &MatmulKernelLauncher::C(MTL::Buffer *buf) {
  encoder_->setBuffer(buf, 0, 2);
  encoder_->setBuffer(bufM_, 0, 3);
  encoder_->setBuffer(bufK_, 0, 4);
  encoder_->setBuffer(bufN_, 0, 5);
  return *this;
}

void MatmulKernelLauncher::launch() {
  constexpr size_t kTileSize = 16;
  MTL::Size gridSize(n_, m_, 1);
  MTL::Size threadGroupSize(std::min(n_, kTileSize), std::min(m_, kTileSize), 1);
  encoder_->dispatchThreads(gridSize, threadGroupSize);

  encoder_->endEncoding();
  cmdBuf_->commit();
  cmdBuf_->waitUntilCompleted();
}

#endif
