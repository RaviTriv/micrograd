#pragma once

#ifdef MICROGRAD_METAL_ENABLED

#include "micrograd/metal/MetalContext.h"
#include <Metal/Metal.hpp>
#include <string>
#include <vector>

class ElementwiseKernelLauncher {
public:
  ElementwiseKernelLauncher(MetalContext &ctx, const std::string &kernel, size_t size);

  ElementwiseKernelLauncher &buffer(MTL::Buffer *buf);
  ElementwiseKernelLauncher &buffer(const ScopedBuffer &buf);

  void launch();

private:
  MetalContext &ctx_;
  MTL::ComputePipelineState *pipeline_;
  MTL::CommandBuffer *cmdBuf_;
  MTL::ComputeCommandEncoder *encoder_;
  size_t size_;
  uint32_t bufferIndex_ = 0;
};

class MatmulKernelLauncher {
public:
  MatmulKernelLauncher(MetalContext &ctx, const std::string &kernel,
                       size_t m, size_t k, size_t n);

  MatmulKernelLauncher &A(MTL::Buffer *buf);
  MatmulKernelLauncher &B(MTL::Buffer *buf);
  MatmulKernelLauncher &C(MTL::Buffer *buf);

  void launch();

private:
  MetalContext &ctx_;
  MTL::ComputePipelineState *pipeline_;
  MTL::CommandBuffer *cmdBuf_;
  MTL::ComputeCommandEncoder *encoder_;
  size_t m_, k_, n_;
  ScopedBuffer bufM_, bufK_, bufN_;
};

#endif
