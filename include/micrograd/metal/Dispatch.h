#pragma once

#ifdef MICROGRAD_METAL_ENABLED

#include <Metal/Metal.hpp>
#include <string>
#include <vector>

#include "micrograd/metal/MetalContext.h"

class ElementwiseKernelLauncher {
 public:
  ElementwiseKernelLauncher(MetalContext &ctx, const std::string &kernel,
                            size_t size);

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
  MatmulKernelLauncher(MetalContext &ctx, const std::string &kernel, size_t m,
                       size_t k, size_t n, size_t output_rows = 0);

  MatmulKernelLauncher &A(MTL::Buffer *buf);
  MatmulKernelLauncher &B(MTL::Buffer *buf);
  MatmulKernelLauncher &C(MTL::Buffer *buf);

  void launch();

 private:
  MetalContext &ctx_;
  MTL::ComputePipelineState *pipeline_;
  MTL::CommandBuffer *cmdBuf_;
  MTL::ComputeCommandEncoder *encoder_;
  size_t rows_, n_;
  ScopedBuffer bufM_, bufK_, bufN_;
};

#endif
