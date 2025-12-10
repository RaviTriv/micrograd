#include "Foundation/NSError.hpp"
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "micrograd/metal/MetalContext.h"

#ifdef MICROGRAD_METAL_ENABLED

#include <iostream>

MetalContext &MetalContext::instance() {
  static MetalContext context;
  return context;
}

MetalContext::MetalContext() = default;

MetalContext::~MetalContext() { shutdown(); }

bool MetalContext::initialize() {
  if (initialized_) {
    return true;
  }

  device_ = MTL::CreateSystemDefaultDevice();
  if (!device_) {
    std::cerr << "NO GPU FOUND!!!!!" << std::endl;
    return false;
  }

  command_queue_ = device_->newCommandQueue();
  if (!command_queue_) {
    std::cerr << "Failed to create command queue." << std::endl;
    return false;
  }

  initialized_ = true;
  std::cout << device_->name()->utf8String() << " initialized to accelerate!"
            << std::endl;
  return true;
}


void MetalContext::synchronize() {
  MTL::CommandBuffer *command_buffer = command_queue_->commandBuffer();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

MTL::Device *MetalContext::device() { return device_; }
MTL::CommandQueue *MetalContext::commandQueue() { return command_queue_; }

#endif