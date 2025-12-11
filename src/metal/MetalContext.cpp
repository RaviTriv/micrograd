#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "micrograd/metal/MetalContext.h"
#include "micrograd/metal/Kernels.h"
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

  std::cout << device_->name()->utf8String() << " initialized to accelerate!"
            << std::endl;

  NS::Error *error = nullptr;
  NS::String *source = NS::String::string(micrograd::metal::KERNEL_SOURCE,
                                          NS::UTF8StringEncoding);
  library_ = device_->newLibrary(source, nullptr, &error);
  if (!library_) {
    std::cerr << "FAILED TO INIT KERNEL :(" << std::endl;
    if (error) {
      std::cerr << "Error: " << error->localizedDescription()->utf8String()
                << std::endl;
    }
    return false;
  }
  initialized_ = true;

  return true;
}

void MetalContext::synchronize() {
  MTL::CommandBuffer *command_buffer = command_queue_->commandBuffer();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

MTL::Buffer *MetalContext::createBuffer(size_t bytes) {
  return device_->newBuffer(bytes, MTL::ResourceStorageModeShared);
}

void MetalContext::releaseBuffer(MTL::Buffer *buffer) {
  if (buffer) {
    buffer->release();
  }
}

MTL::ComputePipelineState *MetalContext::getPipeline(const std::string &name) {
  auto it = pipelines_.find(name);

  if (it != pipelines_.end()) {
    return it->second;
  }

  if (!library_) {
    return nullptr;
  }

  NS::String *fnName = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
  MTL::Function *fn = library_->newFunction(fnName);

  if (!fn) {
    std::cerr << "Failed to create function: " << name << std::endl;
    return nullptr;
  }

  NS::Error *error = nullptr;
  MTL::ComputePipelineState *pipeline =
      device_->newComputePipelineState(fn, &error);
  fn->release();

  if (!pipeline) {
    std::cerr << "Failed to create pipeline for function: " << name << " :("
              << std::endl;
  }

  pipelines_[name] = pipeline;
  return pipeline;
}

void MetalContext::shutdown() {
  for (auto &[name, pipeline] : pipelines_) {
    pipeline->release();
  }
  pipelines_.clear();

  if (command_queue_) {
    command_queue_->release();
  }

  if (device_) {
    device_->release();
  }

  if (library_) {
    library_->release();
  }

  command_queue_ = nullptr;
  device_ = nullptr;
  library_ = nullptr;
  initialized_ = false;
}

bool MetalContext::isAvailable() const {
  return initialized_ && device_ != nullptr;
}

MTL::Device *MetalContext::device() { return device_; }
MTL::CommandQueue *MetalContext::commandQueue() { return command_queue_; }

#endif