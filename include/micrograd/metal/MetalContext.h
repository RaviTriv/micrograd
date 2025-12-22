#pragma once

#ifdef MICROGRAD_METAL_ENABLED

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <string>
#include <unordered_map>

class MetalContext;

class ScopedBuffer {
public:
  ScopedBuffer(MetalContext &ctx, size_t bytes);
  ~ScopedBuffer();

  ScopedBuffer(const ScopedBuffer &) = delete;
  ScopedBuffer &operator=(const ScopedBuffer &) = delete;
  ScopedBuffer(ScopedBuffer &&other) noexcept;
  ScopedBuffer &operator=(ScopedBuffer &&other) noexcept;

  template <typename T> void set(T value) {
    *static_cast<T *>(buffer_->contents()) = value;
  }

  MTL::Buffer *get() const { return buffer_; }
  operator MTL::Buffer *() const { return buffer_; }

private:
  MetalContext *ctx_;
  MTL::Buffer *buffer_;
};

class MetalContext {
public:
  static MetalContext &instance();

  bool initialize();
  void shutdown();
  bool isAvailable() const;

  MTL::Buffer *createBuffer(size_t bytes);
  void releaseBuffer(MTL::Buffer *buffer);

  MTL::ComputePipelineState *getPipeline(const std::string &name);
  MTL::CommandQueue *commandQueue();
  MTL::Device *device();

  void synchronize();

private:
  MetalContext();
  ~MetalContext();

  MetalContext(const MetalContext &) = delete;
  MetalContext &operator=(const MetalContext &) = delete;

  MTL::Device *device_ = nullptr;
  MTL::CommandQueue *command_queue_ = nullptr;
  MTL::Library *library_ = nullptr;

  std::unordered_map<std::string, MTL::ComputePipelineState *> pipelines_;
  bool initialized_ = false;
};

#endif