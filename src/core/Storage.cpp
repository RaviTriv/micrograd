#include "micrograd/Storage.h"

#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

#ifdef MICROGRAD_METAL_ENABLED
#include "micrograd/metal/MetalContext.h"
#endif

namespace micrograd {
namespace {

void *allocate(size_t bytes, Device device) {
  if (bytes == 0) {
    return nullptr;
  }

  switch (device) {
    case Device::CPU:
      return ::operator new(bytes);
    case Device::Metal: {
#ifdef MICROGRAD_METAL_ENABLED
      auto &ctx = MetalContext::instance();
      if (!ctx.isAvailable() && !ctx.initialize()) {
        throw std::runtime_error("Metal device is unavailable");
      }
      MTL::Buffer *buffer = ctx.createBuffer(bytes);
      if (!buffer) {
        throw std::runtime_error("Metal allocation failed");
      }
      return buffer;
#else
      throw std::runtime_error("Metal support is not compiled in");
#endif
    }
    case Device::CUDA:
      throw std::runtime_error("CUDA support is not compiled in");
  }

  throw std::runtime_error("Unknown device");
}

void deallocate(void *data, Device device) {
  if (!data) {
    return;
  }

  switch (device) {
    case Device::CPU:
      ::operator delete(data);
      return;
    case Device::Metal:
#ifdef MICROGRAD_METAL_ENABLED
      MetalContext::instance().releaseBuffer(static_cast<MTL::Buffer *>(data));
#endif
      return;
    case Device::CUDA:
      throw std::runtime_error("CUDA support is not compiled in");
  }
}

}  // namespace

Storage::Storage(size_t bytes, Device device)
    : data_(allocate(bytes, device)), bytes_(bytes), device_(device) {}

Storage::~Storage() { release(); }  // NOLINT(bugprone-exception-escape)

Storage::Storage(Storage &&other) noexcept
    : data_(other.data_), bytes_(other.bytes_), device_(other.device_) {
  other.data_ = nullptr;
  other.bytes_ = 0;
}

Storage &Storage::operator=(  // NOLINT(bugprone-exception-escape)
    Storage &&other) noexcept {
  if (this != &other) {
    release();
    data_ = other.data_;
    bytes_ = other.bytes_;
    device_ = other.device_;
    other.data_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

Storage Storage::copy_to(Device device) const {
  if (device == Device::CUDA || device_ == Device::CUDA) {
    throw std::runtime_error("CUDA support is not compiled in");
  }

  Storage copy(bytes_, device);
  if (bytes_ > 0) {
    std::memcpy(copy.host_pointer(), host_pointer(), bytes_);
  }
  return copy;
}

void *Storage::data() {
  return const_cast<void *>(std::as_const(*this).data());
}

const void *Storage::data() const {
  if (device_ != Device::CPU) {
    throw std::runtime_error("Storage data is not host memory");
  }
  return data_;
}

void *Storage::host_pointer() {
  return const_cast<void *>(std::as_const(*this).host_pointer());
}

const void *Storage::host_pointer() const {
  if (device_ != Device::Metal) {
    return data_;
  }
#ifdef MICROGRAD_METAL_ENABLED
  return static_cast<MTL::Buffer *>(data_)->contents();
#else
  return nullptr;
#endif
}

MTL::Buffer *Storage::buffer() const {
  if (device_ != Device::Metal) {
    return nullptr;
  }
  return static_cast<MTL::Buffer *>(data_);
}

void Storage::release() {
  deallocate(data_, device_);
  data_ = nullptr;
  bytes_ = 0;
}

}  // namespace micrograd
