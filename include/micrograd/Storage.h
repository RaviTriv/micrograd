#pragma once

#include <cstddef>

#include "micrograd/Device.h"

namespace MTL {
class Buffer;
}

namespace micrograd {

class Storage {
 public:
  Storage() = default;
  Storage(size_t bytes, Device device);
  ~Storage();

  Storage(const Storage &) = delete;
  Storage &operator=(const Storage &) = delete;
  Storage(Storage &&other) noexcept;
  Storage &operator=(Storage &&other) noexcept;

  Storage copy_to(Device device) const;

  void *data() const { return data_; }
  MTL::Buffer *buffer() const;
  size_t bytes() const { return bytes_; }
  Device device() const { return device_; }

 private:
  void release();

  void *data_ = nullptr;
  size_t bytes_ = 0;
  Device device_ = Device::CPU;
};

}  // namespace micrograd
