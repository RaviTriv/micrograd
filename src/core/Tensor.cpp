#include "micrograd/Tensor.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#ifdef MICROGRAD_METAL_ENABLED
#include "micrograd/metal/MetalContext.h"
#endif

namespace micrograd {

Tensor::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
  size_t total = 1;
  for (auto dim : shape_) {
    total *= dim;
  }
  data_.resize(total, 0.0f);
  grad_.resize(total, 0.0f);
  compute_strides();
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<scalar_t> data)
    : data_(std::move(data)), shape_(std::move(shape)) {
  size_t total = 1;
  for (auto dim : shape_) {
    total *= dim;
  }
  if (data_.size() != total) {
    throw std::invalid_argument("Tensor and data size mismatch");
  }
  grad_.resize(total, 0.0f);
  compute_strides();
}

void Tensor::compute_strides() {
  strides_.resize(shape_.size());
  size_t stride = 1;
  for (size_t i = shape_.size(); i > 0; i--) {
    strides_[i - 1] = stride;
    stride *= shape_[i - 1];
  }
}

const std::vector<size_t> &Tensor::shape() const { return shape_; }

size_t Tensor::size() const { return data_.size(); }

scalar_t &Tensor::at(const std::vector<size_t> &indices) {
  return data_[flat_index(indices)];
}

scalar_t Tensor::at(const std::vector<size_t> &indices) const {
  return data_[flat_index(indices)];
}

scalar_t &Tensor::grad_at(const std::vector<size_t> &indices) {
  return grad_[flat_index(indices)];
}

scalar_t Tensor::grad_at(const std::vector<size_t> &indices) const {
  return grad_[flat_index(indices)];
}

size_t Tensor::flat_index(const std::vector<size_t> &indices) const {
  size_t idx = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    idx += indices[i] * strides_[i];
  }
  return idx;
}

std::vector<scalar_t> &Tensor::data() { return data_; }

std::vector<scalar_t> &Tensor::grad() { return grad_; }

void Tensor::zero_grad() {
  for (scalar_t &g : grad_) {
    g = 0.0f;
  }
}

void Tensor::to(Backend b) {
  if (backend_ == b) {
    return;
  }

  switch (b) {
    case Backend::Metal: {
#ifdef MICROGRAD_METAL_ENABLED
      auto &ctx = MetalContext::instance();
      if (!ctx.isAvailable() && !ctx.initialize()) {
        throw std::runtime_error("Metal device is unavailable");
      }

      MTL::Buffer *data_buffer = ctx.createBuffer(size() * sizeof(scalar_t));
      if (!data_buffer) {
        throw std::runtime_error("Metal allocation failed");
      }
      MTL::Buffer *grad_buffer = ctx.createBuffer(size() * sizeof(scalar_t));
      if (!grad_buffer) {
        ctx.releaseBuffer(data_buffer);
        throw std::runtime_error("Metal allocation failed");
      }

      gpu_data_ = data_buffer;
      gpu_grad_ = grad_buffer;

      std::ranges::copy(data_, static_cast<scalar_t *>(gpu_data_->contents()));
      std::fill_n(static_cast<scalar_t *>(gpu_grad_->contents()), size(), 0.0f);

      backend_ = Backend::Metal;
#endif
      return;
    }
    case Backend::CPU: {
#ifdef MICROGRAD_METAL_ENABLED
      std::copy_n(static_cast<scalar_t *>(gpu_data_->contents()), size(),
                  data_.begin());
      std::copy_n(static_cast<scalar_t *>(gpu_grad_->contents()), size(),
                  grad_.begin());

      auto &ctx = MetalContext::instance();
      ctx.releaseBuffer(gpu_data_);
      ctx.releaseBuffer(gpu_grad_);
      gpu_data_ = nullptr;
      gpu_grad_ = nullptr;

      backend_ = Backend::CPU;
#endif
      return;
    }
    case Backend::CUDA:
      throw std::runtime_error("CUDA support is not compiled in");
  }

  throw std::runtime_error("Unknown device");
}

Backend Tensor::backend() const { return backend_; }

}  // namespace micrograd
