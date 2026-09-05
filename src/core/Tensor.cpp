#include "micrograd/Tensor.h"

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
  data_.resize(total, 0.0);
  grad_.resize(total, 0.0);
  compute_strides();
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<double> data)
    : data_(std::move(data)), shape_(std::move(shape)) {
  size_t total = 1;
  for (auto dim : shape_) {
    total *= dim;
  }
  if (data_.size() != total) {
    throw std::invalid_argument("Tensor and data size mismatch");
  }
  grad_.resize(total, 0.0);
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

double &Tensor::at(const std::vector<size_t> &indices) {
  return data_[flat_index(indices)];
}

double Tensor::at(const std::vector<size_t> &indices) const {
  return data_[flat_index(indices)];
}

double &Tensor::grad_at(const std::vector<size_t> &indices) {
  return grad_[flat_index(indices)];
}

double Tensor::grad_at(const std::vector<size_t> &indices) const {
  return grad_[flat_index(indices)];
}

size_t Tensor::flat_index(const std::vector<size_t> &indices) const {
  size_t idx = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    idx += indices[i] * strides_[i];
  }
  return idx;
}

std::vector<double> &Tensor::data() { return data_; }

std::vector<double> &Tensor::grad() { return grad_; }

void Tensor::zero_grad() {
  for (double &g : grad_) {
    g = 0.0;
  }
}

void Tensor::to(Backend b) {
  if (backend_ == b) {
    return;
  }

#ifdef MICROGRAD_METAL_ENABLED
  if (b == Backend::Metal) {
    auto &ctx = MetalContext::instance();
    if (!ctx.isAvailable()) {
      ctx.initialize();
    }

    gpu_data_ = ctx.createBuffer(size() * sizeof(float));
    gpu_grad_ = ctx.createBuffer(size() * sizeof(float));

    auto *gpu_ptr = static_cast<float *>(gpu_data_->contents());
    for (size_t i = 0; i < size(); i++) {
      gpu_ptr[i] = static_cast<float>(data_[i]);
    }

    auto *grad_ptr = static_cast<float *>(gpu_grad_->contents());
    for (size_t i = 0; i < size(); i++) {
      grad_ptr[i] = 0.0f;
    }

    backend_ = Backend::Metal;
  } else {
    auto *gpu_ptr = static_cast<float *>(gpu_data_->contents());
    for (size_t i = 0; i < size(); i++) {
      data_[i] = static_cast<double>(gpu_ptr[i]);
    }

    auto *grad_ptr = static_cast<float *>(gpu_grad_->contents());
    for (size_t i = 0; i < size(); i++) {
      grad_[i] = static_cast<double>(grad_ptr[i]);
    }

    auto &ctx = MetalContext::instance();
    ctx.releaseBuffer(gpu_data_);
    ctx.releaseBuffer(gpu_grad_);
    gpu_data_ = nullptr;
    gpu_grad_ = nullptr;

    backend_ = Backend::CPU;
  }
#else
  (void)b;
#endif
}

Backend Tensor::backend() const { return backend_; }

}  // namespace micrograd
