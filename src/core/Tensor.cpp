#include "micrograd/Tensor.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace micrograd {

Tensor::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
  size_t total = 1;
  for (auto dim : shape_) {
    total *= dim;
  }
  data_ = Storage(total * sizeof(scalar_t), Device::CPU);
  grad_ = Storage(total * sizeof(scalar_t), Device::CPU);
  std::ranges::fill(data(), 0.0f);
  zero_grad();
  compute_strides();
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<scalar_t> values)
    : shape_(std::move(shape)) {
  size_t total = 1;
  for (auto dim : shape_) {
    total *= dim;
  }
  if (values.size() != total) {
    throw std::invalid_argument("Tensor and data size mismatch");
  }
  data_ = Storage(total * sizeof(scalar_t), Device::CPU);
  grad_ = Storage(total * sizeof(scalar_t), Device::CPU);
  std::ranges::copy(values, data().begin());
  zero_grad();
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

size_t Tensor::size() const { return data_.bytes() / sizeof(scalar_t); }

scalar_t &Tensor::at(const std::vector<size_t> &indices) {
  return data()[flat_index(indices)];
}

scalar_t Tensor::at(const std::vector<size_t> &indices) const {
  return data()[flat_index(indices)];
}

scalar_t &Tensor::grad_at(const std::vector<size_t> &indices) {
  return grad()[flat_index(indices)];
}

scalar_t Tensor::grad_at(const std::vector<size_t> &indices) const {
  return grad()[flat_index(indices)];
}

size_t Tensor::flat_index(const std::vector<size_t> &indices) const {
  size_t idx = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    idx += indices[i] * strides_[i];
  }
  return idx;
}

std::span<scalar_t> Tensor::data() {
  return {static_cast<scalar_t *>(data_.data()), size()};
}

std::span<const scalar_t> Tensor::data() const {
  return {static_cast<const scalar_t *>(data_.data()), size()};
}

std::span<scalar_t> Tensor::grad() {
  return {static_cast<scalar_t *>(grad_.data()), size()};
}

std::span<const scalar_t> Tensor::grad() const {
  return {static_cast<const scalar_t *>(grad_.data()), size()};
}

void Tensor::zero_grad() {
  std::fill_n(static_cast<scalar_t *>(grad_.host_pointer()), size(), 0.0f);
}

void Tensor::to(Backend device) {
  if (data_.device() == device) {
    return;
  }

  Storage moved_data = data_.copy_to(device);
  Storage moved_grad = grad_.copy_to(device);
  data_ = std::move(moved_data);
  grad_ = std::move(moved_grad);
}

Backend Tensor::backend() const { return data_.device(); }

}  // namespace micrograd
