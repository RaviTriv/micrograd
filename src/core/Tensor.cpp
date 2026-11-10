#include "micrograd/Tensor.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "micrograd/Profiler.h"

#ifdef MICROGRAD_CUDA_ENABLED
#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"
#endif

namespace micrograd {
namespace {

size_t ElementCount(const std::vector<size_t> &shape) {
  if (std::ranges::find(shape, size_t{0}) != shape.end()) {
    return 0;
  }

  const size_t limit = std::numeric_limits<size_t>::max() / sizeof(scalar_t);
  size_t total = 1;
  for (auto dim : shape) {
    if (dim > limit / total) {
      throw std::invalid_argument("Tensor shape is too large");
    }
    total *= dim;
  }
  return total;
}

void FillBytes(Storage &storage, int value, const char *name) {
  if (storage.bytes() == 0) {
    return;
  }
  profiler::Scope scope(name, "-", storage.device());
  if (storage.device() != Device::CUDA) {
    std::memset(storage.host_pointer(), value, storage.bytes());
    return;
  }
#ifdef MICROGRAD_CUDA_ENABLED
  MICROGRAD_CUDA_CHECK(cudaMemsetAsync(storage.device_pointer(), value,
                                       storage.bytes(),
                                       CudaContext::instance().stream()));
#else
  throw std::runtime_error("CUDA support is not compiled in");
#endif
}

void ZeroFill(Storage &storage) { FillBytes(storage, 0, "zero_fill"); }

bool PoisonEmptyTensors() {
  static const bool poison = [] {
    const char *value = std::getenv("MICROGRAD_POISON_EMPTY");
    return value != nullptr && std::string(value) == "1";
  }();
  return poison;
}

}  // namespace

std::shared_ptr<Tensor> Tensor::zeros(std::vector<size_t> shape) {
  return std::make_shared<Tensor>(ConstructorKey{}, std::move(shape));
}

std::shared_ptr<Tensor> Tensor::zeros(std::vector<size_t> shape,
                                      Backend device, DType dtype) {
  return std::make_shared<Tensor>(ConstructorKey{}, std::move(shape), device,
                                  dtype);
}

std::shared_ptr<Tensor> Tensor::empty(std::vector<size_t> shape,
                                      Backend device, DType dtype) {
  Storage data(ElementCount(shape) * dtype_size(dtype), device, dtype);
  if (PoisonEmptyTensors()) {
    FillBytes(data, 0xFF, "poison_fill");
  }
  return std::make_shared<Tensor>(ConstructorKey{}, std::move(shape),
                                  std::move(data));
}

std::shared_ptr<Tensor> Tensor::from_values(std::vector<size_t> shape,
                                            std::vector<scalar_t> values) {
  return std::make_shared<Tensor>(ConstructorKey{}, std::move(shape),
                                  std::move(values));
}

Tensor::Tensor(ConstructorKey key, std::vector<size_t> shape)
    : Tensor(key, std::move(shape), Device::CPU) {}

Tensor::Tensor(ConstructorKey /*key*/, std::vector<size_t> shape,
               Backend device, DType dtype)
    : shape_(std::move(shape)) {
  data_ = Storage(ElementCount(shape_) * dtype_size(dtype), device, dtype);
  ZeroFill(data_);
}

Tensor::Tensor(ConstructorKey /*key*/, std::vector<size_t> shape, Storage data)
    : data_(std::move(data)), shape_(std::move(shape)) {}

Tensor::Tensor(ConstructorKey /*key*/, std::vector<size_t> shape,
               std::vector<scalar_t> values)
    : shape_(std::move(shape)) {
  const size_t total = ElementCount(shape_);
  if (values.size() != total) {
    throw std::invalid_argument("Tensor and data size mismatch");
  }
  data_ = Storage(total * sizeof(scalar_t), Device::CPU);
  std::ranges::copy(values, data().begin());
}

Tensor::~Tensor() {
  backward_fn_ = nullptr;
  std::vector<std::shared_ptr<Tensor>> pending = std::move(children_);
  while (!pending.empty()) {
    std::shared_ptr<Tensor> node = std::move(pending.back());
    pending.pop_back();
    if (node.use_count() != 1) {
      continue;
    }
    node->backward_fn_ = nullptr;
    pending.insert(pending.end(),
                   std::make_move_iterator(node->children_.begin()),
                   std::make_move_iterator(node->children_.end()));
    node->children_.clear();
  }
}

const std::vector<size_t> &Tensor::shape() const { return shape_; }

size_t Tensor::size() const { return data_.bytes() / dtype_size(dtype()); }

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
  if (indices.size() != shape_.size()) {
    throw std::invalid_argument(
        "Tensor index has " + std::to_string(indices.size()) +
        " dimensions, expected " + std::to_string(shape_.size()));
  }

  size_t idx = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i] >= shape_[i]) {
      throw std::out_of_range("Tensor index " + std::to_string(indices[i]) +
                              " is out of range for dimension " +
                              std::to_string(i) + " of size " +
                              std::to_string(shape_[i]));
    }
    idx = (idx * shape_[i]) + indices[i];
  }
  return idx;
}

std::span<scalar_t> Tensor::data() {
  require_float32();
  return {static_cast<scalar_t *>(data_.data()), size()};
}

std::span<const scalar_t> Tensor::data() const {
  require_float32();
  return {static_cast<const scalar_t *>(data_.data()), size()};
}

std::span<scalar_t> Tensor::grad() {
  require_float32();
  ensure_grad();
  return {static_cast<scalar_t *>(grad_.data()), size()};
}

std::span<const scalar_t> Tensor::grad() const {
  require_float32();
  ensure_grad();
  return {static_cast<const scalar_t *>(grad_.data()), size()};
}

void Tensor::require_float32() const {
  if (dtype() != DType::kFloat32) {
    throw std::runtime_error(
        "Tensor element access needs float32, cast the tensor first");
  }
}

bool Tensor::has_grad() const { return grad_.bytes() > 0; }

void Tensor::ensure_grad() const {
  if (has_grad()) {
    return;
  }

  const size_t total = ElementCount(shape_);
  if (total == 0) {
    return;
  }

  grad_ = Storage(total * dtype_size(dtype()), backend(), dtype());
  fill_grad_with_zeros();
}

void Tensor::zero_grad() {
  if (has_grad()) {
    fill_grad_with_zeros();
  }
}

void Tensor::release_grad() { grad_ = Storage(); }

void Tensor::fill_grad_with_zeros() const { ZeroFill(grad_); }

bool Tensor::requires_grad() const { return requires_grad_; }

void Tensor::set_requires_grad(bool requires_grad) {
  requires_grad_ = requires_grad;
}

void Tensor::set_post_accumulate_grad_hook(
    std::function<void(Tensor &)> hook) {
  post_accumulate_grad_hook_ = std::move(hook);
}

void Tensor::to(Backend device) {
  if (data_.device() == device) {
    return;
  }

  Storage moved_data = data_.copy_to(device);
  if (has_grad()) {
    grad_ = grad_.copy_to(device);
  }
  data_ = std::move(moved_data);
}

Backend Tensor::backend() const { return data_.device(); }

DType Tensor::dtype() const { return data_.dtype(); }

Storage &Tensor::data_storage() { return data_; }

const Storage &Tensor::data_storage() const { return data_; }

Storage &Tensor::grad_storage() {
  ensure_grad();
  return grad_;
}

const Storage &Tensor::grad_storage() const {
  ensure_grad();
  return grad_;
}

Storage &Tensor::grad_storage_for_overwrite() {
  if (has_grad()) {
    return grad_;
  }
  const size_t total = ElementCount(shape_);
  if (total == 0) {
    return grad_;
  }
  grad_ = Storage(total * dtype_size(dtype()), backend(), dtype());
  if (PoisonEmptyTensors()) {
    FillBytes(grad_, 0xFF, "poison_fill");
  }
  return grad_;
}

Storage Tensor::take_grad_storage() { return std::move(grad_); }

void Tensor::set_grad_storage(Storage grad) {
  if (grad.bytes() != size() * dtype_size(dtype()) ||
      grad.device() != backend() || grad.dtype() != dtype()) {
    throw std::invalid_argument("Gradient storage does not match the tensor");
  }
  grad_ = std::move(grad);
}

}  // namespace micrograd
