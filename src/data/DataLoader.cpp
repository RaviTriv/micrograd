#include "micrograd/data/DataLoader.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace micrograd::data {

namespace {

size_t element_count(const std::vector<size_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         std::multiplies<>());
}

void require_cpu(const Tensor &tensor, const char *what) {
  if (tensor.backend() != Device::CPU) {
    throw std::invalid_argument(std::string(what) + " must be on the CPU");
  }
}

std::vector<size_t> with_leading(size_t count, const std::vector<size_t> &shape) {
  std::vector<size_t> full{count};
  full.insert(full.end(), shape.begin(), shape.end());
  return full;
}

}  // namespace

TensorDataset::TensorDataset(const Tensor &inputs, const Tensor &targets) {
  require_cpu(inputs, "TensorDataset inputs");
  require_cpu(targets, "TensorDataset targets");
  if (inputs.shape().empty() || targets.shape().empty()) {
    throw std::invalid_argument(
        "TensorDataset needs inputs and targets with a sample dimension");
  }
  if (inputs.shape()[0] != targets.shape()[0]) {
    throw std::invalid_argument(
        "TensorDataset has " + std::to_string(inputs.shape()[0]) +
        " inputs but " + std::to_string(targets.shape()[0]) + " targets");
  }

  count_ = inputs.shape()[0];
  input_shape_.assign(inputs.shape().begin() + 1, inputs.shape().end());
  target_shape_.assign(targets.shape().begin() + 1, targets.shape().end());
  input_size_ = element_count(input_shape_);
  target_size_ = element_count(target_shape_);
  inputs_.assign(inputs.data().begin(), inputs.data().end());
  targets_.assign(targets.data().begin(), targets.data().end());
}

Example TensorDataset::get(size_t index) const {
  if (index >= count_) {
    throw std::out_of_range("TensorDataset index " + std::to_string(index) +
                            " out of range for size " +
                            std::to_string(count_));
  }
  auto input = inputs_.begin() + static_cast<std::ptrdiff_t>(index * input_size_);
  auto target =
      targets_.begin() + static_cast<std::ptrdiff_t>(index * target_size_);
  return {.input = {input, input + static_cast<std::ptrdiff_t>(input_size_)},
          .target = {target,
                     target + static_cast<std::ptrdiff_t>(target_size_)}};
}

DataLoader::DataLoader(const Dataset &dataset, size_t batch_size, bool shuffle,
                       uint64_t seed)
    : dataset_(dataset),
      batch_size_(batch_size),
      shuffle_(shuffle),
      seed_(seed) {
  if (batch_size_ == 0) {
    throw std::invalid_argument("DataLoader batch size must be positive");
  }
  set_epoch(0);
}

void DataLoader::set_epoch(size_t epoch) {
  order_.resize(dataset_.size());
  std::iota(order_.begin(), order_.end(), size_t{0});
  if (!shuffle_) {
    return;
  }
  std::mt19937_64 rng(seed_ + static_cast<uint64_t>(epoch));
  for (size_t i = order_.size(); i > 1; i--) {
    auto j = static_cast<size_t>(rng() % i);
    std::swap(order_[i - 1], order_[j]);
  }
}

size_t DataLoader::size() const {
  return (order_.size() + batch_size_ - 1) / batch_size_;
}

Batch DataLoader::batch(size_t index) const {
  if (index >= size()) {
    throw std::out_of_range("DataLoader batch " + std::to_string(index) +
                            " out of range for " + std::to_string(size()) +
                            " batches");
  }
  const size_t start = index * batch_size_;
  const size_t count = std::min(batch_size_, order_.size() - start);
  const std::vector<size_t> input_shape = dataset_.input_shape();
  const std::vector<size_t> target_shape = dataset_.target_shape();
  const size_t input_size = element_count(input_shape);
  const size_t target_size = element_count(target_shape);

  std::vector<scalar_t> inputs;
  std::vector<scalar_t> targets;
  inputs.reserve(count * input_size);
  targets.reserve(count * target_size);
  for (size_t row = 0; row < count; row++) {
    Example example = dataset_.get(order_[start + row]);
    if (example.input.size() != input_size ||
        example.target.size() != target_size) {
      throw std::runtime_error("DataLoader: sample " +
                               std::to_string(order_[start + row]) +
                               " does not match the dataset shapes");
    }
    inputs.insert(inputs.end(), example.input.begin(), example.input.end());
    targets.insert(targets.end(), example.target.begin(),
                   example.target.end());
  }

  return {.inputs = Tensor::from_values(with_leading(count, input_shape),
                                        std::move(inputs)),
          .targets = Tensor::from_values(with_leading(count, target_shape),
                                         std::move(targets))};
}

std::vector<size_t> class_indices(const Tensor &targets) {
  require_cpu(targets, "class_indices targets");
  std::vector<size_t> indices;
  indices.reserve(targets.size());
  for (scalar_t value : targets.data()) {
    if (!std::isfinite(value) || value < 0 || std::floor(value) != value) {
      throw std::invalid_argument(
          "class_indices expects non-negative whole numbers, got " +
          std::to_string(value));
    }
    indices.push_back(static_cast<size_t>(value));
  }
  return indices;
}

}  // namespace micrograd::data
