#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"

namespace micrograd::data {

struct Example {
  std::vector<scalar_t> input;
  std::vector<scalar_t> target;
};

class Dataset {
 public:
  virtual ~Dataset() = default;
  virtual size_t size() const = 0;
  virtual std::vector<size_t> input_shape() const = 0;
  virtual std::vector<size_t> target_shape() const = 0;
  virtual Example get(size_t index) const = 0;
};

class TensorDataset : public Dataset {
 public:
  TensorDataset(const Tensor &inputs, const Tensor &targets);

  size_t size() const override { return count_; }
  std::vector<size_t> input_shape() const override { return input_shape_; }
  std::vector<size_t> target_shape() const override { return target_shape_; }
  Example get(size_t index) const override;

 private:
  size_t count_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> target_shape_;
  size_t input_size_;
  size_t target_size_;
  std::vector<scalar_t> inputs_;
  std::vector<scalar_t> targets_;
};

struct Batch {
  std::shared_ptr<Tensor> inputs;
  std::shared_ptr<Tensor> targets;
};

class DataLoader {
 public:
  class Iterator {
   public:
    Iterator(const DataLoader *loader, size_t index)
        : loader_(loader), index_(index) {}
    Batch operator*() const { return loader_->batch(index_); }
    Iterator &operator++() {
      index_++;
      return *this;
    }
    bool operator==(const Iterator &other) const {
      return index_ == other.index_;
    }

   private:
    const DataLoader *loader_;
    size_t index_;
  };

  DataLoader(const Dataset &dataset, size_t batch_size, bool shuffle,
             uint64_t seed = 0);

  void set_epoch(size_t epoch);
  size_t size() const;
  Batch batch(size_t index) const;
  const std::vector<size_t> &order() const { return order_; }

  Iterator begin() const { return {this, 0}; }
  Iterator end() const { return {this, size()}; }

 private:
  const Dataset &dataset_;
  size_t batch_size_;
  bool shuffle_;
  uint64_t seed_;
  std::vector<size_t> order_;
};

std::vector<size_t> class_indices(const Tensor &targets);

}  // namespace micrograd::data
