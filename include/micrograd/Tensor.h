#pragma once

#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "micrograd/Backend.h"
#include "micrograd/Scalar.h"
#include "micrograd/Storage.h"

namespace micrograd {
class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  Tensor(std::vector<size_t> shape);
  Tensor(std::vector<size_t> shape, std::vector<scalar_t> values);
  ~Tensor() = default;

  std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor> &b);

  std::shared_ptr<Tensor> add(scalar_t scalar);
  std::shared_ptr<Tensor> sub(scalar_t scalar);
  std::shared_ptr<Tensor> mul(scalar_t scalar);
  std::shared_ptr<Tensor> div(scalar_t scalar);
  std::shared_ptr<Tensor> pow(scalar_t exponent);

  std::shared_ptr<Tensor> sum();
  std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor> &b);

  std::shared_ptr<Tensor> relu();
  std::shared_ptr<Tensor> sigmoid();
  std::shared_ptr<Tensor> tanh();

  void backward();
  void zero_grad();
  bool requires_grad() const;
  void set_requires_grad(bool requires_grad);

  const std::vector<size_t> &shape() const;
  size_t size() const;
  scalar_t &at(const std::vector<size_t> &indices);
  scalar_t at(const std::vector<size_t> &indices) const;
  scalar_t &grad_at(const std::vector<size_t> &indices);
  scalar_t grad_at(const std::vector<size_t> &indices) const;
  std::span<scalar_t> data();
  std::span<const scalar_t> data() const;
  std::span<scalar_t> grad();
  std::span<const scalar_t> grad() const;
  void to(Backend device);
  Backend backend() const;
  Storage &data_storage();
  const Storage &data_storage() const;
  Storage &grad_storage();
  const Storage &grad_storage() const;

 private:
  void compute_strides();

  Storage data_;
  Storage grad_;
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;

  std::vector<std::shared_ptr<Tensor>> children_;
  std::function<void()> backward_fn_;
  bool requires_grad_ = false;

  size_t flat_index(const std::vector<size_t> &indices) const;
};

}  // namespace micrograd
