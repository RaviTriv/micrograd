#pragma once

#include <cstdint>
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
  ~Tensor();

  std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor> &b);

  std::shared_ptr<Tensor> add(scalar_t scalar);
  std::shared_ptr<Tensor> sub(scalar_t scalar);
  std::shared_ptr<Tensor> mul(scalar_t scalar);
  std::shared_ptr<Tensor> div(scalar_t scalar);
  std::shared_ptr<Tensor> pow(scalar_t exponent);

  std::shared_ptr<Tensor> reshape(const std::vector<int64_t> &shape);
  std::shared_ptr<Tensor> view(const std::vector<int64_t> &shape);
  std::shared_ptr<Tensor> transpose(int64_t dim0, int64_t dim1);
  std::shared_ptr<Tensor> permute(const std::vector<int64_t> &dims);
  std::shared_ptr<Tensor> contiguous();

  std::shared_ptr<Tensor> sum();
  std::shared_ptr<Tensor> sum(int64_t dim, bool keepdim = false);
  std::shared_ptr<Tensor> mean(int64_t dim, bool keepdim = false);
  std::shared_ptr<Tensor> max(int64_t dim, bool keepdim = false);
  std::shared_ptr<Tensor> argmax(int64_t dim);
  std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> embedding_lookup(
      const std::shared_ptr<Tensor> &indices);

  std::shared_ptr<Tensor> relu();
  std::shared_ptr<Tensor> sigmoid();
  std::shared_ptr<Tensor> tanh();
  std::shared_ptr<Tensor> exp();
  std::shared_ptr<Tensor> log();
  std::shared_ptr<Tensor> sqrt();
  std::shared_ptr<Tensor> neg();
  std::shared_ptr<Tensor> softmax(int64_t dim);
  std::shared_ptr<Tensor> log_softmax(int64_t dim);

  void backward();
  void backward(const Tensor &grad_output);
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
  std::shared_ptr<Tensor> broadcast_to(const std::vector<size_t> &shape);
  std::shared_ptr<Tensor> strided_copy(const std::vector<size_t> &shape,
                                       const std::vector<size_t> &strides);
  void propagate_gradients();

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
