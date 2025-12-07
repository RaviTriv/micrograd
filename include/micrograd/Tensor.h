#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

class Tensor : public std::enable_shared_from_this<Tensor> {
  friend std::string to_dot(const std::shared_ptr<Tensor> &tensor);

public:
  Tensor(std::vector<size_t> shape);
  Tensor(std::vector<size_t> shape, std::vector<double> data);
  ~Tensor() = default;

  std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor> &b);

  std::shared_ptr<Tensor> add(double scalar);
  std::shared_ptr<Tensor> sub(double scalar);
  std::shared_ptr<Tensor> mul(double scalar);
  std::shared_ptr<Tensor> div(double scalar);
  std::shared_ptr<Tensor> pow(double exponent);

  std::shared_ptr<Tensor> sum();
  std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor> &b);

  std::shared_ptr<Tensor> relu();
  std::shared_ptr<Tensor> sigmoid();
  std::shared_ptr<Tensor> tanh();

  void backward();
  void zero_grad();

  const std::vector<size_t> &shape() const;
  size_t size() const;
  double &at(const std::vector<size_t> &indices);
  double at(const std::vector<size_t> &indices) const;
  double &grad_at(const std::vector<size_t> &indices);
  double grad_at(const std::vector<size_t> &indices) const;

private:
  void compute_strides();

  std::vector<double> data_;
  std::vector<double> grad_;
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;

  std::vector<std::shared_ptr<Tensor>> children_;
  std::function<void()> backward_fn_;

  size_t flat_index(const std::vector<size_t> &indices) const;

  std::string op_;
};
