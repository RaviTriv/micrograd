#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "micrograd/Backend.h"
#include "micrograd/Scalar.h"
#ifdef MICROGRAD_METAL_ENABLED
#include "micrograd/metal/MetalContext.h"
#endif

namespace micrograd {
class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  Tensor(std::vector<size_t> shape);
  Tensor(std::vector<size_t> shape, std::vector<scalar_t> data);
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

  const std::vector<size_t> &shape() const;
  size_t size() const;
  scalar_t &at(const std::vector<size_t> &indices);
  scalar_t at(const std::vector<size_t> &indices) const;
  scalar_t &grad_at(const std::vector<size_t> &indices);
  scalar_t grad_at(const std::vector<size_t> &indices) const;
  std::vector<scalar_t> &data();
  std::vector<scalar_t> &grad();
  void to(Backend backend);
  Backend backend() const;

 private:
  void compute_strides();

  std::vector<scalar_t> data_;
  std::vector<scalar_t> grad_;
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;

  std::vector<std::shared_ptr<Tensor>> children_;
  std::function<void()> backward_fn_;

  size_t flat_index(const std::vector<size_t> &indices) const;

  Backend backend_ = Backend::CPU;

#ifdef MICROGRAD_METAL_ENABLED
  MTL::Buffer *gpu_data_ = nullptr;
  MTL::Buffer *gpu_grad_ = nullptr;
  std::shared_ptr<Tensor> add_metal(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> sub_metal(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> mul_metal(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> div_metal(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> add_scalar_metal(scalar_t scalar);
  std::shared_ptr<Tensor> sub_scalar_metal(scalar_t scalar);
  std::shared_ptr<Tensor> mul_scalar_metal(scalar_t scalar);
  std::shared_ptr<Tensor> div_scalar_metal(scalar_t scalar);
  std::shared_ptr<Tensor> pow_metal(scalar_t exponent);
  std::shared_ptr<Tensor> matmul_metal(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> relu_metal();
  std::shared_ptr<Tensor> sigmoid_metal();
  std::shared_ptr<Tensor> tanh_metal();
  std::shared_ptr<Tensor> sum_metal();
#endif
};

}  // namespace micrograd
