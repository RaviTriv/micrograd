#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "micrograd/Backend.h"
#include "micrograd/DType.h"
#include "micrograd/Scalar.h"
#include "micrograd/Storage.h"

namespace micrograd {
enum class OpId;
struct OpInputs;

class Tensor : public std::enable_shared_from_this<Tensor> {
  struct ConstructorKey {
    explicit ConstructorKey() = default;
  };

 public:
  static std::shared_ptr<Tensor> zeros(std::vector<size_t> shape);
  static std::shared_ptr<Tensor> zeros(std::vector<size_t> shape,
                                       Backend device,
                                       DType dtype = DType::kFloat32);
  static std::shared_ptr<Tensor> empty(std::vector<size_t> shape,
                                       Backend device,
                                       DType dtype = DType::kFloat32);
  static std::shared_ptr<Tensor> from_values(std::vector<size_t> shape,
                                             std::vector<scalar_t> values);

  Tensor(ConstructorKey key, std::vector<size_t> shape);
  Tensor(ConstructorKey key, std::vector<size_t> shape, Backend device,
         DType dtype = DType::kFloat32);
  Tensor(ConstructorKey key, std::vector<size_t> shape, Storage data);
  Tensor(ConstructorKey key, std::vector<size_t> shape,
         std::vector<scalar_t> values);
  ~Tensor();

  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor(Tensor &&) = delete;
  Tensor &operator=(Tensor &&) = delete;

  std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> add_bias(const std::shared_ptr<Tensor> &bias);
  std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor> &b);

  std::shared_ptr<Tensor> add(scalar_t scalar);
  std::shared_ptr<Tensor> sub(scalar_t scalar);
  std::shared_ptr<Tensor> mul(scalar_t scalar);
  std::shared_ptr<Tensor> div(scalar_t scalar);
  std::shared_ptr<Tensor> pow(scalar_t exponent);
  std::shared_ptr<Tensor> clamp(scalar_t min, scalar_t max);

  std::shared_ptr<Tensor> reshape(const std::vector<int64_t> &shape);
  std::shared_ptr<Tensor> view(const std::vector<int64_t> &shape);
  std::shared_ptr<Tensor> transpose(int64_t dim0, int64_t dim1);
  std::shared_ptr<Tensor> permute(const std::vector<int64_t> &dims);
  std::shared_ptr<Tensor> contiguous();
  std::shared_ptr<Tensor> cast(DType dtype);

  std::shared_ptr<Tensor> sum();
  std::shared_ptr<Tensor> sum(int64_t dim, bool keepdim = false);
  std::shared_ptr<Tensor> mean(int64_t dim, bool keepdim = false);
  std::shared_ptr<Tensor> max(int64_t dim, bool keepdim = false);
  std::shared_ptr<Tensor> argmax(int64_t dim);
  std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> matmul_nt(const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> linear(const std::shared_ptr<Tensor> &weight,
                                 const std::shared_ptr<Tensor> &bias);
  std::shared_ptr<Tensor> embedding_lookup(
      const std::shared_ptr<Tensor> &indices);
  std::shared_ptr<Tensor> gather_per_row(
      const std::shared_ptr<Tensor> &indices);
  std::shared_ptr<Tensor> gather_per_row(const std::vector<size_t> &indices);
  std::shared_ptr<Tensor> cross_entropy_rows(
      const std::vector<size_t> &targets, size_t classes = 0);
  std::shared_ptr<Tensor> layer_norm(
      const std::vector<size_t> &normalized_shape,
      const std::shared_ptr<Tensor> &gain, const std::shared_ptr<Tensor> &bias,
      scalar_t eps);
  std::shared_ptr<Tensor> rms_norm(const std::vector<size_t> &normalized_shape,
                                   const std::shared_ptr<Tensor> &gain,
                                   scalar_t eps);
  std::shared_ptr<Tensor> rotary_embedding(scalar_t base = 10000.0f,
                                           int64_t seq_dim = -2);
  std::shared_ptr<Tensor> head_norm_rotary(size_t n_head,
                                           const std::shared_ptr<Tensor> &gain,
                                           scalar_t eps, bool rotate,
                                           scalar_t base = 10000.0f);
  std::shared_ptr<Tensor> flash_attention(const std::shared_ptr<Tensor> &key,
                                          const std::shared_ptr<Tensor> &value,
                                          scalar_t scale);
  std::shared_ptr<Tensor> causal_attention_probs(
      const std::shared_ptr<Tensor> &key, size_t n_head, scalar_t scale);
  std::shared_ptr<Tensor> attention_combine(
      const std::shared_ptr<Tensor> &value);
  std::shared_ptr<Tensor> causal_self_attention(
      const std::shared_ptr<Tensor> &key, const std::shared_ptr<Tensor> &value,
      size_t n_head, scalar_t scale);

  std::shared_ptr<Tensor> relu();
  std::shared_ptr<Tensor> relu_squared();
  std::shared_ptr<Tensor> sigmoid();
  std::shared_ptr<Tensor> tanh();
  std::shared_ptr<Tensor> gelu();
  std::shared_ptr<Tensor> exp();
  std::shared_ptr<Tensor> log();
  std::shared_ptr<Tensor> sqrt();
  std::shared_ptr<Tensor> neg();
  std::shared_ptr<Tensor> softmax(int64_t dim);
  std::shared_ptr<Tensor> log_softmax(int64_t dim);

  void backward(bool retain_graph = false);
  void backward(const Tensor &grad_output, bool retain_graph = false);
  void zero_grad();
  void release_grad();
  bool has_grad() const;
  bool requires_grad() const;
  void set_requires_grad(bool requires_grad);
  void set_post_accumulate_grad_hook(std::function<void(Tensor &)> hook);

  static std::shared_ptr<Tensor> checkpoint(
      const std::shared_ptr<Tensor> &input,
      const std::function<
          std::shared_ptr<Tensor>(const std::shared_ptr<Tensor> &)> &recompute);

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
  std::optional<std::span<const scalar_t>> grad_if_present() const;
  void to(Backend device);
  Backend backend() const;
  DType dtype() const;
  Storage &data_storage();
  const Storage &data_storage() const;
  Storage &grad_storage();
  const Storage &grad_storage() const;
  Storage &grad_storage_for_overwrite();
  Storage take_grad_storage();
  void set_grad_storage(Storage grad);

 private:
  void ensure_grad() const;
  void require_float32() const;
  void fill_grad_with_zeros() const;
  std::shared_ptr<Tensor> apply(OpId op, std::vector<size_t> out_shape,
                                const OpInputs &inputs);
  std::shared_ptr<Tensor> apply_binary(OpId op,
                                       const std::shared_ptr<Tensor> &b);
  std::shared_ptr<Tensor> broadcast_to(const std::vector<size_t> &shape);
  std::shared_ptr<Tensor> strided_copy(const std::vector<size_t> &shape,
                                       const std::vector<size_t> &strides);
  void propagate_gradients(Storage seed, bool retain_graph);
  void require_grad_for_backward() const;
  void accumulate_into_grad(const Storage &addend);

  Storage data_;
  mutable Storage grad_;
  std::vector<size_t> shape_;

  std::vector<std::shared_ptr<Tensor>> children_;
  std::function<void()> backward_fn_;
  std::function<void(Tensor &)> post_accumulate_grad_hook_;
  bool requires_grad_ = false;
  bool graph_released_ = false;

  size_t flat_index(const std::vector<size_t> &indices) const;
};

}  // namespace micrograd
