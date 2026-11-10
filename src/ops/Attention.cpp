#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::flash_attention(
    const std::shared_ptr<Tensor> &key, const std::shared_ptr<Tensor> &value,
    scalar_t scale) {
  if (shape_.size() != 3) {
    throw std::invalid_argument(
        "flash_attention expects a (batch_heads, seq, head_dim) input");
  }
  if (key->shape() != shape_ || value->shape() != shape_) {
    throw std::invalid_argument(
        "flash_attention expects query, key, and value to share a shape");
  }

  return apply(
      OpId::kFlashAttention, shape_,
      {.lhs = shared_from_this(), .rhs = key, .extra = value, .scalar = scale});
}

std::shared_ptr<Tensor> Tensor::causal_attention_probs(
    const std::shared_ptr<Tensor> &key, size_t n_head, scalar_t scale) {
  if (shape_.size() != 3) {
    throw std::invalid_argument(
        "causal_attention_probs expects a (batch, seq, embed) query");
  }
  if (key->shape() != shape_) {
    throw std::invalid_argument(
        "causal_attention_probs expects query and key to share a shape");
  }
  if (key->backend() != backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }
  if (n_head == 0 || shape_[2] % n_head != 0) {
    throw std::invalid_argument(
        "causal_attention_probs: embed must be divisible by n_head");
  }

  const size_t seq_len = shape_[1];
  return apply(OpId::kCausalAttentionProbs,
               {shape_[0], n_head, seq_len, seq_len},
               {.lhs = shared_from_this(),
                .rhs = key,
                .scalar = scale,
                .dim = static_cast<int64_t>(n_head)});
}

std::shared_ptr<Tensor> Tensor::causal_self_attention(
    const std::shared_ptr<Tensor> &key, const std::shared_ptr<Tensor> &value,
    size_t n_head, scalar_t scale) {
  if (shape_.size() != 3) {
    throw std::invalid_argument(
        "causal_self_attention expects a (batch, seq, embed) query");
  }
  if (key->shape() != shape_ || value->shape() != shape_) {
    throw std::invalid_argument(
        "causal_self_attention expects query, key, and value to share a shape");
  }
  if (key->backend() != backend() || value->backend() != backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }
  if (n_head == 0 || shape_[2] % n_head != 0) {
    throw std::invalid_argument(
        "causal_self_attention: embed must be divisible by n_head");
  }

  auto log_sum_exp = Tensor::empty({shape_[0], n_head, shape_[1]}, backend());
  return apply(OpId::kCausalSelfAttention, shape_,
               {.lhs = shared_from_this(),
                .rhs = key,
                .extra = value,
                .aux = std::move(log_sum_exp),
                .scalar = scale,
                .dim = static_cast<int64_t>(n_head)});
}

std::shared_ptr<Tensor> Tensor::attention_combine(
    const std::shared_ptr<Tensor> &value) {
  if (shape_.size() != 4) {
    throw std::invalid_argument(
        "attention_combine expects (batch, heads, seq, keys) weights");
  }
  const std::vector<size_t> &value_shape = value->shape();
  if (value_shape.size() != 3 || value_shape[0] != shape_[0] ||
      value_shape[1] != shape_[3] || value_shape[2] % shape_[1] != 0) {
    throw std::invalid_argument(
        "attention_combine expects a (batch, keys, heads * head_dim) value");
  }
  if (value->backend() != backend()) {
    throw std::invalid_argument("Tensor devices do not match");
  }

  return apply(OpId::kAttentionCombine, {shape_[0], shape_[2], value_shape[2]},
               {.lhs = shared_from_this(),
                .rhs = value,
                .dim = static_cast<int64_t>(shape_[1])});
}

}  // namespace micrograd
