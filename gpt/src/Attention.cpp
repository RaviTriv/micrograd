#include "gpt/Attention.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace micrograd::gpt {

std::shared_ptr<Tensor> causal_mask(size_t seq_len, Device device) {
  auto mask = Tensor::zeros({seq_len, seq_len});
  std::span<scalar_t> values = mask->data();
  for (size_t row = 0; row < seq_len; row++) {
    for (size_t col = row + 1; col < seq_len; col++) {
      values[(row * seq_len) + col] =
          -std::numeric_limits<scalar_t>::infinity();
    }
  }
  mask->to(device);
  return mask;
}

std::shared_ptr<Tensor> causal_attention(const std::shared_ptr<Tensor> &q,
                                         const std::shared_ptr<Tensor> &k,
                                         const std::shared_ptr<Tensor> &v,
                                         scalar_t scale,
                                         const std::shared_ptr<Tensor> &mask) {
  auto scores = q->mul(scale)->matmul(k->transpose(1, 2))->add(mask);
  return scores->softmax(-1)->matmul(v);
}

CausalSelfAttention::CausalSelfAttention(size_t n_embd, size_t n_head,
                                         scalar_t dropout,
                                         const ArchOptions &arch)
    : n_embd_(n_embd), n_head_(n_head), arch_(arch) {
  if (n_head == 0 || n_embd % n_head != 0) {
    throw std::invalid_argument(
        "CausalSelfAttention: n_embd must be divisible by n_head");
  }

  query_ = std::make_shared<Linear>(n_embd, n_embd);
  key_ = std::make_shared<Linear>(n_embd, n_embd);
  value_ = std::make_shared<Linear>(n_embd, n_embd);
  out_proj_ = std::make_shared<Linear>(n_embd, n_embd);
  resid_dropout_ = std::make_shared<Dropout>(dropout);

  register_module("query", query_);
  register_module("key", key_);
  register_module("value", value_);
  register_module("out_proj", out_proj_);
  register_module("resid_dropout", resid_dropout_);

  if (arch_.qk_norm) {
    const size_t head_dim = n_embd / n_head;
    q_norm_ = std::make_shared<RMSNorm>(std::vector<size_t>{head_dim});
    k_norm_ = std::make_shared<RMSNorm>(std::vector<size_t>{head_dim});
    register_module("q_norm", q_norm_);
    register_module("k_norm", k_norm_);
  }
}

std::shared_ptr<Tensor> CausalSelfAttention::forward(
    const std::shared_ptr<Tensor> &input) {
  const std::vector<size_t> &shape = input->shape();
  if (shape.size() != 3) {
    throw std::invalid_argument(
        "CausalSelfAttention expects a (batch, seq, embed) input");
  }

  size_t head_dim = n_embd_ / n_head_;
  scalar_t scale = 1.0f / std::sqrt(static_cast<scalar_t>(head_dim));

  std::shared_ptr<Tensor> q = query_->forward(input);
  std::shared_ptr<Tensor> k = key_->forward(input);
  std::shared_ptr<Tensor> v = value_->forward(input);
  if (arch_.qk_norm || arch_.rotary) {
    const auto prepare = [&](const std::shared_ptr<Tensor> &x,
                             const std::shared_ptr<RMSNorm> &norm) {
      return x->head_norm_rotary(n_head_, norm ? norm->gain() : nullptr,
                                 norm ? norm->eps() : 0.0f, arch_.rotary);
    };
    q = prepare(q, q_norm_);
    k = prepare(k, k_norm_);
  }
  std::shared_ptr<Tensor> attended =
      q->causal_self_attention(k, v, n_head_, scale);
  return resid_dropout_->forward(out_proj_->forward(attended));
}

}  // namespace micrograd::gpt
