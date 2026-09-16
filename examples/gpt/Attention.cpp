#include "examples/gpt/Attention.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace micrograd::gpt {

namespace {

constexpr scalar_t kMaskValue = -1e9f;

std::shared_ptr<Tensor> causal_bias(size_t seq_len, Backend backend) {
  std::vector<scalar_t> values(seq_len * seq_len, 0.0f);
  for (size_t row = 0; row < seq_len; row++) {
    for (size_t col = row + 1; col < seq_len; col++) {
      values[(row * seq_len) + col] = kMaskValue;
    }
  }

  auto bias = std::make_shared<Tensor>(std::vector<size_t>{seq_len, seq_len},
                                       std::move(values));
  bias->to(backend);
  return bias;
}

std::shared_ptr<Tensor> split_heads(const std::shared_ptr<Tensor> &proj,
                                    size_t batch, size_t seq_len, size_t n_head,
                                    size_t head_dim) {
  return proj
      ->reshape({static_cast<int64_t>(batch), static_cast<int64_t>(seq_len),
                 static_cast<int64_t>(n_head), static_cast<int64_t>(head_dim)})
      ->permute({0, 2, 1, 3})
      ->reshape({static_cast<int64_t>(batch * n_head),
                 static_cast<int64_t>(seq_len),
                 static_cast<int64_t>(head_dim)});
}

std::shared_ptr<Tensor> merge_heads(const std::shared_ptr<Tensor> &heads,
                                    size_t batch, size_t seq_len, size_t n_head,
                                    size_t head_dim) {
  return heads
      ->reshape({static_cast<int64_t>(batch), static_cast<int64_t>(n_head),
                 static_cast<int64_t>(seq_len), static_cast<int64_t>(head_dim)})
      ->permute({0, 2, 1, 3})
      ->reshape({static_cast<int64_t>(batch), static_cast<int64_t>(seq_len),
                 static_cast<int64_t>(n_head * head_dim)});
}

}  // namespace

CausalSelfAttention::CausalSelfAttention(size_t n_embd, size_t n_head,
                                         scalar_t dropout)
    : n_embd_(n_embd), n_head_(n_head) {
  if (n_head == 0 || n_embd % n_head != 0) {
    throw std::invalid_argument(
        "CausalSelfAttention: n_embd must be divisible by n_head");
  }

  query_ = std::make_shared<Linear>(n_embd, n_embd);
  key_ = std::make_shared<Linear>(n_embd, n_embd);
  value_ = std::make_shared<Linear>(n_embd, n_embd);
  out_proj_ = std::make_shared<Linear>(n_embd, n_embd);
  attn_dropout_ = std::make_shared<Dropout>(dropout);
  resid_dropout_ = std::make_shared<Dropout>(dropout);

  register_module("query", query_);
  register_module("key", key_);
  register_module("value", value_);
  register_module("out_proj", out_proj_);
  register_module("attn_dropout", attn_dropout_);
  register_module("resid_dropout", resid_dropout_);
}

std::shared_ptr<Tensor> CausalSelfAttention::forward(
    const std::shared_ptr<Tensor> &input) {
  const std::vector<size_t> &shape = input->shape();
  if (shape.size() != 3) {
    throw std::invalid_argument(
        "CausalSelfAttention expects a (batch, seq, embed) input");
  }

  size_t batch = shape[0];
  size_t seq_len = shape[1];
  size_t head_dim = n_embd_ / n_head_;

  std::shared_ptr<Tensor> q =
      split_heads(query_->forward(input), batch, seq_len, n_head_, head_dim);
  std::shared_ptr<Tensor> k =
      split_heads(key_->forward(input), batch, seq_len, n_head_, head_dim);
  std::shared_ptr<Tensor> v =
      split_heads(value_->forward(input), batch, seq_len, n_head_, head_dim);

  scalar_t scale = 1.0f / std::sqrt(static_cast<scalar_t>(head_dim));
  std::shared_ptr<Tensor> scores =
      q->matmul(k->transpose(1, 2))
          ->mul(scale)
          ->add(causal_bias(seq_len, input->backend()));

  std::shared_ptr<Tensor> weights = attn_dropout_->forward(scores->softmax(-1));
  std::shared_ptr<Tensor> merged =
      merge_heads(weights->matmul(v), batch, seq_len, n_head_, head_dim);

  return resid_dropout_->forward(out_proj_->forward(merged));
}

}  // namespace micrograd::gpt
