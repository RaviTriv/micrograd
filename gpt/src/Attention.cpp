#include "gpt/Attention.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace micrograd::gpt {

namespace {

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

namespace {

std::shared_ptr<Tensor> BroadcastCausalMask(size_t batch_heads, size_t seq_len,
                                            Device device) {
  static std::shared_ptr<Tensor> cached;
  const std::vector<size_t> shape = {batch_heads, seq_len, seq_len};
  if (cached && cached->backend() == device && cached->shape() == shape) {
    return cached;
  }
  cached = Tensor::zeros(shape, device)->add(causal_mask(seq_len, device));
  return cached;
}

}  // namespace

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

  size_t batch = shape[0];
  size_t seq_len = shape[1];
  size_t head_dim = n_embd_ / n_head_;
  scalar_t scale = 1.0f / std::sqrt(static_cast<scalar_t>(head_dim));

  if (!arch_.rotary) {
    std::shared_ptr<Tensor> q = query_->forward(input);
    std::shared_ptr<Tensor> k = key_->forward(input);
    std::shared_ptr<Tensor> v = value_->forward(input);
    if (arch_.qk_norm) {
      const std::vector<int64_t> per_head = {
          static_cast<int64_t>(batch), static_cast<int64_t>(seq_len),
          static_cast<int64_t>(n_head_), static_cast<int64_t>(head_dim)};
      const std::vector<int64_t> merged = {static_cast<int64_t>(batch),
                                           static_cast<int64_t>(seq_len),
                                           static_cast<int64_t>(n_embd_)};
      q = q_norm_->forward(q->reshape(per_head))->reshape(merged);
      k = k_norm_->forward(k->reshape(per_head))->reshape(merged);
    }
    std::shared_ptr<Tensor> attended =
        q->causal_self_attention(k, v, n_head_, scale);
    return resid_dropout_->forward(out_proj_->forward(attended));
  }

  std::shared_ptr<Tensor> q =
      split_heads(query_->forward(input), batch, seq_len, n_head_, head_dim);
  std::shared_ptr<Tensor> k =
      split_heads(key_->forward(input), batch, seq_len, n_head_, head_dim);
  std::shared_ptr<Tensor> v =
      split_heads(value_->forward(input), batch, seq_len, n_head_, head_dim);

  if (arch_.qk_norm) {
    q = q_norm_->forward(q);
    k = k_norm_->forward(k);
  }
  if (arch_.rotary) {
    q = q->rotary_embedding();
    k = k->rotary_embedding();
  }

  const auto mask = BroadcastCausalMask(batch * n_head_, seq_len, q->backend());
  std::shared_ptr<Tensor> attended = causal_attention(q, k, v, scale, mask);
  std::shared_ptr<Tensor> merged =
      merge_heads(attended, batch, seq_len, n_head_, head_dim);

  return resid_dropout_->forward(out_proj_->forward(merged));
}

}  // namespace micrograd::gpt
