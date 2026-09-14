#include "examples/gpt/KVCache.h"

#include <algorithm>
#include <span>
#include <stdexcept>

namespace micrograd::gpt {

namespace {

void validate_step(const std::shared_ptr<Tensor> &step, size_t n_kv_head,
                   size_t head_dim) {
  const std::vector<size_t> &shape = step->shape();
  if (shape.size() != 3 || shape[0] != n_kv_head || shape[2] != head_dim) {
    throw std::invalid_argument("KVCache: step shape does not match cache");
  }
}

std::shared_ptr<Tensor> concat_seq(const std::shared_ptr<Tensor> &existing,
                                   const std::shared_ptr<Tensor> &step) {
  std::shared_ptr<Tensor> contiguous_step = step->contiguous();
  if (existing == nullptr) {
    return contiguous_step;
  }

  std::shared_ptr<Tensor> contiguous_existing = existing->contiguous();
  size_t n_kv_head = contiguous_existing->shape()[0];
  size_t old_len = contiguous_existing->shape()[1];
  size_t head_dim = contiguous_existing->shape()[2];
  size_t new_len = contiguous_step->shape()[1];
  size_t total_len = old_len + new_len;

  std::vector<scalar_t> merged(n_kv_head * total_len * head_dim);
  std::span<const scalar_t> old_data = contiguous_existing->data();
  std::span<const scalar_t> new_data = contiguous_step->data();

  for (size_t head = 0; head < n_kv_head; head++) {
    scalar_t *dest = merged.data() + (head * total_len * head_dim);
    const scalar_t *old_src = old_data.data() + (head * old_len * head_dim);
    const scalar_t *new_src = new_data.data() + (head * new_len * head_dim);
    std::copy(old_src, old_src + (old_len * head_dim), dest);
    std::copy(new_src, new_src + (new_len * head_dim),
              dest + (old_len * head_dim));
  }

  return std::make_shared<Tensor>(
      std::vector<size_t>{n_kv_head, total_len, head_dim}, std::move(merged));
}

}  // namespace

KVCache::KVCache(size_t n_layer, size_t n_kv_head, size_t head_dim)
    : n_kv_head_(n_kv_head),
      head_dim_(head_dim),
      keys_(n_layer),
      values_(n_layer) {
  if (n_layer == 0 || n_kv_head == 0 || head_dim == 0) {
    throw std::invalid_argument("KVCache: dimensions must be positive");
  }
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> KVCache::append(
    size_t layer, const std::shared_ptr<Tensor> &keys,
    const std::shared_ptr<Tensor> &values) {
  validate_step(keys, n_kv_head_, head_dim_);
  validate_step(values, n_kv_head_, head_dim_);

  keys_.at(layer) = concat_seq(keys_.at(layer), keys);
  values_.at(layer) = concat_seq(values_.at(layer), values);
  return {keys_[layer], values_[layer]};
}

size_t KVCache::length(size_t layer) const {
  return keys_.at(layer) ? keys_[layer]->shape()[1] : 0;
}

void KVCache::reset() {
  std::fill(keys_.begin(), keys_.end(), nullptr);
  std::fill(values_.begin(), values_.end(), nullptr);
}

}  // namespace micrograd::gpt
