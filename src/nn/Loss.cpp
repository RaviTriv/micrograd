#include "micrograd/nn/Loss.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace micrograd {

std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor> &prediction,
                                 const std::shared_ptr<Tensor> &target) {
  auto diff = prediction->sub(target);
  auto squared = diff->pow(2.0f);

  auto sum = squared->sum();
  auto mean = sum->div(static_cast<scalar_t>(prediction->size()));

  return mean;
}

std::shared_ptr<Tensor> cross_entropy(
    const std::shared_ptr<Tensor> &logits,
    const std::vector<size_t> &target_indices) {
  if (logits->shape().size() != 2) {
    throw std::invalid_argument("cross_entropy expects rank 2 logits");
  }

  size_t batch = logits->shape()[0];
  size_t classes = logits->shape()[1];
  if (target_indices.size() != batch) {
    throw std::invalid_argument(
        "cross_entropy target count does not match the batch size");
  }

  std::vector<scalar_t> index_values(batch);
  for (size_t i = 0; i < batch; i++) {
    if (target_indices[i] >= classes) {
      throw std::out_of_range("cross_entropy target index is out of range");
    }
    index_values[i] = static_cast<scalar_t>(target_indices[i]);
  }
  auto indices =
      std::make_shared<Tensor>(std::vector<size_t>{batch}, index_values);
  indices->to(logits->backend());

  std::vector<scalar_t> diagonal(batch * batch, 0.0f);
  for (size_t i = 0; i < batch; i++) {
    diagonal[(i * batch) + i] = 1.0f;
  }
  auto diagonal_mask =
      std::make_shared<Tensor>(std::vector<size_t>{batch, batch}, diagonal);
  diagonal_mask->to(logits->backend());

  auto log_probs = logits->log_softmax(1);
  auto gathered = log_probs->transpose(0, 1)->embedding_lookup(indices);
  auto picked = gathered->mul(diagonal_mask);

  return picked->sum()->neg()->div(static_cast<scalar_t>(batch));
}

std::shared_ptr<Tensor> masked_cross_entropy(
    const std::shared_ptr<Tensor> &logits,
    const std::vector<size_t> &target_indices,
    const std::vector<bool> &loss_mask) {
  if (logits->shape().size() != 2) {
    throw std::invalid_argument("masked_cross_entropy expects rank 2 logits");
  }

  size_t batch = logits->shape()[0];
  size_t classes = logits->shape()[1];
  if (target_indices.size() != batch) {
    throw std::invalid_argument(
        "masked_cross_entropy target count does not match the batch size");
  }
  if (loss_mask.size() != batch) {
    throw std::invalid_argument(
        "masked_cross_entropy mask count does not match the batch size");
  }

  size_t masked_count = 0;
  for (bool masked : loss_mask) {
    if (masked) {
      masked_count++;
    }
  }
  if (masked_count == 0) {
    throw std::invalid_argument(
        "masked_cross_entropy requires at least one masked position");
  }

  std::vector<scalar_t> index_values(batch);
  for (size_t i = 0; i < batch; i++) {
    if (target_indices[i] >= classes) {
      throw std::out_of_range(
          "masked_cross_entropy target index is out of range");
    }
    index_values[i] = static_cast<scalar_t>(target_indices[i]);
  }
  auto indices =
      std::make_shared<Tensor>(std::vector<size_t>{batch}, index_values);
  indices->to(logits->backend());

  std::vector<scalar_t> diagonal(batch * batch, 0.0f);
  for (size_t i = 0; i < batch; i++) {
    if (loss_mask[i]) {
      diagonal[(i * batch) + i] = 1.0f;
    }
  }
  auto diagonal_mask =
      std::make_shared<Tensor>(std::vector<size_t>{batch, batch}, diagonal);
  diagonal_mask->to(logits->backend());

  auto log_probs = logits->log_softmax(1);
  auto gathered = log_probs->transpose(0, 1)->embedding_lookup(indices);
  auto picked = gathered->mul(diagonal_mask);

  return picked->sum()->neg()->div(static_cast<scalar_t>(masked_count));
}

}  // namespace micrograd
