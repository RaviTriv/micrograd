#include "micrograd/nn/Loss.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace micrograd {

std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor> &prediction,
                                 const std::shared_ptr<Tensor> &target) {
  if (prediction->shape() != target->shape()) {
    throw std::invalid_argument(
        "mse_loss prediction and target shapes do not match");
  }
  if (prediction->size() == 0) {
    throw std::invalid_argument("mse_loss requires a nonempty input");
  }

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
  if (batch == 0) {
    throw std::invalid_argument("cross_entropy requires a nonempty batch");
  }
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
  auto indices = Tensor::from_values(std::vector<size_t>{batch}, index_values);
  indices->to(logits->backend());

  auto log_probs = logits->log_softmax(1);
  auto picked = log_probs->gather_per_row(indices);

  return picked->sum()->neg()->div(static_cast<scalar_t>(batch));
}

}  // namespace micrograd
