#include "micrograd/nn/Loss.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "micrograd/Storage.h"

namespace micrograd {
namespace {

constexpr scalar_t kProbabilityEpsilon = 1e-7f;

void RequireUnitInterval(const std::shared_ptr<Tensor> &tensor,
                         const char *message) {
  Storage host = tensor->data_storage().copy_to(Device::CPU);
  const auto *values = static_cast<const scalar_t *>(host.host_pointer());
  for (size_t i = 0; i < tensor->size(); i++) {
    if (!(values[i] >= 0.0f && values[i] <= 1.0f)) {
      throw std::invalid_argument(message);
    }
  }
}

}  // namespace

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
  const std::vector<size_t> &shape = logits->shape();
  if (shape.size() < 2) {
    throw std::invalid_argument("cross_entropy expects logits of rank 2 or more");
  }

  size_t rows = 1;
  for (size_t i = 0; i + 1 < shape.size(); i++) {
    rows *= shape[i];
  }
  size_t classes = shape.back();
  if (rows == 0) {
    throw std::invalid_argument("cross_entropy requires a nonempty batch");
  }
  if (target_indices.size() != rows) {
    throw std::invalid_argument(
        "cross_entropy target count does not match the batch size");
  }

  auto flat = shape.size() == 2
                  ? logits
                  : logits->reshape({static_cast<int64_t>(rows),
                                     static_cast<int64_t>(classes)});
  auto picked = flat->log_softmax(1)->gather_per_row(target_indices);

  return picked->sum()->neg()->div(static_cast<scalar_t>(rows));
}

std::shared_ptr<Tensor> binary_cross_entropy(
    const std::shared_ptr<Tensor> &predictions,
    const std::shared_ptr<Tensor> &targets) {
  if (predictions->shape() != targets->shape()) {
    throw std::invalid_argument(
        "binary_cross_entropy prediction and target shapes do not match");
  }
  if (predictions->size() == 0) {
    throw std::invalid_argument(
        "binary_cross_entropy requires a nonempty input");
  }
  RequireUnitInterval(predictions,
                      "binary_cross_entropy predictions must be in [0, 1]");
  RequireUnitInterval(targets,
                      "binary_cross_entropy targets must be in [0, 1]");

  auto probs =
      predictions->clamp(kProbabilityEpsilon, 1.0f - kProbabilityEpsilon);
  auto positive = targets->mul(probs->log());
  auto negative =
      targets->neg()->add(1.0f)->mul(probs->neg()->add(1.0f)->log());

  return positive->add(negative)->sum()->neg()->div(
      static_cast<scalar_t>(predictions->size()));
}

}  // namespace micrograd
