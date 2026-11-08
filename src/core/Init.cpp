#include "micrograd/Init.h"

#include <cmath>
#include <random>

#include "micrograd/Random.h"

namespace micrograd::init {

void kaiming_uniform_(const std::shared_ptr<Tensor> &tensor, size_t fan_in) {
  scalar_t bound = std::sqrt(6.0f / static_cast<scalar_t>(fan_in));
  std::uniform_real_distribution<scalar_t> dis(-bound, bound);
  for (auto &value : tensor->data()) {
    value = dis(global_rng());
  }
}

void xavier_uniform_(const std::shared_ptr<Tensor> &tensor, size_t fan_in,
                     size_t fan_out) {
  scalar_t bound = std::sqrt(6.0f / static_cast<scalar_t>(fan_in + fan_out));
  std::uniform_real_distribution<scalar_t> dis(-bound, bound);
  for (auto &value : tensor->data()) {
    value = dis(global_rng());
  }
}

void normal_(const std::shared_ptr<Tensor> &tensor, scalar_t mean,
             scalar_t std) {
  std::normal_distribution<scalar_t> dis(mean, std);
  for (auto &value : tensor->data()) {
    value = dis(global_rng());
  }
}

void zeros_(const std::shared_ptr<Tensor> &tensor) {
  for (auto &value : tensor->data()) {
    value = 0.0f;
  }
}

}  // namespace micrograd::init
