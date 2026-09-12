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

}  // namespace micrograd::init
