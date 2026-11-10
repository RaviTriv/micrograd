#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "micrograd/Tensor.h"

namespace micrograd {

std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor> &prediction,
                                 const std::shared_ptr<Tensor> &target);
std::shared_ptr<Tensor> cross_entropy(
    const std::shared_ptr<Tensor> &logits,
    const std::vector<size_t> &target_indices, size_t classes = 0);
std::shared_ptr<Tensor> binary_cross_entropy(
    const std::shared_ptr<Tensor> &predictions,
    const std::shared_ptr<Tensor> &targets);

}  // namespace micrograd
