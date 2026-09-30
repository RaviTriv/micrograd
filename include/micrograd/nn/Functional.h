#pragma once

#include <cstddef>
#include <memory>

#include "micrograd/Tensor.h"

namespace micrograd {

std::shared_ptr<Tensor> avg_pool2d(const std::shared_ptr<Tensor> &input,
                                   size_t kernel);
std::shared_ptr<Tensor> gelu(const std::shared_ptr<Tensor> &input);
std::shared_ptr<Tensor> relu_squared(const std::shared_ptr<Tensor> &input);
std::shared_ptr<Tensor> qk_norm(const std::shared_ptr<Tensor> &input,
                                scalar_t eps = 1e-5f);

}  // namespace micrograd
