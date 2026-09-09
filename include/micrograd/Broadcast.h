#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "micrograd/Scalar.h"

namespace micrograd {

class Tensor;

std::vector<size_t> ContiguousStrides(const std::vector<size_t> &shape);

std::vector<size_t> BroadcastShapes(const std::vector<size_t> &a,
                                    const std::vector<size_t> &b);

std::vector<size_t> BroadcastStrides(const std::vector<size_t> &shape,
                                     const std::vector<size_t> &strides,
                                     const std::vector<size_t> &target);

void ReduceBroadcastGradient(std::span<const scalar_t> gradient,
                             const std::vector<size_t> &shape,
                             std::span<scalar_t> reduced,
                             const std::vector<size_t> &target);

std::shared_ptr<Tensor> BroadcastTo(const std::shared_ptr<Tensor> &tensor,
                                    const std::vector<size_t> &shape);

}  // namespace micrograd
