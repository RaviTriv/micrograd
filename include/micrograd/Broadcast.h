#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace micrograd {

std::vector<size_t> ContiguousStrides(const std::vector<size_t> &shape);

size_t NormalizeDim(int64_t dim, size_t rank);

std::vector<size_t> BroadcastShapes(const std::vector<size_t> &a,
                                    const std::vector<size_t> &b);

std::vector<size_t> BroadcastStrides(const std::vector<size_t> &shape,
                                     const std::vector<size_t> &strides,
                                     const std::vector<size_t> &target);

}  // namespace micrograd
