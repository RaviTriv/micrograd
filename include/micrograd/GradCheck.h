#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "micrograd/Backend.h"
#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"

namespace micrograd {

struct GradCheckOptions {
  scalar_t eps = 1e-2f;
  scalar_t rtol = 1e-3f;
  scalar_t atol = 1e-5f;
};

struct GradCheckMismatch {
  size_t input;
  size_t index;
  scalar_t analytic;
  scalar_t numeric;
};

using GradCheckFunction = std::function<std::shared_ptr<Tensor>(
    const std::vector<std::shared_ptr<Tensor>> &)>;

namespace internal {

inline std::vector<std::shared_ptr<Tensor>> MakeGradCheckLeaves(
    const std::vector<std::vector<size_t>> &shapes,
    const std::vector<std::vector<scalar_t>> &values) {
  std::vector<std::shared_ptr<Tensor>> leaves;
  leaves.reserve(shapes.size());
  for (size_t i = 0; i < shapes.size(); i++) {
    auto leaf = std::make_shared<Tensor>(shapes[i], values[i]);
    leaf->set_requires_grad(true);
    leaves.push_back(std::move(leaf));
  }
  return leaves;
}

inline std::vector<scalar_t> ProjectionWeights(size_t count) {
  std::vector<scalar_t> weights(count);
  for (size_t i = 0; i < count; i++) {
    weights[i] = std::sin(static_cast<scalar_t>(i) + 1.0f);
  }
  return weights;
}

inline scalar_t Project(const std::shared_ptr<Tensor> &output,
                        const std::vector<scalar_t> &weights) {
  if (output->size() != weights.size()) {
    throw std::invalid_argument(
        "gradcheck function returned a different output size");
  }

  output->to(Backend::CPU);
  std::span<const scalar_t> values = output->data();
  scalar_t total = 0.0f;
  for (size_t i = 0; i < weights.size(); i++) {
    total += weights[i] * values[i];
  }
  return total;
}

}  // namespace internal

inline std::vector<GradCheckMismatch> GradCheck(
    const GradCheckFunction &function,
    const std::vector<std::shared_ptr<Tensor>> &inputs,
    const GradCheckOptions &options = {}) {
  std::vector<std::vector<size_t>> shapes;
  std::vector<std::vector<scalar_t>> values;
  for (const auto &input : inputs) {
    std::span<const scalar_t> span = input->data();
    shapes.push_back(input->shape());
    values.emplace_back(span.begin(), span.end());
  }

  auto leaves = internal::MakeGradCheckLeaves(shapes, values);
  auto output = function(leaves);
  const std::vector<scalar_t> weights =
      internal::ProjectionWeights(output->size());

  const Tensor seed(output->shape(), weights);
  output->backward(seed);

  std::vector<std::vector<scalar_t>> analytic;
  analytic.reserve(leaves.size());
  for (const auto &leaf : leaves) {
    std::span<const scalar_t> span = leaf->grad();
    analytic.emplace_back(span.begin(), span.end());
  }

  std::vector<GradCheckMismatch> mismatches;
  for (size_t i = 0; i < values.size(); i++) {
    for (size_t j = 0; j < values[i].size(); j++) {
      const scalar_t original = values[i][j];

      values[i][j] = original + options.eps;
      const scalar_t up = internal::Project(
          function(internal::MakeGradCheckLeaves(shapes, values)), weights);

      values[i][j] = original - options.eps;
      const scalar_t down = internal::Project(
          function(internal::MakeGradCheckLeaves(shapes, values)), weights);

      values[i][j] = original;

      const scalar_t numeric = (up - down) / (2.0f * options.eps);
      const scalar_t difference = std::abs(analytic[i][j] - numeric);
      const scalar_t scale =
          std::max(std::abs(analytic[i][j]), std::abs(numeric));
      if (difference > options.atol + (options.rtol * scale)) {
        mismatches.push_back({.input = i,
                              .index = j,
                              .analytic = analytic[i][j],
                              .numeric = numeric});
      }
    }
  }

  return mismatches;
}

}  // namespace micrograd
