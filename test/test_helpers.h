#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "micrograd/Tensor.h"

namespace micrograd::test_helpers {

inline std::shared_ptr<Tensor> Make(std::vector<size_t> shape,
                                    std::vector<scalar_t> values) {
  return Tensor::from_values(std::move(shape), std::move(values));
}

inline std::vector<scalar_t> Arange(size_t count) {
  std::vector<scalar_t> values(count);
  for (size_t i = 0; i < count; i++) {
    values[i] = static_cast<scalar_t>(i);
  }
  return values;
}

inline std::vector<scalar_t> Values(const std::shared_ptr<Tensor> &tensor) {
  auto data = tensor->data();
  return {data.begin(), data.end()};
}

inline std::shared_ptr<Tensor> Wave(std::vector<size_t> shape, scalar_t phase) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= dim;
  }
  std::vector<scalar_t> values(count);
  for (size_t i = 0; i < count; i++) {
    values[i] = std::sin((0.37f * static_cast<scalar_t>(i)) + phase);
  }
  auto tensor = Tensor::from_values(std::move(shape), std::move(values));
  tensor->set_requires_grad(true);
  return tensor;
}

inline void ExpectClose(std::span<const scalar_t> expected,
                        std::span<const scalar_t> actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(expected[i], actual[i], 1e-5f) << "index " << i;
  }
}

inline auto leaf(const std::vector<size_t> &shape,
                 const std::vector<scalar_t> &values) {
  auto t = Tensor::from_values(shape, values);
  t->set_requires_grad(true);
  return t;
}

inline auto scalar(scalar_t val) { return leaf({1}, {val}); }

inline auto vec(const std::vector<scalar_t> &values) {
  return leaf({values.size()}, values);
}

inline void expect_grad_matches_numeric(
    const std::vector<scalar_t> &values,
    const std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>)> &f) {
  auto x = vec(values);
  f(x)->backward();
  const std::vector<scalar_t> analytic(x->grad().begin(), x->grad().end());

  constexpr scalar_t h = 1e-2f;
  for (size_t i = 0; i < values.size(); i++) {
    auto shifted = values;

    shifted[i] = values[i] + h;
    const scalar_t up = f(vec(shifted))->data()[0];

    shifted[i] = values[i] - h;
    const scalar_t down = f(vec(shifted))->data()[0];

    EXPECT_NEAR(analytic[i], (up - down) / (2 * h), 1e-3) << "at index " << i;
  }
}

}  // namespace micrograd::test_helpers
