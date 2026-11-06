#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "micrograd/Tensor.h"

namespace micrograd::test_helpers {

inline std::shared_ptr<Tensor> Make(std::vector<size_t> shape,
                                    std::vector<scalar_t> values) {
  return std::make_shared<Tensor>(std::move(shape), std::move(values));
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

inline auto leaf(const std::vector<size_t> &shape,
                 const std::vector<scalar_t> &values) {
  auto t = std::make_shared<Tensor>(shape, values);
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
