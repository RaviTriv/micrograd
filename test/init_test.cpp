#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "micrograd/Init.h"
#include "micrograd/Random.h"
#include "test_helpers.h"

using namespace micrograd;
using namespace micrograd::test_helpers;

namespace {

struct Moments {
  double mean;
  double variance;
};

Moments MomentsOf(const std::vector<scalar_t> &values) {
  double sum = 0.0;
  double sum_sq = 0.0;
  for (auto value : values) {
    sum += value;
    sum_sq += static_cast<double>(value) * value;
  }
  const auto n = static_cast<double>(values.size());
  const double mean = sum / n;
  return {.mean = mean, .variance = sum_sq / n - mean * mean};
}

}  // namespace

TEST(Init, NormalMatchesMeanAndStd) {
  manual_seed(7);
  auto tensor = Tensor::zeros({512, 512});
  init::normal_(tensor, 0.5f, 0.02f);

  const auto moments = MomentsOf(Values(tensor));
  EXPECT_NEAR(moments.mean, 0.5, 1e-3);
  EXPECT_NEAR(std::sqrt(moments.variance), 0.02, 0.02 * 0.01);
}

TEST(Init, NormalIsSeeded) {
  auto first = Tensor::zeros({64});
  auto second = Tensor::zeros({64});

  manual_seed(11);
  init::normal_(first, 0.0f, 1.0f);
  manual_seed(11);
  init::normal_(second, 0.0f, 1.0f);

  EXPECT_EQ(Values(first), Values(second));
}

TEST(Init, ZerosClearsEveryElement) {
  auto tensor = Make({2, 3}, {1, -2, 3, -4, 5, -6});
  init::zeros_(tensor);
  for (auto value : Values(tensor)) {
    EXPECT_EQ(value, 0.0f);
  }
}
