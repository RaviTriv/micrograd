#include <gtest/gtest.h>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include "micrograd/nn/Functional.h"
#include "micrograd/nn/Layers.h"
#include "test_helpers.h"

using namespace micrograd;
using namespace micrograd::test_helpers;

namespace {

constexpr scalar_t kInf = std::numeric_limits<scalar_t>::infinity();
constexpr scalar_t kNaN = std::numeric_limits<scalar_t>::quiet_NaN();
constexpr scalar_t kTolerance = 1e-5f;

void ExpectValues(const std::shared_ptr<Tensor> &tensor,
                  const std::vector<scalar_t> &expected) {
  const auto values = Values(tensor);
  ASSERT_EQ(values.size(), expected.size());
  for (size_t i = 0; i < values.size(); i++) {
    EXPECT_NEAR(values[i], expected[i], kTolerance) << "at index " << i;
  }
}

scalar_t Sum(const std::vector<scalar_t> &values) {
  scalar_t total = 0;
  for (scalar_t value : values) {
    total += value;
  }
  return total;
}

}  // namespace

TEST(Activations, SigmoidKnownValues) {
  Sigmoid sigmoid;
  ExpectValues(sigmoid.forward(Make({1}, {0})), {0.5f});
  ExpectValues(sigmoid.forward(Make({5}, {-3, -1, 0, 1, 3})),
               {0.0474259f, 0.2689414f, 0.5f, 0.7310586f, 0.9525741f});
}

TEST(Activations, SigmoidStaysInOpenUnitInterval) {
  Sigmoid sigmoid;
  for (scalar_t value :
       Values(sigmoid.forward(Make({5}, {-10, -1, 0, 1, 10})))) {
    EXPECT_GT(value, 0.0f);
    EXPECT_LT(value, 1.0f);
  }
}

TEST(Activations, SigmoidSaturatesOnExtremeInputs) {
  Sigmoid sigmoid;
  ExpectValues(sigmoid.forward(Make({2}, {-1000, 1000})), {0.0f, 1.0f});
}

TEST(Activations, SigmoidDoesNotOverflowOnExtremeInputs) {
  Sigmoid sigmoid;
  std::feclearexcept(FE_ALL_EXCEPT);
  auto result = sigmoid.forward(Make({2}, {-1000, 1000}));
  EXPECT_EQ(std::fetestexcept(FE_OVERFLOW | FE_INVALID), 0);
  ExpectValues(result, {0.0f, 1.0f});
}

TEST(Activations, SigmoidKeepsSubnormalOutputs) {
  Sigmoid sigmoid;
  const auto values = Values(sigmoid.forward(Make({1}, {-95})));
  EXPECT_GT(values[0], 0.0f);
  EXPECT_NEAR(values[0], 5.5210823e-42f, 1e-44f);
}

TEST(Activations, SigmoidPropagatesNaN) {
  Sigmoid sigmoid;
  EXPECT_TRUE(std::isnan(Values(sigmoid.forward(Make({1}, {kNaN})))[0]));
}

TEST(Activations, ReluKnownValues) {
  ReLU relu;
  ExpectValues(relu.forward(Make({5}, {-2, -1, 0, 1, 2})), {0, 0, 0, 1, 2});
  ExpectValues(relu.forward(Make({3}, {-5, -3, -1})), {0, 0, 0});
  ExpectValues(relu.forward(Make({3}, {1, 3, 5})), {1, 3, 5});
}

TEST(Activations, ReluCreatesSparsity) {
  ReLU relu;
  const auto values = Values(relu.forward(Make({4}, {-1, -2, -3, 1})));
  EXPECT_EQ(std::ranges::count(values, 0.0f), 3);
}

TEST(Activations, ReluPropagatesNaN) {
  ReLU relu;
  const auto values = Values(relu.forward(Make({3}, {kNaN, -1, 2})));
  EXPECT_TRUE(std::isnan(values[0]));
  EXPECT_EQ(values[1], 0.0f);
  EXPECT_EQ(values[2], 2.0f);
}

TEST(Activations, ReluMapsNegativeZeroToPositiveZero) {
  ReLU relu;
  const auto values = Values(relu.forward(Make({1}, {-0.0f})));
  EXPECT_EQ(values[0], 0.0f);
  EXPECT_FALSE(std::signbit(values[0]));
}

TEST(Activations, TanhKnownValues) {
  Tanh tanh;
  ExpectValues(tanh.forward(Make({1}, {0})), {0.0f});
  ExpectValues(tanh.forward(Make({2}, {-1000, 1000})), {-1.0f, 1.0f});
}

TEST(Activations, TanhStaysInClosedUnitRange) {
  Tanh tanh;
  for (scalar_t value : Values(tanh.forward(Make({5}, {-10, -1, 0, 1, 10})))) {
    EXPECT_GE(value, -1.0f);
    EXPECT_LE(value, 1.0f);
  }
}

TEST(Activations, TanhIsOdd) {
  Tanh tanh;
  const auto positive = Values(tanh.forward(Make({1}, {2})));
  const auto negative = Values(tanh.forward(Make({1}, {-2})));
  EXPECT_NEAR(positive[0], -negative[0], kTolerance);
}

TEST(Activations, GeluKnownValues) {
  GELU gelu;
  ExpectValues(gelu.forward(Make({3}, {-1, 0, 1})),
               {-0.158808f, 0.0f, 0.841192f});
}

TEST(Activations, GeluIsContinuousAroundZero) {
  GELU gelu;
  const auto values = Values(gelu.forward(Make({3}, {-0.001f, 0, 0.001f})));
  EXPECT_LT(std::fabs(values[1] - values[0]), 0.01f);
  EXPECT_LT(std::fabs(values[2] - values[1]), 0.01f);
}

TEST(Activations, GeluSaturatesOnLargeInputs) {
  GELU gelu;
  ExpectValues(gelu.forward(Make({2}, {-1e6f, 1e3f})), {0.0f, 1e3f});
}

TEST(Activations, SoftmaxKnownValues) {
  Softmax softmax;
  ExpectValues(softmax.forward(Make({3}, {1, 2, 3})),
               {0.0900306f, 0.2447285f, 0.6652409f});
}

TEST(Activations, SoftmaxIsStableOnLargeInputs) {
  Softmax softmax;
  ExpectValues(softmax.forward(Make({3}, {1000, 1001, 1002})),
               {0.0900306f, 0.2447285f, 0.6652409f});
}

TEST(Activations, SoftmaxDefaultsToLastDimension) {
  Softmax softmax;
  auto result = softmax.forward(Make({2, 2}, {1, 2, 3, 4}));
  EXPECT_EQ(result->shape(), (std::vector<size_t>{2, 2}));
  ExpectValues(result, {0.2689414f, 0.7310586f, 0.2689414f, 0.7310586f});
}

TEST(Activations, SoftmaxUsesConfiguredDimension) {
  Softmax softmax(0);
  ExpectValues(softmax.forward(Make({2, 2}, {1, 2, 2, 4})),
               {0.2689414f, 0.1192029f, 0.7310586f, 0.8807971f});
}

TEST(Activations, SoftmaxSumsToOneAcrossLastDimensionIn3D) {
  Softmax softmax;
  auto result =
      softmax.forward(Make({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
  const auto values = Values(result);
  EXPECT_EQ(result->shape(), (std::vector<size_t>{2, 2, 3}));
  for (size_t row = 0; row < 4; row++) {
    EXPECT_NEAR(values[row * 3] + values[(row * 3) + 1] + values[(row * 3) + 2],
                1.0f, kTolerance);
  }
}

TEST(Activations, SoftmaxHandlesPartialMask) {
  Softmax softmax;
  ExpectValues(softmax.forward(Make({3}, {-kInf, 1, 2})),
               {0.0f, 0.2689414f, 0.7310586f});
}

TEST(Activations, SoftmaxReturnsZerosForFullyMaskedSlice) {
  Softmax softmax;
  std::feclearexcept(FE_ALL_EXCEPT);
  auto result = softmax.forward(Make({2, 2}, {-kInf, -kInf, 0, -kInf}));
  EXPECT_EQ(std::fetestexcept(FE_OVERFLOW | FE_INVALID | FE_DIVBYZERO), 0);
  ExpectValues(result, {0.0f, 0.0f, 1.0f, 0.0f});
}

TEST(Activations, SoftmaxFullyMaskedSliceHasZeroGradient) {
  auto x = leaf({2, 2}, {-kInf, -kInf, 0.5f, 1.5f});
  auto upstream = Make({2, 2}, {1, 2, 3, 4});
  x->softmax(-1)->mul(upstream)->sum()->backward();
  const std::vector<scalar_t> gradient(x->grad().begin(), x->grad().end());
  EXPECT_EQ(gradient[0], 0.0f);
  EXPECT_EQ(gradient[1], 0.0f);
  EXPECT_TRUE(std::isfinite(gradient[2]));
  EXPECT_TRUE(std::isfinite(gradient[3]));
}

TEST(Activations, ModulesHaveNoParameters) {
  EXPECT_TRUE(ReLU().parameters().empty());
  EXPECT_TRUE(Sigmoid().parameters().empty());
  EXPECT_TRUE(Tanh().parameters().empty());
  EXPECT_TRUE(GELU().parameters().empty());
  EXPECT_TRUE(Softmax().parameters().empty());
}

TEST(Activations, ModulesPreserveShape) {
  auto input = Make({2, 2}, {1, -1, 2, -2});
  EXPECT_EQ(ReLU().forward(input)->shape(), input->shape());
  EXPECT_EQ(Sigmoid().forward(input)->shape(), input->shape());
  EXPECT_EQ(Tanh().forward(input)->shape(), input->shape());
  EXPECT_EQ(GELU().forward(input)->shape(), input->shape());
  EXPECT_EQ(Softmax().forward(input)->shape(), input->shape());
}

TEST(Activations, ModulesComposeInSequential) {
  Sequential net({std::make_shared<ReLU>(), std::make_shared<Softmax>()});
  const auto values = Values(net.forward(Make({1, 4}, {-1, 0, 1, 2})));
  EXPECT_NEAR(Sum(values), 1.0f, kTolerance);
}

TEST(Activations, ModulesAgreeWithTensorMethods) {
  auto input = Make({3}, {-1.5f, 0.25f, 2.0f});
  EXPECT_EQ(Values(ReLU().forward(input)), Values(input->relu()));
  EXPECT_EQ(Values(Sigmoid().forward(input)), Values(input->sigmoid()));
  EXPECT_EQ(Values(Tanh().forward(input)), Values(input->tanh()));
  EXPECT_EQ(Values(GELU().forward(input)), Values(micrograd::gelu(input)));
  EXPECT_EQ(Values(Softmax().forward(input)), Values(input->softmax(-1)));
}

TEST(Activations, ModulesRecordGradients) {
  auto x = vec({-1.5f, 0.25f, 2.0f});
  GELU().forward(x)->sum()->backward();
  ASSERT_TRUE(x->has_grad());
  for (scalar_t gradient : x->grad()) {
    EXPECT_TRUE(std::isfinite(gradient));
  }
}

TEST(Activations, NumericGradGelu) {
  expect_grad_matches_numeric({-1.5f, -0.25f, 0.75f, 2.0f}, [](const auto &x) {
    return micrograd::gelu(x)->sum();
  });
}

TEST(Activations, NumericGradSoftmaxWithNonUniformUpstream) {
  auto upstream = Make({4}, {1, 0, 0, 0});
  expect_grad_matches_numeric({-1.5f, -0.25f, 0.75f, 2.0f},
                              [upstream](const auto &x) {
                                return x->softmax(-1)->mul(upstream)->sum();
                              });
}
