#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "micrograd/Random.h"
#include "micrograd/nn/Layers.h"
#include "test_helpers.h"

using namespace micrograd;
using namespace micrograd::test_helpers;

namespace {

constexpr scalar_t kTolerance = 1e-5f;

std::shared_ptr<Tensor> Ones(std::vector<size_t> shape) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= dim;
  }
  return Make(std::move(shape), std::vector<scalar_t>(count, 1.0f));
}

void Fill(const std::shared_ptr<Tensor> &tensor,
          const std::vector<scalar_t> &values) {
  auto data = tensor->data();
  ASSERT_EQ(data.size(), values.size());
  std::ranges::copy(values, data.begin());
}

}  // namespace

TEST(Linear, ParameterShapes) {
  Linear layer(3, 2);
  EXPECT_EQ(layer.weights()->shape(), (std::vector<size_t>{3, 2}));
  EXPECT_EQ(layer.bias()->shape(), (std::vector<size_t>{1, 2}));
}

TEST(Linear, OutputShapeForBatch) {
  Linear layer(3, 2);
  auto output = layer.forward(Ones({4, 3}));
  EXPECT_EQ(output->shape(), (std::vector<size_t>{4, 2}));
}

TEST(Linear, ExactArithmetic) {
  Linear layer(3, 2);
  Fill(layer.weights(), {1, 2, 3, 4, 5, 6});
  Fill(layer.bias(), {0.5f, -1});

  auto input = Make({2, 3}, {1, 0, -1, 2, 1, 0});
  const auto values = Values(layer.forward(input));

  const std::vector<scalar_t> expected = {-3.5f, -5, 5.5f, 7};
  ASSERT_EQ(values.size(), expected.size());
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(values[i], expected[i], kTolerance) << "at index " << i;
  }
}

TEST(Linear, ParametersAreWeightThenBias) {
  Linear layer(3, 2);
  const auto params = layer.parameters();
  ASSERT_EQ(params.size(), 2u);
  EXPECT_EQ(params[0], layer.weights());
  EXPECT_EQ(params[1], layer.bias());
  EXPECT_TRUE(params[0]->requires_grad());
  EXPECT_TRUE(params[1]->requires_grad());

  const auto named = layer.named_parameters();
  ASSERT_EQ(named.size(), 2u);
  EXPECT_EQ(named[0].first, "weight");
  EXPECT_EQ(named[1].first, "bias");
}

TEST(Linear, BiasStartsAtZero) {
  Linear layer(16, 8);
  for (auto value : Values(layer.bias())) {
    EXPECT_EQ(value, 0.0f);
  }
}

TEST(Linear, LeCunNormalInit) {
  manual_seed(7);
  constexpr size_t kFanIn = 512;
  Linear layer(kFanIn, 512);

  const auto values = Values(layer.weights());

  double sum = 0.0;
  double sum_sq = 0.0;
  for (auto value : values) {
    sum += value;
    sum_sq += static_cast<double>(value) * value;
  }
  const auto n = static_cast<double>(values.size());
  const double mean = sum / n;
  const double variance = sum_sq / n - mean * mean;
  const double expected_variance = 1.0 / kFanIn;

  EXPECT_NEAR(mean, 0.0, 1e-3);
  EXPECT_NEAR(variance, expected_variance, 0.02 * expected_variance);
}

TEST(Linear, GradientsOfSummedOutput) {
  Linear layer(3, 2);
  auto input = Make({2, 3}, {1, 2, 3, 4, 5, 6});

  layer.forward(input)->sum()->backward();

  const std::vector<scalar_t> weight_grad(layer.weights()->grad().begin(),
                                          layer.weights()->grad().end());
  const std::vector<scalar_t> expected_weight = {5, 5, 7, 7, 9, 9};
  ASSERT_EQ(weight_grad.size(), expected_weight.size());
  for (size_t i = 0; i < expected_weight.size(); i++) {
    EXPECT_NEAR(weight_grad[i], expected_weight[i], kTolerance)
        << "at index " << i;
  }

  for (auto value : layer.bias()->grad()) {
    EXPECT_NEAR(value, 2.0f, kTolerance);
  }
}

TEST(Dropout, RejectsInvalidProbability) {
  EXPECT_THROW(Dropout(-0.1f), std::invalid_argument);
  EXPECT_THROW(Dropout(1.0f), std::invalid_argument);
  EXPECT_THROW(Dropout(1.5f), std::invalid_argument);
  EXPECT_NO_THROW(Dropout(0.0f));
  EXPECT_NO_THROW(Dropout(0.5f));
}

TEST(Dropout, HasNoParameters) {
  EXPECT_TRUE(Dropout(0.5f).parameters().empty());
}

TEST(Dropout, EvalIsIdentity) {
  Dropout dropout(0.5f);
  dropout.eval();
  auto input = Ones({4, 4});
  EXPECT_EQ(dropout.forward(input), input);
}

TEST(Dropout, ZeroProbabilityIsIdentity) {
  Dropout dropout(0.0f);
  auto input = Ones({4, 4});
  EXPECT_EQ(dropout.forward(input), input);
}

TEST(Dropout, MaskValuesAndSurvivalRate) {
  manual_seed(7);
  constexpr size_t kCount = 10000;
  constexpr scalar_t kP = 0.3f;
  Dropout dropout(kP);

  const auto values = Values(dropout.forward(Ones({kCount})));
  const scalar_t scale = 1.0f / (1.0f - kP);

  size_t kept = 0;
  for (auto value : values) {
    if (value == 0.0f) {
      continue;
    }
    EXPECT_NEAR(value, scale, kTolerance);
    kept++;
  }

  const double expected = kCount * (1.0 - kP);
  const double std_error = std::sqrt(kCount * kP * (1.0 - kP));
  EXPECT_NEAR(static_cast<double>(kept), expected, 3.0 * std_error);
}

TEST(Dropout, SameSeedSameMask) {
  Dropout dropout(0.5f);
  auto input = Ones({64});

  manual_seed(11);
  const auto first = Values(dropout.forward(input));
  manual_seed(11);
  const auto second = Values(dropout.forward(input));

  EXPECT_EQ(first, second);
}

TEST(Dropout, GradientIsTheMask) {
  manual_seed(3);
  Dropout dropout(0.5f);
  auto input = leaf({32}, std::vector<scalar_t>(32, 1.0f));

  auto output = dropout.forward(input);
  output->sum()->backward();

  const auto mask = Values(output);
  const std::vector<scalar_t> grad(input->grad().begin(), input->grad().end());
  ASSERT_EQ(grad.size(), mask.size());
  for (size_t i = 0; i < mask.size(); i++) {
    EXPECT_NEAR(grad[i], mask[i], kTolerance) << "at index " << i;
  }
}

TEST(Sequential, CollectsParametersInOrder) {
  auto first = std::make_shared<Linear>(4, 3);
  auto second = std::make_shared<Linear>(3, 2);
  Sequential net({first, std::make_shared<ReLU>(), second});

  const auto params = net.parameters();
  ASSERT_EQ(params.size(), 4u);
  EXPECT_EQ(params[0], first->weights());
  EXPECT_EQ(params[1], first->bias());
  EXPECT_EQ(params[2], second->weights());
  EXPECT_EQ(params[3], second->bias());

  size_t total = 0;
  for (const auto &param : params) {
    total += param->size();
  }
  EXPECT_EQ(total, 4u * 3 + 3 + 3 * 2 + 2);

  const auto named = net.named_parameters();
  ASSERT_EQ(named.size(), 4u);
  EXPECT_EQ(named[0].first, "0.weight");
  EXPECT_EQ(named[1].first, "0.bias");
  EXPECT_EQ(named[2].first, "2.weight");
  EXPECT_EQ(named[3].first, "2.bias");
}

TEST(Sequential, ChainsForward) {
  auto first = std::make_shared<Linear>(2, 2);
  auto second = std::make_shared<Linear>(2, 1);
  Fill(first->weights(), {1, -1, 1, -1});
  Fill(second->weights(), {1, 1});
  Sequential net({first, std::make_shared<ReLU>(), second});

  const auto values = Values(net.forward(Make({1, 2}, {1, 2})));
  ASSERT_EQ(values.size(), 1u);
  EXPECT_NEAR(values[0], 3.0f, kTolerance);
}

TEST(Mode, ModulesStartInTraining) {
  EXPECT_TRUE(Linear(2, 2).is_training());
  EXPECT_TRUE(Dropout(0.5f).is_training());
  EXPECT_TRUE(
      Sequential(std::vector<std::shared_ptr<nn::Module>>{}).is_training());
}

TEST(Mode, EvalAndTrainReachNestedChildren) {
  auto dropout = std::make_shared<Dropout>(0.5f);
  auto inner = std::make_shared<Sequential>(
      std::vector<std::shared_ptr<nn::Module>>{std::make_shared<Linear>(4, 4),
                                               dropout});
  Sequential outer({inner});

  outer.eval();
  EXPECT_FALSE(outer.is_training());
  EXPECT_FALSE(inner->is_training());
  EXPECT_FALSE(dropout->is_training());

  outer.train();
  EXPECT_TRUE(inner->is_training());
  EXPECT_TRUE(dropout->is_training());
}

TEST(Mode, EvalDisablesDropoutInsideSequential) {
  Sequential net({std::make_shared<Dropout>(0.9f)});
  auto input = Ones({256});

  net.eval();
  EXPECT_EQ(Values(net.forward(input)), Values(input));
}
