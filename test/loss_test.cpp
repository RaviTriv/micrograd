#include "micrograd/nn/Loss.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "test_helpers.h"

using namespace micrograd;
using namespace micrograd::test_helpers;

namespace {

constexpr scalar_t kTolerance = 1e-5f;
constexpr scalar_t kFloat32SpacingNear1000 = 6.1e-5f;

scalar_t Scalar(const std::shared_ptr<Tensor> &tensor) {
  return tensor->at({});
}

}  // namespace

TEST(Loss, MseKnownValue) {
  auto loss =
      mse_loss(Make({3}, {1.0f, 2.0f, 3.0f}), Make({3}, {1.5f, 2.5f, 2.8f}));
  EXPECT_NEAR(Scalar(loss), 0.18f, kTolerance);
}

TEST(Loss, MsePerfectPredictionIsZero) {
  auto values = std::vector<scalar_t>{0.5f, -1.0f, 2.0f, 4.0f};
  EXPECT_EQ(Scalar(mse_loss(Make({2, 2}, values), Make({2, 2}, values))), 0.0f);
}

TEST(Loss, MseRejectsMismatchedShapes) {
  auto prediction = Make({3, 1}, {1.0f, 2.0f, 3.0f});
  auto target = Make({3}, {1.0f, 2.0f, 3.0f});
  EXPECT_THROW(mse_loss(prediction, target), std::invalid_argument);
}

TEST(Loss, MseRejectsSameSizeDifferentShape) {
  auto prediction = Make({2, 3}, Arange(6));
  auto target = Make({3, 2}, Arange(6));
  EXPECT_THROW(mse_loss(prediction, target), std::invalid_argument);
}

TEST(Loss, MseRejectsEmptyInput) {
  EXPECT_THROW(mse_loss(Make({0}, {}), Make({0}, {})), std::invalid_argument);
}

TEST(Loss, CrossEntropyUniformLogitsGiveLogClassCount) {
  auto loss = cross_entropy(Make({1, 3}, {0.0f, 0.0f, 0.0f}), {2});
  EXPECT_NEAR(Scalar(loss), std::log(3.0f), kTolerance);
}

TEST(Loss, CrossEntropyIsStableOnLargeLogits) {
  auto loss = cross_entropy(Make({1, 3}, {1000.0f, 1002.0f, 1001.0f}), {1});
  const scalar_t value = Scalar(loss);
  EXPECT_TRUE(std::isfinite(value));
  EXPECT_NEAR(value, 0.4076059f, kFloat32SpacingNear1000);
}

TEST(Loss, CrossEntropyConfidentCorrectIsNearZero) {
  auto loss = cross_entropy(Make({1, 3}, {10.0f, 0.0f, 0.0f}), {0});
  EXPECT_LT(Scalar(loss), 0.01f);
  EXPECT_GE(Scalar(loss), 0.0f);
}

TEST(Loss, CrossEntropyConfidentWrongIsLarge) {
  auto loss = cross_entropy(Make({1, 3}, {10.0f, 0.0f, 0.0f}), {1});
  EXPECT_GT(Scalar(loss), 5.0f);
}

TEST(Loss, CrossEntropyAveragesOverBatch) {
  auto logits = Make({2, 3}, {0.0f, 0.0f, 0.0f, 1000.0f, 1002.0f, 1001.0f});
  auto loss = cross_entropy(logits, {0, 1});
  EXPECT_NEAR(Scalar(loss), (std::log(3.0f) + 0.4076059f) / 2.0f,
              kFloat32SpacingNear1000);
}

TEST(Loss, CrossEntropyRejectsTargetEqualToClassCount) {
  EXPECT_THROW(cross_entropy(Make({1, 3}, {0.0f, 0.0f, 0.0f}), {3}),
               std::out_of_range);
}

TEST(Loss, CrossEntropyRejectsTargetCountMismatch) {
  EXPECT_THROW(cross_entropy(Make({2, 3}, Arange(6)), {0}),
               std::invalid_argument);
}

TEST(Loss, CrossEntropyRejectsEmptyBatch) {
  EXPECT_THROW(cross_entropy(Make({0, 3}, {}), {}), std::invalid_argument);
}

TEST(Loss, CrossEntropyFlattensLeadingDimensions) {
  const std::vector<scalar_t> values = {0.0f,  1.0f, 2.0f, 3.0f, 1.0f, 0.5f,
                                        -1.0f, 2.0f, 0.0f, 4.0f, 4.0f, 4.0f};
  const std::vector<size_t> targets = {2, 0, 1, 1};
  auto rank3 = cross_entropy(Make({2, 2, 3}, values), targets);
  auto rank2 = cross_entropy(Make({4, 3}, values), targets);
  EXPECT_NEAR(Scalar(rank3), Scalar(rank2), kTolerance);
}

TEST(Loss, CrossEntropyRank3GradientMatchesRank2) {
  const std::vector<scalar_t> values = {0.0f,  1.0f, 2.0f, 3.0f, 1.0f, 0.5f,
                                        -1.0f, 2.0f, 0.0f, 4.0f, 4.0f, 4.0f};
  const std::vector<size_t> targets = {2, 0, 1, 1};
  auto rank3 = leaf({2, 2, 3}, values);
  auto rank2 = leaf({4, 3}, values);
  cross_entropy(rank3, targets)->backward();
  cross_entropy(rank2, targets)->backward();
  for (size_t i = 0; i < values.size(); i++) {
    EXPECT_NEAR(rank3->grad()[i], rank2->grad()[i], kTolerance)
        << "at index " << i;
  }
}

TEST(Loss, CrossEntropyRank3RejectsTargetCountMismatch) {
  EXPECT_THROW(cross_entropy(Make({2, 2, 3}, Arange(12)), {0, 1, 2}),
               std::invalid_argument);
}

TEST(Loss, CrossEntropyRejectsRankOneLogits) {
  EXPECT_THROW(cross_entropy(Make({3}, {0.0f, 0.0f, 0.0f}), {0}),
               std::invalid_argument);
}

TEST(Loss, BinaryCrossEntropyKnownValue) {
  auto loss = binary_cross_entropy(Make({4}, {0.9f, 0.1f, 0.7f, 0.3f}),
                                   Make({4}, {1.0f, 0.0f, 1.0f, 0.0f}));
  EXPECT_NEAR(Scalar(loss), -(std::log(0.9f) + std::log(0.7f)) / 2.0f,
              kTolerance);
}

TEST(Loss, BinaryCrossEntropyUniformPredictionGivesLogTwo) {
  auto loss =
      binary_cross_entropy(Make({2}, {0.5f, 0.5f}), Make({2}, {1.0f, 0.0f}));
  EXPECT_NEAR(Scalar(loss), std::log(2.0f), kTolerance);
}

TEST(Loss, BinaryCrossEntropyPerfectPredictionIsFiniteAndNearZero) {
  auto loss =
      binary_cross_entropy(Make({2}, {1.0f, 0.0f}), Make({2}, {1.0f, 0.0f}));
  EXPECT_TRUE(std::isfinite(Scalar(loss)));
  EXPECT_LT(Scalar(loss), 0.01f);
}

TEST(Loss, BinaryCrossEntropyWorstPredictionIsFiniteAndLarge) {
  auto loss =
      binary_cross_entropy(Make({2}, {0.0f, 1.0f}), Make({2}, {1.0f, 0.0f}));
  EXPECT_TRUE(std::isfinite(Scalar(loss)));
  EXPECT_GT(Scalar(loss), 5.0f);
}

TEST(Loss, BinaryCrossEntropyHasNoPredictionGradientWhereClamped) {
  auto predictions = leaf({2}, {0.0f, 1.0f});
  binary_cross_entropy(predictions, Make({2}, {1.0f, 0.0f}))->backward();
  EXPECT_EQ(predictions->grad()[0], 0.0f);
  EXPECT_EQ(predictions->grad()[1], 0.0f);
}

TEST(Loss, BinaryCrossEntropyRejectsMismatchedShapes) {
  EXPECT_THROW(
      binary_cross_entropy(Make({2, 1}, {0.5f, 0.5f}), Make({2}, {1.0f, 0.0f})),
      std::invalid_argument);
}

TEST(Loss, BinaryCrossEntropyRejectsEmptyInput) {
  EXPECT_THROW(binary_cross_entropy(Make({0}, {}), Make({0}, {})),
               std::invalid_argument);
}

TEST(Loss, BinaryCrossEntropyRejectsValuesOutsideUnitInterval) {
  auto targets = Make({2}, {1.0f, 0.0f});
  EXPECT_THROW(binary_cross_entropy(Make({2}, {1.5f, 0.5f}), targets),
               std::invalid_argument);
  EXPECT_THROW(
      binary_cross_entropy(Make({2}, {0.5f, 0.5f}), Make({2}, {-0.1f, 0.0f})),
      std::invalid_argument);
  EXPECT_THROW(binary_cross_entropy(Make({2}, {std::nanf(""), 0.5f}), targets),
               std::invalid_argument);
}

TEST(Loss, LogSoftmaxStaysFiniteAsLogitsGrow) {
  const std::vector<std::vector<scalar_t>> rows = {
      {1.0f, 2.0f, 3.0f},
      {10.0f, 20.0f, 30.0f},
      {100.0f, 200.0f, 300.0f},
      {500.0f, 600.0f, 700.0f},
  };
  for (const auto &row : rows) {
    auto log_probs = Values(Make({1, 3}, row)->log_softmax(1));
    scalar_t total = 0;
    for (scalar_t value : log_probs) {
      EXPECT_TRUE(std::isfinite(value)) << "row max " << row.back();
      total += std::exp(value);
    }
    EXPECT_NEAR(total, 1.0f, 1e-6f) << "row max " << row.back();
  }
}

TEST(Loss, FusedCrossEntropyMatchesComposed) {
  constexpr size_t kRows = 5;
  constexpr size_t kCols = 37;
  std::vector<scalar_t> values(kRows * kCols);
  for (size_t i = 0; i < values.size(); i++) {
    values[i] = 40.0f * std::sin(0.37f * static_cast<scalar_t>(i));
  }
  const std::vector<size_t> targets = {0, 36, 5, 17, 17};

  auto fused_in = Tensor::from_values({kRows, kCols}, values);
  fused_in->set_requires_grad(true);
  auto fused = fused_in->cross_entropy_rows(targets);
  fused->sum()->backward();

  auto composed_in = Tensor::from_values({kRows, kCols}, values);
  composed_in->set_requires_grad(true);
  auto composed = composed_in->log_softmax(1)->gather_per_row(targets)->neg();
  composed->sum()->backward();

  for (size_t i = 0; i < kRows; i++) {
    EXPECT_NEAR(fused->data()[i], composed->data()[i], 1e-4f) << "row " << i;
  }
  for (size_t i = 0; i < values.size(); i++) {
    EXPECT_NEAR(fused_in->grad()[i], composed_in->grad()[i], 1e-5f)
        << "element " << i;
  }
}

TEST(Loss, FusedCrossEntropyRejectsOutOfRangeTarget) {
  auto logits = Tensor::from_values({1, 3}, {0.0f, 1.0f, 2.0f});
  EXPECT_THROW(logits->cross_entropy_rows({3}), std::out_of_range);
}
