#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "micrograd/nn/Loss.h"
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
  auto loss = mse_loss(Make({3}, {1.0f, 2.0f, 3.0f}),
                       Make({3}, {1.5f, 2.5f, 2.8f}));
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

TEST(Loss, CrossEntropyRejectsNonMatrixLogits) {
  EXPECT_THROW(cross_entropy(Make({3}, {0.0f, 0.0f, 0.0f}), {0}),
               std::invalid_argument);
}

TEST(Loss, MaskedCrossEntropyIgnoresMaskedOutRows) {
  auto logits = Make({2, 3}, {0.0f, 0.0f, 0.0f, 10.0f, 0.0f, 0.0f});
  auto masked = masked_cross_entropy(logits, {0, 1}, {true, false});
  EXPECT_NEAR(Scalar(masked), std::log(3.0f), kTolerance);
}

TEST(Loss, MaskedCrossEntropyRejectsEmptyMask) {
  EXPECT_THROW(
      masked_cross_entropy(Make({1, 3}, {0.0f, 0.0f, 0.0f}), {0}, {false}),
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
