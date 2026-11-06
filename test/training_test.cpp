#include <gtest/gtest.h>

#include <memory>
#include <type_traits>
#include <vector>

#include "micrograd/NN.h"
#include "micrograd/Tensor.h"

using namespace micrograd;

static_assert(!std::is_copy_constructible_v<Linear>);
static_assert(!std::is_copy_assignable_v<Linear>);
static_assert(!std::is_move_constructible_v<Linear>);
static_assert(!std::is_move_assignable_v<Linear>);
static_assert(!std::is_copy_constructible_v<Sequential>);
static_assert(!std::is_move_constructible_v<Sequential>);

TEST(Training, LinearSgdReducesLoss) {
  Linear layer(2, 1);
  SGD optimizer({layer.weights(), layer.bias()}, 0.1f);

  auto input = Tensor::from_values(std::vector<size_t>{1, 2},
                                   std::vector<scalar_t>{0.5, -0.5});
  auto target = Tensor::from_values(std::vector<size_t>{1, 1},
                                    std::vector<scalar_t>{1.0});

  const scalar_t before = mse_loss(layer.forward(input), target)->at({});

  for (int step = 0; step < 50; step++) {
    auto loss = mse_loss(layer.forward(input), target);
    optimizer.zero_grad();
    loss->backward();
    optimizer.step();
  }

  const scalar_t after = mse_loss(layer.forward(input), target)->at({});
  EXPECT_LT(after, before);
  EXPECT_NEAR(after, 0.0, 1e-3);
}

TEST(Training, LossesAreZeroDimScalars) {
  auto prediction = Tensor::from_values({1, 2}, {0.5f, -0.5f});
  auto target = Tensor::from_values({1, 2}, {1.0f, 1.0f});

  EXPECT_TRUE(mse_loss(prediction, target)->shape().empty());
  EXPECT_TRUE(cross_entropy(prediction, {1})->shape().empty());
}
