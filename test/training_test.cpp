#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "micrograd/NN.h"
#include "micrograd/Tensor.h"

using namespace micrograd;

TEST(Training, LinearSgdReducesLoss) {
  Linear layer(2, 1);
  SGD optimizer({layer.weights(), layer.bias()}, 0.1f);

  auto input = std::make_shared<Tensor>(std::vector<size_t>{1, 2},
                                        std::vector<scalar_t>{0.5, -0.5});
  auto target = std::make_shared<Tensor>(std::vector<size_t>{1, 1},
                                         std::vector<scalar_t>{1.0});

  const scalar_t before = mse_loss(layer.forward(input), target)->at({0});

  for (int step = 0; step < 50; step++) {
    auto loss = mse_loss(layer.forward(input), target);
    optimizer.zero_grad();
    loss->backward();
    optimizer.step();
  }

  const scalar_t after = mse_loss(layer.forward(input), target)->at({0});
  EXPECT_LT(after, before);
  EXPECT_NEAR(after, 0.0, 1e-3);
}
