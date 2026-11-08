#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "micrograd/nn/Layers.h"
#include "micrograd/nn/Stats.h"

using namespace micrograd;

TEST(Stats, LinearMatchesBookMemoryTable) {
  Linear layer(784, 256);
  EXPECT_EQ(param_count(layer), 200960u);
  EXPECT_EQ(param_bytes(layer), 803840u);
}

TEST(Stats, LinearWithoutBias) {
  Linear layer(784, 256, false);
  EXPECT_EQ(param_count(layer), 200704u);
  EXPECT_EQ(param_bytes(layer), 802816u);
}

TEST(Stats, ActivationHasNoParameters) {
  ReLU relu;
  EXPECT_EQ(param_count(relu), 0u);
  EXPECT_EQ(param_bytes(relu), 0u);
}

TEST(Stats, SequentialSumsAllLayers) {
  Sequential net({std::make_shared<Linear>(784, 128), std::make_shared<ReLU>(),
                  std::make_shared<Linear>(128, 64), std::make_shared<ReLU>(),
                  std::make_shared<Linear>(64, 10)});
  EXPECT_EQ(param_count(net), 109386u);
  EXPECT_EQ(param_bytes(net), 109386u * 4);
}

TEST(Stats, LinearFlopsMatchBook) {
  EXPECT_EQ(linear_flops(32, 784, 256), 12853248u);
  EXPECT_EQ(linear_flops(32, 784, 256, false), 12845056u);
  EXPECT_EQ(linear_flops(1, 784, 256), 401664u);
}
