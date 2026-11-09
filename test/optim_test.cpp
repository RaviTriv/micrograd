#include <gtest/gtest.h>

#include <memory>

#include "micrograd/Tensor.h"
#include "micrograd/nn/Optim.h"

using namespace micrograd;

namespace {

void SetRavineGrad(Tensor &p) {
  p.grad()[0] = 100.0f * p.data()[0];
  p.grad()[1] = p.data()[1];
}

}  // namespace

TEST(Optim, AdamWMatchesBookTrace) {
  auto w = Tensor::from_values({1}, {1.0f});
  AdamW optimizer({w}, 0.1f, {0.9f, 0.999f}, 1e-8f, 0.01f);

  w->grad()[0] = 0.5f;
  optimizer.step();
  EXPECT_NEAR(w->data()[0], 0.899000f, 1e-6f);

  optimizer.zero_grad();
  w->grad()[0] = 0.5f;
  optimizer.step();
  EXPECT_NEAR(w->data()[0], 0.798101f, 1e-6f);
}

TEST(Optim, SgdMomentumMatchesBookRavine) {
  auto p = Tensor::from_values({2}, {1.0f, 1.0f});
  SGD optimizer({p}, 0.019f, 0.9f);

  for (int step = 0; step < 100; step++) {
    optimizer.zero_grad();
    SetRavineGrad(*p);
    optimizer.step();
  }

  EXPECT_NEAR(p->data()[0], 0.005154f, 1e-6f);
  EXPECT_NEAR(p->data()[1], 0.005234f, 1e-6f);
}

TEST(Optim, AdamWMatchesBookRavine) {
  auto p = Tensor::from_values({2}, {1.0f, 1.0f});
  AdamW optimizer({p}, 0.1f, {0.9f, 0.999f}, 1e-8f, 0.0f);

  for (int step = 0; step < 100; step++) {
    optimizer.zero_grad();
    SetRavineGrad(*p);
    optimizer.step();
  }

  EXPECT_NEAR(p->data()[0], 0.002937f, 1e-6f);
  EXPECT_NEAR(p->data()[1], 0.002937f, 1e-6f);
}

TEST(Optim, AdamWKeepsWeightsWrittenAfterConstruction) {
  auto w = Tensor::from_values({1}, {1.0f});
  AdamW optimizer({w}, 0.1f, {0.9f, 0.999f}, 1e-8f, 0.0f);

  w->data()[0] = 5.0f;
  w->grad()[0] = 0.0f;
  optimizer.step();

  EXPECT_FLOAT_EQ(w->data()[0], 5.0f);
}
