#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

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

TEST(Optim, SgdMatchesBookRavine) {
  auto p = Tensor::from_values({2}, {1.0f, 1.0f});
  SGD optimizer({p}, 0.019f);

  for (int step = 0; step < 100; step++) {
    optimizer.zero_grad();
    SetRavineGrad(*p);
    optimizer.step();
  }

  EXPECT_NEAR(p->data()[0], 0.000027f, 1e-6f);
  EXPECT_NEAR(p->data()[1], 0.146859f, 1e-6f);
}

TEST(Optim, AdamMatchesAdamWWithoutDecay) {
  auto a = Tensor::from_values({2}, {1.0f, 1.0f});
  auto b = Tensor::from_values({2}, {1.0f, 1.0f});
  Adam adam({a}, 0.1f, {0.9f, 0.999f}, 1e-8f, 0.0f);
  AdamW adamw({b}, 0.1f, {0.9f, 0.999f}, 1e-8f, 0.0f);

  for (int step = 0; step < 100; step++) {
    adam.zero_grad();
    adamw.zero_grad();
    SetRavineGrad(*a);
    SetRavineGrad(*b);
    adam.step();
    adamw.step();
  }

  EXPECT_FLOAT_EQ(a->data()[0], b->data()[0]);
  EXPECT_FLOAT_EQ(a->data()[1], b->data()[1]);
}

TEST(Optim, AdamCouplesDecayIntoMoments) {
  auto w = Tensor::from_values({1}, {1.0f});
  Adam optimizer({w}, 0.1f, {0.9f, 0.999f}, 1e-8f, 0.01f);

  w->grad()[0] = 0.5f;
  optimizer.step();
  EXPECT_NEAR(w->data()[0], 0.900000f, 1e-6f);

  optimizer.zero_grad();
  w->grad()[0] = 0.5f;
  optimizer.step();
  EXPECT_NEAR(w->data()[0], 0.800005f, 1e-6f);
}

TEST(Optim, MuonOrthogonalizesSquareUpdate) {
  auto w = Tensor::zeros({2, 2});
  Muon optimizer({w}, 1.0f, 0.95f, 0.0f);

  w->grad()[0] = 3.0f;
  w->grad()[3] = 1.0f;
  optimizer.step();

  EXPECT_FLOAT_EQ(w->data()[1], 0.0f);
  EXPECT_FLOAT_EQ(w->data()[2], 0.0f);
  for (size_t i : {0u, 3u}) {
    EXPECT_GT(-w->data()[i], 0.6f);
    EXPECT_LT(-w->data()[i], 1.2f);
  }
}

TEST(Optim, MuonScalesTallUpdate) {
  auto w = Tensor::zeros({3, 2});
  Muon optimizer({w}, 1.0f, 0.95f, 0.0f);

  w->grad()[0] = 2.0f;
  w->grad()[3] = 1.0f;
  optimizer.step();

  const scalar_t scale = std::sqrt(1.5f);
  for (size_t i : {1u, 2u, 4u, 5u}) {
    EXPECT_FLOAT_EQ(w->data()[i], 0.0f);
  }
  for (size_t i : {0u, 3u}) {
    EXPECT_GT(-w->data()[i] / scale, 0.6f);
    EXPECT_LT(-w->data()[i] / scale, 1.2f);
  }
}

TEST(Optim, MuonRejectsNonMatrixParameters) {
  auto v = Tensor::from_values({2}, {1.0f, 1.0f});
  EXPECT_THROW(Muon({v}, 0.02f), std::invalid_argument);
}

TEST(Optim, RejectsDuplicateParameters) {
  auto p = Tensor::from_values({1}, {1.0f});
  EXPECT_THROW(SGD({p, p}, 0.1f), std::invalid_argument);
  EXPECT_THROW(AdamW({p, p}, 0.1f), std::invalid_argument);
}

TEST(Optim, RejectsNullParameter) {
  EXPECT_THROW(SGD({nullptr}, 0.1f), std::invalid_argument);
}

TEST(Optim, RejectsInvalidHyperparameters) {
  auto p = Tensor::from_values({1}, {1.0f});
  auto matrix = Tensor::from_values({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const scalar_t nan = std::numeric_limits<scalar_t>::quiet_NaN();

  EXPECT_THROW(SGD({p}, -0.1f), std::invalid_argument);
  EXPECT_THROW(SGD({p}, nan), std::invalid_argument);
  EXPECT_THROW(SGD({p}, 0.1f, -0.9f), std::invalid_argument);
  EXPECT_THROW(SGD({p}, 0.1f, 0.0f, -0.01f), std::invalid_argument);
  EXPECT_THROW(AdamW({p}, 0.1f, {1.0f, 0.999f}), std::invalid_argument);
  EXPECT_THROW(AdamW({p}, 0.1f, {0.9f, -0.1f}), std::invalid_argument);
  EXPECT_THROW(AdamW({p}, 0.1f, {0.9f, 0.999f}, -1e-8f), std::invalid_argument);
  EXPECT_THROW(Adam({p}, 0.1f, {0.9f, 1.0f}), std::invalid_argument);
  EXPECT_THROW(Adam({p}, 0.1f, {0.9f, 0.999f}, 1e-8f, -0.01f),
               std::invalid_argument);
  EXPECT_THROW(Muon({matrix}, -0.02f), std::invalid_argument);
}

TEST(Optim, StepsThroughBaseClass) {
  auto p = Tensor::from_values({1}, {1.0f});
  SGD sgd({p}, 0.1f);
  Optimizer &optimizer = sgd;

  p->grad()[0] = 1.0f;
  optimizer.step();
  EXPECT_FLOAT_EQ(p->data()[0], 0.9f);
  EXPECT_EQ(optimizer.step_count(), 1u);
  EXPECT_EQ(optimizer.parameters().size(), 1u);

  optimizer.zero_grad();
  EXPECT_FLOAT_EQ(p->grad()[0], 0.0f);
}
