#include "micrograd/nn/Optim.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Optimizer.h"

using namespace micrograd;

namespace {

void SetRavineGrad(Tensor &p) {
  p.grad()[0] = 100.0f * p.data()[0];
  p.grad()[1] = p.data()[1];
}

std::shared_ptr<Tensor> Pattern(std::vector<size_t> shape, size_t seed) {
  size_t n = 1;
  for (size_t d : shape) {
    n *= d;
  }
  std::vector<scalar_t> values(n);
  for (size_t i = 0; i < n; i++) {
    values[i] = 0.1f * static_cast<scalar_t>((i * 37 + seed * 11) % 13) - 0.6f;
  }
  return Tensor::from_values(std::move(shape), std::move(values));
}

std::shared_ptr<Tensor> Copy(const Tensor &t) {
  return Tensor::from_values(
      t.shape(), std::vector<scalar_t>(t.data().begin(), t.data().end()));
}

void FillGrad(Tensor &p, size_t seed) {
  auto grad = Pattern(p.shape(), seed);
  std::ranges::copy(grad->data(), p.grad().begin());
}

void ExpectBitwiseEqual(const Tensor &a, const Tensor &b) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); i++) {
    EXPECT_EQ(std::bit_cast<uint32_t>(a.data()[i]),
              std::bit_cast<uint32_t>(b.data()[i]))
        << "index " << i;
  }
}

void ExpectMuonScale(size_t fan_in, size_t fan_out) {
  SCOPED_TRACE(testing::Message() << fan_in << "x" << fan_out);
  auto w = Pattern({fan_in, fan_out}, 1);
  auto ref = Copy(*w);
  auto buf = Tensor::zeros({fan_in, fan_out});
  Muon optimizer({w}, 0.02f, 0.95f, 0.01f);

  FillGrad(*w, 2);
  FillGrad(*ref, 2);
  optimizer.step();
  const scalar_t scale = std::sqrt(std::max(
      1.0f, static_cast<scalar_t>(fan_out) / static_cast<scalar_t>(fan_in)));
  ops::MuonStep(*ref, *buf, 0.02f, 0.95f, 0.01f, true, 5, scale);

  ExpectBitwiseEqual(*w, *ref);
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

TEST(Optim, MuonScalesWideUpdate) {
  auto w = Tensor::zeros({2, 3});
  Muon optimizer({w}, 1.0f, 0.95f, 0.0f);

  w->grad()[0] = 2.0f;
  w->grad()[4] = 1.0f;
  optimizer.step();

  const scalar_t scale = std::sqrt(1.5f);
  for (size_t i : {1u, 2u, 3u, 5u}) {
    EXPECT_FLOAT_EQ(w->data()[i], 0.0f);
  }
  for (size_t i : {0u, 4u}) {
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

TEST(Optim, SetLearningRateOnEveryOptimizer) {
  auto p = Tensor::from_values({1}, {1.0f});
  auto matrix = Tensor::zeros({2, 2});
  SGD sgd({p}, 0.1f);
  Adam adam({Tensor::from_values({1}, {1.0f})}, 0.1f);
  AdamW adamw({Tensor::from_values({1}, {1.0f})}, 0.1f);
  Muon muon({matrix}, 0.1f);

  for (Optimizer *optimizer :
       std::vector<Optimizer *>{&sgd, &adam, &adamw, &muon}) {
    EXPECT_FLOAT_EQ(optimizer->learning_rate(), 0.1f);
    optimizer->set_learning_rate(0.5f);
    EXPECT_FLOAT_EQ(optimizer->learning_rate(), 0.5f);
    EXPECT_THROW(optimizer->set_learning_rate(-1.0f), std::invalid_argument);
    EXPECT_THROW(optimizer->set_learning_rate(
                     std::numeric_limits<scalar_t>::quiet_NaN()),
                 std::invalid_argument);
    EXPECT_FLOAT_EQ(optimizer->learning_rate(), 0.5f);
  }

  p->grad()[0] = 1.0f;
  sgd.step();
  EXPECT_FLOAT_EQ(p->data()[0], 0.5f);
}

TEST(Optim, AdamWLearningRateChangeKeepsMomentsAndStepCount) {
  auto w = Pattern({2, 3}, 0);
  auto ref = Copy(*w);
  auto m = Tensor::zeros({2, 3});
  auto v = Tensor::zeros({2, 3});
  AdamW optimizer({w}, 0.1f, {0.9f, 0.999f}, 1e-8f, 0.01f);

  const scalar_t lrs[] = {0.1f, 0.1f, 0.02f, 0.02f};
  for (size_t t = 0; t < 4; t++) {
    optimizer.set_learning_rate(lrs[t]);
    FillGrad(*w, t + 1);
    FillGrad(*ref, t + 1);
    optimizer.step();
    const auto step = static_cast<scalar_t>(t + 1);
    ops::AdamWStep(*ref, *m, *v, lrs[t], 0.9f, 0.999f, 1e-8f, 0.01f, true,
                   1.0f - std::pow(0.9f, step), 1.0f - std::pow(0.999f, step));
  }

  EXPECT_EQ(optimizer.step_count(), 4u);
  ExpectBitwiseEqual(*w, *ref);
}

TEST(Optim, AdamWSkipsDecayForFlaggedParameters) {
  auto decayed = Tensor::from_values({2}, {1.0f, -2.0f});
  auto kept = Tensor::from_values({2}, {1.0f, -2.0f});
  AdamW optimizer({decayed, kept}, {true, false}, 0.1f, {0.9f, 0.999f}, 1e-8f,
                  0.1f);

  optimizer.step();

  EXPECT_FLOAT_EQ(decayed->data()[0], 0.99f);
  EXPECT_FLOAT_EQ(decayed->data()[1], -1.98f);
  EXPECT_EQ(kept->data()[0], 1.0f);
  EXPECT_EQ(kept->data()[1], -2.0f);
}

TEST(Optim, AdamWRejectsDecayFlagCountMismatch) {
  auto p = Tensor::from_values({1}, {1.0f});
  EXPECT_THROW(AdamW({p}, {true, false}, 0.1f), std::invalid_argument);
}

TEST(Optim, AdamWRejectsNonFloat32Parameters) {
  auto p = Tensor::zeros({2}, Device::CPU, DType::kBFloat16);
  AdamW optimizer({p}, 0.1f);
  EXPECT_THROW(optimizer.step(), std::invalid_argument);
  EXPECT_EQ(optimizer.step_count(), 0u);
}

TEST(Optim, MuonScaleUsesFanOutOverFanIn) {
  ExpectMuonScale(4, 16);
  ExpectMuonScale(16, 4);
}

TEST(Optim, AdamWStateRoundTripGivesIdenticalNextStep) {
  auto w = Pattern({2, 3}, 0);
  auto b = Pattern({3}, 1);
  AdamW first({w, b}, {true, false}, 0.05f, {0.9f, 0.999f}, 1e-8f, 0.1f);
  for (size_t t = 0; t < 3; t++) {
    FillGrad(*w, t + 2);
    FillGrad(*b, t + 5);
    first.step();
  }

  auto w2 = Copy(*w);
  auto b2 = Copy(*b);
  AdamW second({w2, b2}, {true, false}, 0.05f, {0.9f, 0.999f}, 1e-8f, 0.1f);
  second.load_state(first.state());
  EXPECT_EQ(second.step_count(), 3u);

  for (auto *p : {w.get(), w2.get()}) {
    FillGrad(*p, 9);
  }
  for (auto *p : {b.get(), b2.get()}) {
    FillGrad(*p, 10);
  }
  first.step();
  second.step();

  ExpectBitwiseEqual(*w, *w2);
  ExpectBitwiseEqual(*b, *b2);
}

TEST(Optim, MuonStateRoundTripGivesIdenticalNextStep) {
  auto a = Pattern({3, 5}, 0);
  auto c = Pattern({4, 2}, 1);
  Muon first({a, c}, 0.02f);
  for (size_t t = 0; t < 3; t++) {
    FillGrad(*a, t + 2);
    FillGrad(*c, t + 5);
    first.step();
  }

  auto a2 = Copy(*a);
  auto c2 = Copy(*c);
  Muon second({a2, c2}, 0.02f);
  second.load_state(first.state());
  EXPECT_EQ(second.step_count(), 3u);

  for (auto *p : {a.get(), a2.get()}) {
    FillGrad(*p, 9);
  }
  for (auto *p : {c.get(), c2.get()}) {
    FillGrad(*p, 10);
  }
  first.step();
  second.step();

  ExpectBitwiseEqual(*a, *a2);
  ExpectBitwiseEqual(*c, *c2);
}

TEST(Optim, LoadStateRejectsMismatchedBuffers) {
  auto w = Tensor::zeros({2, 3});
  AdamW adamw({w}, 0.1f);
  Muon muon({w}, 0.1f);

  AdamWState wrong_count = adamw.state();
  wrong_count.v.clear();
  wrong_count.step_count = 7;
  EXPECT_THROW(adamw.load_state(wrong_count), std::invalid_argument);

  AdamWState wrong_shape = adamw.state();
  wrong_shape.m[0].shape = {3, 2};
  EXPECT_THROW(adamw.load_state(wrong_shape), std::invalid_argument);

  AdamWState wrong_values = adamw.state();
  wrong_values.v[0].values.pop_back();
  EXPECT_THROW(adamw.load_state(wrong_values), std::invalid_argument);

  MuonState muon_shape = muon.state();
  muon_shape.momentum[0].shape = {6};
  muon_shape.step_count = 7;
  EXPECT_THROW(muon.load_state(muon_shape), std::invalid_argument);

  EXPECT_EQ(adamw.step_count(), 0u);
  EXPECT_EQ(muon.step_count(), 0u);
}
