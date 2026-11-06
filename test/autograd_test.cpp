#include "micrograd/Autograd.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "test_helpers.h"

using namespace micrograd;
using namespace micrograd::test_helpers;

TEST(Autograd, ResultRequiresGradWhenAnyInputDoes) {
  auto tracked = scalar(2);
  auto untracked = Tensor::from_values({1}, {3});

  EXPECT_TRUE(tracked->add(untracked)->requires_grad());
  EXPECT_TRUE(untracked->add(tracked)->requires_grad());
  EXPECT_TRUE(tracked->mul(2.0f)->requires_grad());
}

TEST(Autograd, ResultSkipsGraphWhenNoInputRequiresGrad) {
  auto a = Tensor::from_values({1}, {2});
  auto b = Tensor::from_values({1}, {3});

  EXPECT_FALSE(a->add(b)->requires_grad());
  EXPECT_FALSE(a->mul(2.0f)->requires_grad());
}

TEST(Autograd, NoGradGuardStopsRecording) {
  auto a = scalar(2);
  auto b = scalar(3);
  NoGradGuard guard;

  EXPECT_FALSE(a->add(b)->requires_grad());
  EXPECT_FALSE(a->mul(2.0f)->requires_grad());
}

TEST(Autograd, GradientFlowsToEachInputOfASharedOperand) {
  auto a = scalar(3);

  a->mul(a)->backward();

  EXPECT_FLOAT_EQ(a->grad()[0], 6.0f);
}

TEST(Autograd, SanityCheck) {
  auto x = scalar(-4.0);

  auto z = x->mul(2.0)->add(2.0)->add(x);
  auto q = z->relu()->add(z->mul(x));
  auto h = z->mul(z)->relu();
  auto y = h->add(q)->add(q->mul(x));

  y->backward();

  EXPECT_NEAR(y->data()[0], -20.0, 1e-9);
  EXPECT_NEAR(x->grad()[0], 46.0, 1e-9);
}

TEST(Autograd, Add) {
  auto a = scalar(4);
  auto b = scalar(5);
  auto c = a->add(b);

  c->backward();

  EXPECT_NEAR(c->data()[0], 9.0, 1e-9);
  EXPECT_NEAR(a->grad()[0], 1, 1e-9);
  EXPECT_NEAR(b->grad()[0], 1, 1e-9);
}

TEST(Autograd, Sub) {
  auto a = scalar(4);
  auto b = scalar(7);
  auto c = a->sub(b);

  c->backward();

  EXPECT_NEAR(c->data()[0], -3.0, 1e-9);
  EXPECT_NEAR(a->grad()[0], 1, 1e-9);
  EXPECT_NEAR(b->grad()[0], -1, 1e-9);
}

TEST(Autograd, Mul) {
  auto a = scalar(4);
  auto b = scalar(5);
  auto c = a->mul(b);

  c->backward();

  EXPECT_NEAR(c->data()[0], 20.0, 1e-9);
  EXPECT_NEAR(a->grad()[0], 5, 1e-9);
  EXPECT_NEAR(b->grad()[0], 4, 1e-9);
}

TEST(Autograd, Div) {
  auto a = scalar(4);
  auto b = scalar(5);
  auto c = a->div(b);

  c->backward();

  EXPECT_NEAR(c->data()[0], 0.80, 1e-6);
  EXPECT_NEAR(a->grad()[0], 0.20, 1e-6);
  EXPECT_NEAR(b->grad()[0], -0.16, 1e-6);
}

TEST(Autograd, Pow) {
  auto a = scalar(4);
  auto b = a->pow(3.0);

  b->backward();

  EXPECT_NEAR(b->data()[0], 64, 1e-9);
  EXPECT_NEAR(a->grad()[0], 48, 1e-9);
}

TEST(Autograd, ChainRule) {
  auto x = scalar(3);
  auto a = x->pow(3.0);
  auto b = a->add(1.0);
  auto y = b->pow(2.0);

  y->backward();

  EXPECT_NEAR(y->data()[0], 784, 1e-9);
  EXPECT_NEAR(x->grad()[0], 1512, 1e-9);
}

TEST(Autograd, NumericGradRelu) {
  expect_grad_matches_numeric({-2.0f, 0.5f, 3.0f},
                              [](const auto &x) { return x->relu()->sum(); });
}

TEST(Autograd, NumericGradSigmoid) {
  expect_grad_matches_numeric(
      {-1.5f, 0.3f, 2.0f}, [](const auto &x) { return x->sigmoid()->sum(); });
}

TEST(Autograd, NumericGradTanh) {
  expect_grad_matches_numeric({-1.5f, 0.3f, 2.0f},
                              [](const auto &x) { return x->tanh()->sum(); });
}

TEST(Autograd, NumericGradPow) {
  expect_grad_matches_numeric(
      {0.5f, 1.5f, 2.5f}, [](const auto &x) { return x->pow(3.0f)->sum(); });
}

TEST(Autograd, NumericGradDiv) {
  expect_grad_matches_numeric({1.0f, 2.0f}, [](const auto &x) {
    return x->div(vec({4.0f, 5.0f}))->sum();
  });
}

TEST(Autograd, NumericGradComposite) {
  expect_grad_matches_numeric({0.4f, -0.7f, 1.2f}, [](const auto &x) {
    return x->mul(x)->add(x->tanh())->sigmoid()->sum();
  });
}

TEST(Autograd, MatmulNonSquare) {
  auto a = leaf({2, 3}, {1, 2, 3, 4, 5, 6});
  auto b = leaf({3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  auto c = a->matmul(b);
  c->sum()->backward();

  ASSERT_EQ(c->shape(), (std::vector<size_t>{2, 4}));
  EXPECT_NEAR(c->data()[0], 38.0, 1e-9);
  EXPECT_NEAR(c->data()[3], 56.0, 1e-9);
  EXPECT_NEAR(c->data()[7], 128.0, 1e-9);

  EXPECT_NEAR(a->grad()[0], 10.0, 1e-9);
  EXPECT_NEAR(a->grad()[1], 26.0, 1e-9);
  EXPECT_NEAR(a->grad()[2], 42.0, 1e-9);

  EXPECT_NEAR(b->grad()[0], 5.0, 1e-9);
  EXPECT_NEAR(b->grad()[4], 7.0, 1e-9);
  EXPECT_NEAR(b->grad()[8], 9.0, 1e-9);
}

TEST(Autograd, GraphIsFreed) {
  std::weak_ptr<Tensor> w;
  {
    auto x = scalar(2.0);
    auto y = x->mul(3.0)->relu();
    y->backward();
    w = y;
  }
  EXPECT_TRUE(w.expired());
}
