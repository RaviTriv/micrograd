#include "micrograd/Autograd.h"

#include <gtest/gtest.h>

#include <memory>
#include <span>
#include <stdexcept>
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

TEST(Autograd, TrackedIndicesDoNotEnableGrad) {
  auto weights = Tensor::from_values({2, 2}, {1, 2, 3, 4});
  auto indices = vec({1, 0});

  EXPECT_FALSE(weights->embedding_lookup(indices)->requires_grad());
  EXPECT_FALSE(weights->gather_per_row(indices)->requires_grad());
}

TEST(Autograd, GatherPerRowFromHostIndicesMatchesTensorIndices) {
  auto values = Tensor::from_values({3, 2}, {1, 2, 3, 4, 5, 6});
  auto from_tensor = values->gather_per_row(vec({1, 0, 1}));
  auto from_host = values->gather_per_row(std::vector<size_t>{1, 0, 1});
  EXPECT_EQ(Values(from_host), Values(from_tensor));
  EXPECT_EQ(Values(from_host), (std::vector<scalar_t>{2, 3, 6}));
}

TEST(Autograd, GatherPerRowRejectsOutOfRangeHostIndex) {
  auto values = Tensor::from_values({2, 2}, {1, 2, 3, 4});
  EXPECT_THROW(values->gather_per_row(std::vector<size_t>{0, 2}),
               std::out_of_range);
  EXPECT_THROW(values->gather_per_row(std::vector<size_t>{0}),
               std::invalid_argument);
}

TEST(Autograd, GradientSkipsIndices) {
  auto values = leaf({2, 2}, {1, 2, 3, 4});
  auto indices = vec({1, 0});

  values->gather_per_row(indices)->sum()->backward();

  EXPECT_EQ(Values(values), (std::vector<scalar_t>{1, 2, 3, 4}));
  const std::span<const scalar_t> gradient = values->grad();
  EXPECT_EQ(std::vector<scalar_t>(gradient.begin(), gradient.end()),
            (std::vector<scalar_t>{0, 1, 1, 0}));
  const std::span<const scalar_t> index_gradient = indices->grad();
  EXPECT_EQ(std::vector<scalar_t>(index_gradient.begin(), index_gradient.end()),
            (std::vector<scalar_t>{0, 0}));
}

TEST(LazyGrad, OpResultsHaveNoGradBufferBeforeBackward) {
  auto a = scalar(2);

  auto b = a->mul(3.0f);
  auto c = b->reshape({1});

  EXPECT_FALSE(a->has_grad());
  EXPECT_FALSE(b->has_grad());
  EXPECT_FALSE(c->has_grad());
}

TEST(LazyGrad, BackwardAllocatesEveryTensorInTheGraph) {
  auto a = vec({1, 2});
  auto b = a->mul(3.0f);
  auto c = b->reshape({2, 1});
  auto loss = c->sum();

  loss->backward();

  EXPECT_TRUE(a->has_grad());
  EXPECT_TRUE(b->has_grad());
  EXPECT_TRUE(c->has_grad());
  EXPECT_TRUE(loss->has_grad());
  EXPECT_EQ(a->grad()[0], 3.0f);
  EXPECT_EQ(a->grad()[1], 3.0f);
}

TEST(LazyGrad, NoGradResultsStayWithoutGradBuffer) {
  auto a = vec({1, 2});
  NoGradGuard guard;

  auto b = a->mul(3.0f);

  EXPECT_FALSE(b->has_grad());
  EXPECT_FALSE(a->has_grad());
}

TEST(LazyGrad, ReadingGradBeforeBackwardGivesZeros) {
  auto t = Tensor::from_values({2, 3}, {1, 2, 3, 4, 5, 6});

  const std::span<const scalar_t> gradient = t->grad();

  EXPECT_EQ(std::vector<scalar_t>(gradient.begin(), gradient.end()),
            std::vector<scalar_t>(6, 0.0f));
  EXPECT_EQ(t->grad_at({1, 2}), 0.0f);
  EXPECT_EQ(t->grad_storage().bytes(), 6 * sizeof(scalar_t));
  EXPECT_TRUE(t->has_grad());
}

TEST(LazyGrad, ConstAccessReadsZeros) {
  auto t = Tensor::from_values({2}, {1, 2});
  const Tensor &view = *t;

  EXPECT_EQ(view.grad_at({1}), 0.0f);
  EXPECT_EQ(view.grad().size(), 2u);
}

TEST(LazyGrad, ZeroDimTensorGetsOneElementGrad) {
  auto t = Tensor::zeros({});

  EXPECT_EQ(t->grad_at({}), 0.0f);
  EXPECT_TRUE(t->has_grad());
}

TEST(LazyGrad, EmptyTensorHasNoGradBuffer) {
  auto t = Tensor::zeros({0});

  EXPECT_TRUE(t->grad().empty());
  EXPECT_FALSE(t->has_grad());
}

TEST(LazyGrad, ZeroGradDoesNotAllocate) {
  auto t = vec({1, 2});

  t->zero_grad();

  EXPECT_FALSE(t->has_grad());
}

TEST(LazyGrad, ZeroGradClearsAnExistingBuffer) {
  auto a = vec({1, 2});
  a->sum()->backward();
  ASSERT_EQ(a->grad()[0], 1.0f);

  a->zero_grad();

  EXPECT_TRUE(a->has_grad());
  EXPECT_EQ(a->grad()[0], 0.0f);
  EXPECT_EQ(a->grad()[1], 0.0f);
}

TEST(LazyGrad, RepeatedBackwardKeepsAccumulatingOnLeaves) {
  auto a = vec({1, 2});

  a->sum()->backward();
  a->sum()->backward();

  EXPECT_EQ(a->grad()[0], 2.0f);
  EXPECT_EQ(a->grad()[1], 2.0f);
}

TEST(Autograd, BackwardWorksFromZeroDimResult) {
  auto x = vec({1, 2});

  auto loss = x->sum();
  loss->backward();

  EXPECT_TRUE(loss->shape().empty());
  EXPECT_EQ(x->grad()[0], 1.0f);
  EXPECT_EQ(x->grad()[1], 1.0f);
}

TEST(Autograd, GradientFlowsThroughBroadcastZeroDim) {
  auto x = vec({1, 2});

  x->mul(x->sum())->sum()->backward();

  EXPECT_EQ(x->grad()[0], 6.0f);
  EXPECT_EQ(x->grad()[1], 6.0f);
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

TEST(Autograd, SharedIntermediateAcrossTwoBackwards) {
  auto x = scalar(2);
  auto h = x->mul(2.0f);
  auto left = h->mul(3.0f)->sum();
  auto right = h->mul(4.0f)->sum();

  left->backward(true);
  EXPECT_FLOAT_EQ(x->grad()[0], 6.0f);

  right->backward();
  EXPECT_FLOAT_EQ(x->grad()[0], 14.0f);
}

TEST(Autograd, SameRootTwiceWithRetainedGraphAddsOnce) {
  auto x = scalar(3);
  auto loss = x->mul(x)->mul(2.0f)->sum();

  loss->backward(true);
  loss->backward();

  EXPECT_FLOAT_EQ(x->grad()[0], 24.0f);
}

TEST(Autograd, BackwardTwiceWithoutRetainThrows) {
  auto x = scalar(3);
  auto loss = x->mul(x)->sum();
  loss->backward();

  EXPECT_THROW(loss->backward(), std::runtime_error);
  EXPECT_FLOAT_EQ(x->grad()[0], 6.0f);
}

TEST(Autograd, NewRootOnReleasedIntermediateThrowsBeforeTouchingGrads) {
  auto x = scalar(2);
  auto h = x->mul(2.0f);
  h->mul(3.0f)->sum()->backward();

  EXPECT_THROW(h->mul(5.0f)->sum()->backward(), std::runtime_error);
  EXPECT_FLOAT_EQ(x->grad()[0], 6.0f);
}

TEST(Autograd, BackwardReleasesIntermediatesWhileRootIsAlive) {
  auto x = scalar(2);
  std::weak_ptr<Tensor> intermediate;
  std::shared_ptr<Tensor> loss;
  {
    auto h = x->mul(2.0f);
    intermediate = h;
    loss = h->relu()->sum();
  }
  EXPECT_FALSE(intermediate.expired());

  loss->backward();

  EXPECT_TRUE(intermediate.expired());
  EXPECT_FLOAT_EQ(x->grad()[0], 2.0f);
}
