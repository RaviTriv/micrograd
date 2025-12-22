#include "micrograd/Tensor.h"
#include <gtest/gtest.h>
#include <memory>

auto scalar(double val) {
  return std::make_shared<Tensor>(std::vector<size_t>{1},
                                  std::vector<double>{val});
}

TEST(TensorTest, SanityCheck) {
  auto x = scalar(-4.0);

  auto z = x->mul(2.0)->add(2.0)->add(x);
  auto q = z->relu()->add(z->mul(x));
  auto h = z->mul(z)->relu();
  auto y = h->add(q)->add(q->mul(x));

  y->backward();

  EXPECT_NEAR(y->data()[0], -20.0, 1e-9);
  EXPECT_NEAR(x->grad()[0], 46.0, 1e-9);
}

TEST(TensorTest, Add) {
  auto a = scalar(4);
  auto b = scalar(5);
  auto c = a->add(b);

  c->backward();

  EXPECT_NEAR(c->data()[0], 9.0, 1e-9);
  EXPECT_NEAR(a->grad()[0], 1, 1e-9);
  EXPECT_NEAR(b->grad()[0], 1, 1e-9);
}

TEST(TensorTest, Sub) {
  auto a = scalar(4);
  auto b = scalar(7);
  auto c = a->sub(b);

  c->backward();

  EXPECT_NEAR(c->data()[0], -3.0, 1e-9);
  EXPECT_NEAR(a->grad()[0], 1, 1e-9);
  EXPECT_NEAR(b->grad()[0], -1, 1e-9);
}

TEST(TensorTest, Mul) {
  auto a = scalar(4);
  auto b = scalar(5);
  auto c = a->mul(b);

  c->backward();

  EXPECT_NEAR(c->data()[0], 20.0, 1e-9);
  EXPECT_NEAR(a->grad()[0], 5, 1e-9);
  EXPECT_NEAR(b->grad()[0], 4, 1e-9);
}

TEST(TensorTest, Div) {
  auto a = scalar(4);
  auto b = scalar(5);
  auto c = a->div(b);

  c->backward();

  EXPECT_NEAR(c->data()[0], 0.80, 1e-9);
  EXPECT_NEAR(a->grad()[0], 0.20, 1e-9);
  EXPECT_NEAR(b->grad()[0], -0.16, 1e-9);
}

TEST(TensorTest, Pow) {
  auto a = scalar(4);
  auto b = a->pow(3.0);

  b->backward();

  EXPECT_NEAR(b->data()[0], 64, 1e-9);
  EXPECT_NEAR(a->grad()[0], 48, 1e-9);
}

TEST(TensorTest, Quadratic) {
  auto x = scalar(2);
  auto b = x->pow(2.0);
  auto c = b->mul(2.0);
  auto d = x->mul(3.0);
  auto e = d->add(5.0);
  auto y = c->add(e);

  y->backward();

  EXPECT_NEAR(y->data()[0], 19, 1e-9);
  EXPECT_NEAR(x->grad()[0], 11, 1e-9);
}

TEST(TensorTest, ChainRule) {
  auto x = scalar(3);
  auto a = x->pow(3.0);
  auto b = a->add(1.0);
  auto y = b->pow(2.0);

  y->backward();

  EXPECT_NEAR(y->data()[0], 784, 1e-9);
  EXPECT_NEAR(x->grad()[0], 1512, 1e-9);
}

TEST(TensorTest, SumMetal) {
  auto x = std::make_shared<Tensor>(std::vector<size_t>{4},
                                    std::vector<double>{1.0, 2.0, 3.0, 4.0});
  x->to(micrograd::Backend::Metal);

  auto y = x->sum();

  EXPECT_NEAR(y->data()[0], 10.0, 1e-4);

  y->backward();

  x->to(micrograd::Backend::CPU);
  EXPECT_NEAR(x->grad()[0], 1.0, 1e-4);
  EXPECT_NEAR(x->grad()[1], 1.0, 1e-4);
  EXPECT_NEAR(x->grad()[2], 1.0, 1e-4);
  EXPECT_NEAR(x->grad()[3], 1.0, 1e-4);
}

TEST(TensorTest, SumMetalTimeComparison) {
  std::vector<double> data(10000);
  double expected = 0.0;
  for (int i = 0; i < 10000; i++) {
    data[static_cast<size_t>(i)] = static_cast<double>(i + 1);
    expected += static_cast<double>(i + 1);
  }

  auto x = std::make_shared<Tensor>(std::vector<size_t>{10000}, data);
  x->to(micrograd::Backend::Metal);

  auto y = x->sum();

  EXPECT_NEAR(y->data()[0], expected, 1e-1);
}

TEST(TensorTest, MatmulMetalBackward) {
  auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 2},
                                    std::vector<double>{1, 2, 3, 4});
  auto b = std::make_shared<Tensor>(std::vector<size_t>{2, 2},
                                    std::vector<double>{5, 6, 7, 8});

  a->to(micrograd::Backend::Metal);
  b->to(micrograd::Backend::Metal);

  auto c = a->matmul(b);

  c->to(micrograd::Backend::CPU);
  EXPECT_NEAR(c->data()[0], 19.0, 1e-4);
  EXPECT_NEAR(c->data()[1], 22.0, 1e-4);
  EXPECT_NEAR(c->data()[2], 43.0, 1e-4);
  EXPECT_NEAR(c->data()[3], 50.0, 1e-4);

  c->to(micrograd::Backend::Metal);
  auto loss = c->sum();
  loss->backward();

  EXPECT_NEAR(a->grad()[0], 11.0, 1e-4);
  EXPECT_NEAR(a->grad()[1], 15.0, 1e-4);
  EXPECT_NEAR(a->grad()[2], 11.0, 1e-4);
  EXPECT_NEAR(a->grad()[3], 15.0, 1e-4);

  EXPECT_NEAR(b->grad()[0], 4.0, 1e-4);
  EXPECT_NEAR(b->grad()[1], 4.0, 1e-4);
  EXPECT_NEAR(b->grad()[2], 6.0, 1e-4);
  EXPECT_NEAR(b->grad()[3], 6.0, 1e-4);
}
