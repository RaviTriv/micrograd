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

TEST(TensorTest, Pow){
  auto a = scalar(4);
  auto b = a->pow(3.0);

  b->backward();

  EXPECT_NEAR(b->data()[0], 64, 1e-9);
  EXPECT_NEAR(a->grad()[0], 48, 1e-9);
}