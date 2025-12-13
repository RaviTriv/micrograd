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
