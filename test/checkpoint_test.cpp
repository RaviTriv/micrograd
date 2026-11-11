#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Random.h"
#include "test_helpers.h"

using namespace micrograd;
using namespace micrograd::test_helpers;

namespace {

std::vector<scalar_t> Grad(const std::shared_ptr<Tensor> &t) {
  return {t->grad().begin(), t->grad().end()};
}

}  // namespace

TEST(Checkpoint, RawInputStillGivesSegmentParametersGradients) {
  auto x = Make({2}, {1.0f, 2.0f});
  auto w = leaf({2}, {3.0f, 4.0f});

  auto out = Tensor::checkpoint(
      x, [&w](const std::shared_ptr<Tensor> &in) { return in->mul(w); });
  out->sum()->backward();

  EXPECT_EQ(Grad(w), (std::vector<scalar_t>{1.0f, 2.0f}));
}

TEST(Checkpoint, OutputNeedsNoGradUnderNoGradGuard) {
  auto x = leaf({2}, {1.0f, 2.0f});
  const NoGradGuard no_grad;

  auto out = Tensor::checkpoint(
      x, [](const std::shared_ptr<Tensor> &in) { return in->mul(2.0f); });

  EXPECT_FALSE(out->requires_grad());
}

TEST(Checkpoint, FailedRecomputeRestoresRngAndInputGraph) {
  auto a = leaf({2}, {1.0f, 2.0f});
  auto input = a->mul(3.0f);
  bool fail = false;
  auto recompute = [&fail](const std::shared_ptr<Tensor> &in) {
    global_rng()();
    if (fail) {
      throw std::runtime_error("recompute failed");
    }
    return in->mul(in);
  };

  auto loss = Tensor::checkpoint(input, recompute)->sum();
  global_rng().discard(5);
  const std::mt19937_64 before_backward = global_rng();

  fail = true;
  EXPECT_THROW(loss->backward(), std::runtime_error);
  EXPECT_TRUE(global_rng() == before_backward);

  fail = false;
  loss->backward();
  EXPECT_EQ(Grad(a), (std::vector<scalar_t>{18.0f, 36.0f}));
}
