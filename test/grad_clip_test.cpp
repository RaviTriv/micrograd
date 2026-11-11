#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Optimizer.h"

using namespace micrograd;

namespace {

std::shared_ptr<Tensor> ParamWithGrad(std::vector<scalar_t> grad) {
  auto param = Tensor::from_values({grad.size()},
                                   std::vector<scalar_t>(grad.size(), 1.0f));
  param->set_requires_grad(true);
  for (size_t i = 0; i < grad.size(); i++) {
    param->grad()[i] = grad[i];
  }
  return param;
}

}  // namespace

TEST(GradClip, ScalesGlobalNormDownToMax) {
  auto a = ParamWithGrad({3.0f, 0.0f});
  auto b = ParamWithGrad({0.0f, 4.0f});

  EXPECT_NEAR(ops::ClipGradNorm({a, b}, 1.0f), 5.0f, 1e-6f);

  EXPECT_NEAR(a->grad()[0], 0.6f, 1e-6f);
  EXPECT_NEAR(b->grad()[1], 0.8f, 1e-6f);
  const double norm =
      std::sqrt((a->grad()[0] * a->grad()[0]) + (b->grad()[1] * b->grad()[1]));
  EXPECT_NEAR(norm, 1.0, 1e-6);
}

TEST(GradClip, LeavesGradsBelowMaxUnchanged) {
  auto a = ParamWithGrad({3.0f, 0.0f});
  auto b = ParamWithGrad({0.0f, 4.0f});

  EXPECT_NEAR(ops::ClipGradNorm({a, b}, 10.0f), 5.0f, 1e-6f);

  EXPECT_EQ(a->grad()[0], 3.0f);
  EXPECT_EQ(b->grad()[1], 4.0f);
}

TEST(GradClip, NonPositiveMaxDisablesClipping) {
  auto a = ParamWithGrad({30.0f, 40.0f});

  EXPECT_NEAR(ops::ClipGradNorm({a}, 0.0f), 50.0f, 1e-5f);

  EXPECT_EQ(a->grad()[0], 30.0f);
  EXPECT_EQ(a->grad()[1], 40.0f);
}

TEST(GradClip, EmptyParameterListHasZeroNorm) {
  EXPECT_EQ(ops::ClipGradNorm({}, 1.0f), 0.0f);
}
