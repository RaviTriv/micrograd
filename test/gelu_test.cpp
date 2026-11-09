#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "micrograd/Tensor.h"

using namespace micrograd;

namespace {

std::shared_ptr<Tensor> Leaf(const std::vector<scalar_t> &values) {
  auto leaf = Tensor::from_values({values.size()}, values);
  leaf->set_requires_grad(true);
  return leaf;
}

std::shared_ptr<Tensor> ComposedGelu(const std::shared_ptr<Tensor> &x) {
  constexpr scalar_t kSqrt2OverPi = 0.7978845608028654f;
  constexpr scalar_t kCubicCoeff = 0.044715f;
  auto cubed = x->mul(x)->mul(x);
  auto inner = x->add(cubed->mul(kCubicCoeff))->mul(kSqrt2OverPi);
  auto gate = inner->tanh()->add(1.0f)->mul(0.5f);
  return x->mul(gate);
}

}  // namespace

TEST(Gelu, FusedMatchesComposedForwardAndGradient) {
  std::vector<scalar_t> values;
  for (int i = -40; i <= 40; i++) {
    values.push_back(0.1f * static_cast<scalar_t>(i));
  }

  auto fused_x = Leaf(values);
  auto fused = fused_x->gelu();
  fused->sum()->backward();

  auto composed_x = Leaf(values);
  auto composed = ComposedGelu(composed_x);
  composed->sum()->backward();

  for (size_t i = 0; i < values.size(); i++) {
    EXPECT_NEAR(fused->data()[i], composed->data()[i], 1e-6f)
        << "x " << values[i];
    EXPECT_NEAR(fused_x->grad()[i], composed_x->grad()[i], 1e-5f)
        << "x " << values[i];
  }
}
