#include "gpt/Attention.h"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <span>
#include <vector>

#include "micrograd/Tensor.h"

using namespace micrograd;

namespace {

std::shared_ptr<Tensor> Wave(std::vector<size_t> shape, scalar_t phase) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= dim;
  }
  std::vector<scalar_t> values(count);
  for (size_t i = 0; i < count; i++) {
    values[i] = std::sin((0.37f * static_cast<scalar_t>(i)) + phase);
  }
  auto tensor = Tensor::from_values(std::move(shape), std::move(values));
  tensor->set_requires_grad(true);
  return tensor;
}

void ExpectClose(std::span<const scalar_t> expected,
                 std::span<const scalar_t> actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(expected[i], actual[i], 1e-5f) << "index " << i;
  }
}

void CompareWithFused(const std::vector<size_t> &shape, scalar_t scale) {
  auto q_fused = Wave(shape, 0.1f);
  auto k_fused = Wave(shape, 0.8f);
  auto v_fused = Wave(shape, 1.7f);
  auto fused = q_fused->flash_attention(k_fused, v_fused, scale);
  fused->sum()->backward();

  auto q = Wave(shape, 0.1f);
  auto k = Wave(shape, 0.8f);
  auto v = Wave(shape, 1.7f);
  auto composed = gpt::causal_attention(
      q, k, v, scale, gpt::causal_mask(shape[1], Device::CPU));
  composed->sum()->backward();

  ExpectClose(fused->data(), composed->data());
  ExpectClose(q_fused->grad(), q->grad());
  ExpectClose(k_fused->grad(), k->grad());
  ExpectClose(v_fused->grad(), v->grad());
}

}  // namespace

TEST(Attention, CausalMaskBlocksFuturePositions) {
  auto mask = gpt::causal_mask(3, Device::CPU);
  std::span<const scalar_t> values = mask->data();
  for (size_t row = 0; row < 3; row++) {
    for (size_t col = 0; col < 3; col++) {
      if (col <= row) {
        EXPECT_EQ(values[(row * 3) + col], 0.0f);
      } else {
        EXPECT_TRUE(std::isinf(values[(row * 3) + col]));
        EXPECT_LT(values[(row * 3) + col], 0.0f);
      }
    }
  }
}

TEST(Attention, ComposedMatchesFusedKernel) {
  CompareWithFused({2, 5, 4}, 0.5f);
}

TEST(Attention, ComposedMatchesFusedKernelPastOneTile) {
  CompareWithFused({2, 40, 8}, 0.35f);
}
