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

void CompareWithBroadcastAdd(const std::vector<size_t> &shape) {
  const size_t cols = shape.back();

  auto x_ref = Wave(shape, 0.2f);
  auto b_ref = Wave({cols}, 1.3f);
  auto ref = x_ref->add(b_ref);
  ref->mul(ref)->sum()->backward();

  auto x = Wave(shape, 0.2f);
  auto b = Wave({cols}, 1.3f);
  auto out = x->add_bias(b);
  out->mul(out)->sum()->backward();

  ExpectClose(ref->data(), out->data());
  ExpectClose(x_ref->grad(), x->grad());
  ExpectClose(b_ref->grad(), b->grad());
}

}  // namespace

TEST(AddBias, MatchesBroadcastAddOn2D) { CompareWithBroadcastAdd({5, 7}); }

TEST(AddBias, MatchesBroadcastAddOn3D) { CompareWithBroadcastAdd({2, 3, 4}); }

TEST(AddBias, FallsBackToBroadcastAddForOtherShapes) {
  auto x = Wave({2, 3}, 0.2f);
  auto column = Wave({2, 1}, 0.4f);
  auto out = x->add_bias(column);
  auto ref = x->add(column);

  ExpectClose(ref->data(), out->data());
}
