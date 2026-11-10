#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "micrograd/NN.h"
#include "micrograd/Random.h"
#include "test_helpers.h"

using namespace micrograd;
using namespace micrograd::test_helpers;

TEST(DType, Bfloat16RoundsToNearestEven) {
  EXPECT_EQ(float_to_bfloat16_bits(1.0f), 0x3f80);
  EXPECT_EQ(bfloat16_bits_to_float(float_to_bfloat16_bits(1.00390625f)),
            1.0f);
  EXPECT_EQ(bfloat16_bits_to_float(float_to_bfloat16_bits(1.01171875f)),
            1.015625f);
  EXPECT_EQ(bfloat16_bits_to_float(float_to_bfloat16_bits(-3.0f)), -3.0f);
}

TEST(DType, CastToBfloat16HalvesStorageAndKeepsShape) {
  auto x = Make({2, 3}, {1, 2, 3, 4, 5, 6});
  auto y = x->cast(DType::kBFloat16);

  EXPECT_EQ(y->dtype(), DType::kBFloat16);
  EXPECT_EQ(y->shape(), x->shape());
  EXPECT_EQ(y->size(), 6u);
  EXPECT_EQ(y->data_storage().bytes(), 12u);
}

TEST(DType, CastToSameDtypeReturnsTheTensor) {
  auto x = Make({2}, {1, 2});
  EXPECT_EQ(x->cast(DType::kFloat32), x);
}

TEST(DType, CastRoundTripRoundsToBfloat16) {
  auto x = Make({3}, {1.00390625f, 0.1f, -7.5f});
  auto back = x->cast(DType::kBFloat16)->cast(DType::kFloat32);

  EXPECT_EQ(back->dtype(), DType::kFloat32);
  EXPECT_EQ(Values(back), (std::vector<scalar_t>{
                              1.0f, bfloat16_bits_to_float(0x3dcd), -7.5f}));
}

TEST(DType, CastPassesGradientsBackToFloat32) {
  auto x = leaf({3}, {0.5f, -1.0f, 2.0f});
  auto y = x->cast(DType::kBFloat16)->cast(DType::kFloat32)->mul(3.0f)->sum();
  y->backward();

  EXPECT_EQ(x->grad()[0], 3.0f);
  EXPECT_EQ(x->grad()[1], 3.0f);
  EXPECT_EQ(x->grad()[2], 3.0f);
}

TEST(DType, ElementAccessOnBfloat16Throws) {
  auto y = Make({2}, {1, 2})->cast(DType::kBFloat16);
  EXPECT_THROW(static_cast<void>(y->data()), std::runtime_error);
}

TEST(DType, MixedDtypeArithmeticThrows) {
  auto x = Make({2}, {1, 2});
  auto y = Make({2}, {1, 2})->cast(DType::kBFloat16);
  EXPECT_THROW(static_cast<void>(x->add(y)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Make({2, 2}, {1, 2, 3, 4})->matmul(
                   Make({2, 2}, {1, 2, 3, 4})->cast(DType::kBFloat16))),
               std::invalid_argument);
}

TEST(DType, Bfloat16OpsWithoutKernelsThrowOnCpu) {
  auto y = Make({2}, {1, 2})->cast(DType::kBFloat16);
  EXPECT_THROW(static_cast<void>(y->gelu()), std::runtime_error);
}

TEST(Loss, CrossEntropyIgnoresPaddedClasses) {
  auto padded = leaf({2, 5}, {0.5f, -1.0f, 2.0f, 9.0f, 9.0f, 1.0f, 0.0f,
                              -2.0f, 9.0f, 9.0f});
  auto narrow = leaf({2, 3}, {0.5f, -1.0f, 2.0f, 1.0f, 0.0f, -2.0f});

  auto padded_loss = cross_entropy(padded, {2, 0}, 3);
  auto narrow_loss = cross_entropy(narrow, {2, 0});
  padded_loss->backward();
  narrow_loss->backward();

  EXPECT_FLOAT_EQ(padded_loss->data()[0], narrow_loss->data()[0]);
  for (size_t row = 0; row < 2; row++) {
    for (size_t col = 0; col < 3; col++) {
      EXPECT_FLOAT_EQ(padded->grad_at({row, col}), narrow->grad_at({row, col}));
    }
    EXPECT_EQ(padded->grad_at({row, 3}), 0.0f);
    EXPECT_EQ(padded->grad_at({row, 4}), 0.0f);
  }
}

TEST(Loss, CrossEntropyRejectsTargetInPaddedClasses) {
  EXPECT_THROW(cross_entropy(Make({1, 4}, {0, 0, 0, 0}), {3}, 3),
               std::out_of_range);
  EXPECT_THROW(cross_entropy(Make({1, 4}, {0, 0, 0, 0}), {0}, 5),
               std::invalid_argument);
}

TEST(Layers, PaddedEmbeddingKeepsInitAndZerosPadding) {
  manual_seed(11);
  Embedding plain(5, 3);
  manual_seed(11);
  Embedding padded(5, 3, 8);

  EXPECT_EQ(padded.weight()->shape(), (std::vector<size_t>{8, 3}));
  std::vector<scalar_t> head = Values(plain.weight());
  std::vector<scalar_t> values = Values(padded.weight());
  for (size_t i = 0; i < head.size(); i++) {
    EXPECT_EQ(values[i], head[i]);
  }
  for (size_t i = head.size(); i < values.size(); i++) {
    EXPECT_EQ(values[i], 0.0f);
  }
}
