#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "test_helpers.h"

using namespace micrograd;
using namespace micrograd::test_helpers;

static_assert(!std::is_default_constructible_v<Tensor>);
static_assert(!std::is_constructible_v<Tensor, std::vector<size_t>>);
static_assert(!std::is_constructible_v<Tensor, std::vector<size_t>,
                                       std::vector<scalar_t>>);
static_assert(!std::is_convertible_v<std::vector<size_t>, Tensor>);
static_assert(!std::is_copy_constructible_v<Tensor>);
static_assert(!std::is_copy_assignable_v<Tensor>);
static_assert(!std::is_move_constructible_v<Tensor>);
static_assert(!std::is_move_assignable_v<Tensor>);

TEST(TensorFactories, ZerosFillsDataAndGradWithZeros) {
  auto t = Tensor::zeros({2, 3});

  EXPECT_EQ(t->shape(), (std::vector<size_t>{2, 3}));
  EXPECT_EQ(Values(t), std::vector<scalar_t>(6, 0.0f));
  EXPECT_EQ(t->grad_storage().bytes(), 6 * sizeof(scalar_t));
  EXPECT_EQ(t->grad_at({1, 2}), 0.0f);
  EXPECT_FALSE(t->requires_grad());
}

TEST(TensorFactories, ZerosWithEmptyShapeIsZeroDimScalar) {
  auto t = Tensor::zeros({});

  EXPECT_TRUE(t->shape().empty());
  EXPECT_EQ(t->size(), 1u);
  EXPECT_EQ(t->at({}), 0.0f);
}

TEST(TensorFactories, FromValuesKeepsValuesInOrder) {
  auto t = Tensor::from_values({2, 3}, {1, 2, 3, 4, 5, 6});

  EXPECT_EQ(t->shape(), (std::vector<size_t>{2, 3}));
  EXPECT_EQ(t->at({0, 2}), 3.0f);
  EXPECT_EQ(t->at({1, 0}), 4.0f);
  EXPECT_FALSE(t->requires_grad());
}

TEST(TensorFactories, FromValuesRejectsWrongValueCount) {
  EXPECT_THROW(static_cast<void>(Tensor::from_values({2, 2}, {1, 2, 3})),
               std::invalid_argument);
}

TEST(TensorFactories, ResultCanJoinTheGraph) {
  auto t = Tensor::from_values({2}, {1, 2});
  auto other = Tensor::from_values({2}, {3, 4});

  EXPECT_EQ(Values(t->add(other)), (std::vector<scalar_t>{4, 6}));
}

TEST(TensorIndexing, AtReadsAndWritesByPosition) {
  auto t = Make({2, 3}, Arange(6));

  EXPECT_EQ(t->at({1, 2}), 5.0f);
  t->at({0, 1}) = 9.0f;

  EXPECT_EQ(Values(t), (std::vector<scalar_t>{0, 9, 2, 3, 4, 5}));
}

TEST(TensorIndexing, AtRejectsTooFewIndices) {
  auto t = Make({2, 3}, Arange(6));

  EXPECT_THROW(t->at({1}), std::invalid_argument);
}

TEST(TensorIndexing, AtRejectsTooManyIndices) {
  auto t = Make({2, 3}, Arange(6));

  EXPECT_THROW(t->at({1, 2, 0}), std::invalid_argument);
}

TEST(TensorIndexing, AtRejectsIndexOnZeroDimTensor) {
  auto t = Tensor::zeros({});

  EXPECT_THROW(t->at({0}), std::invalid_argument);
}

TEST(TensorIndexing, AtRejectsOutOfRangeIndexInEachDimension) {
  auto t = Make({2, 3}, Arange(6));

  EXPECT_THROW(t->at({2, 0}), std::out_of_range);
  EXPECT_THROW(t->at({0, 3}), std::out_of_range);
}

TEST(TensorIndexing, ConstAtUsesTheSameChecks) {
  const auto t = Make({2, 3}, Arange(6));
  const Tensor &view = *t;

  EXPECT_EQ(view.at({1, 2}), 5.0f);
  EXPECT_THROW(static_cast<void>(view.at({1})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(view.at({2, 0})), std::out_of_range);
}

TEST(TensorIndexing, GradAtUsesTheSameChecks) {
  auto t = Make({2, 3}, Arange(6));

  t->grad_at({1, 2}) = 7.0f;

  EXPECT_EQ(t->grad_at({1, 2}), 7.0f);
  EXPECT_THROW(t->grad_at({1}), std::invalid_argument);
  EXPECT_THROW(t->grad_at({0, 3}), std::out_of_range);
}

TEST(TensorReshaping, Reshape) {
  auto t = Make({12}, Arange(12));

  auto reshaped = t->reshape({3, 4});

  EXPECT_EQ(reshaped->shape(), (std::vector<size_t>{3, 4}));
  EXPECT_EQ(Values(reshaped), Arange(12));
  EXPECT_EQ(reshaped->at({1, 2}), 6.0f);
  EXPECT_EQ(reshaped->at({2, 3}), 11.0f);
}

TEST(TensorReshaping, Flatten) {
  auto t = Make({2, 3, 4}, Arange(24));

  auto flat = t->reshape({-1});

  EXPECT_EQ(flat->shape(), (std::vector<size_t>{24}));
  EXPECT_EQ(Values(flat), Arange(24));
}

TEST(TensorReshaping, Transpose) {
  auto t = Make({2, 3}, {1, 2, 3, 4, 5, 6});

  auto transposed = t->transpose(0, 1);

  EXPECT_EQ(transposed->shape(), (std::vector<size_t>{3, 2}));
  EXPECT_EQ(Values(transposed), (std::vector<scalar_t>{1, 4, 2, 5, 3, 6}));
}

TEST(TensorReshaping, TransposeMovesEveryElementIn3D) {
  auto t = Make({2, 3, 4}, Arange(24));

  auto swapped = t->transpose(0, 2);

  ASSERT_EQ(swapped->shape(), (std::vector<size_t>{4, 3, 2}));
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < 4; k++) {
        EXPECT_EQ(swapped->at({k, j, i}), t->at({i, j, k}));
      }
    }
  }
}

TEST(TensorReshaping, PermuteMovesEveryElementIn3D) {
  auto t = Make({2, 3, 4}, Arange(24));

  auto permuted = t->permute({2, 0, 1});

  ASSERT_EQ(permuted->shape(), (std::vector<size_t>{4, 2, 3}));
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < 4; k++) {
        EXPECT_EQ(permuted->at({k, i, j}), t->at({i, j, k}));
      }
    }
  }
}

TEST(TensorReshaping, TransposeAcceptsNegativeDims) {
  auto t = Make({2, 3}, {1, 2, 3, 4, 5, 6});

  auto transposed = t->transpose(-2, -1);

  EXPECT_EQ(transposed->shape(), (std::vector<size_t>{3, 2}));
  EXPECT_EQ(Values(transposed), (std::vector<scalar_t>{1, 4, 2, 5, 3, 6}));
}

TEST(TensorReshaping, ContiguousReturnsTheSameTensor) {
  auto t = Make({2, 3}, Arange(6));
  auto transposed = t->transpose(0, 1);

  EXPECT_EQ(t->contiguous().get(), t.get());
  EXPECT_EQ(transposed->contiguous().get(), transposed.get());
}

TEST(TensorIndexing, AtUsesRowMajorOrderIn3D) {
  auto t = Make({2, 3, 4}, Arange(24));

  EXPECT_EQ(t->at({0, 1, 2}), 6.0f);
  EXPECT_EQ(t->at({1, 2, 3}), 23.0f);
  EXPECT_EQ(t->at({1, 0, 0}), 12.0f);
}

TEST(TensorCore, InvalidShapesThrow) {
  EXPECT_THROW(vec({1.0, 2.0})->add(vec({1.0, 2.0, 3.0})),
               std::invalid_argument);
  EXPECT_THROW(vec({1.0, 2.0})->mul(vec({1.0, 2.0, 3.0})),
               std::invalid_argument);
  EXPECT_THROW(vec({1.0, 2.0})->matmul(vec({1.0, 2.0})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Tensor::from_values(
                   std::vector<size_t>{2, 2}, std::vector<scalar_t>{1.0})),
               std::invalid_argument);

  auto a = Tensor::from_values(std::vector<size_t>{2, 3},
                               std::vector<scalar_t>(6, 1.0));
  EXPECT_THROW(a->matmul(a), std::invalid_argument);
}
