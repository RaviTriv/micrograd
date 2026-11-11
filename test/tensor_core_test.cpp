#include <gtest/gtest.h>

#include <limits>
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

TEST(TensorFactories, ZerosFillsDataWithZeros) {
  auto t = Tensor::zeros({2, 3});

  EXPECT_EQ(t->shape(), (std::vector<size_t>{2, 3}));
  EXPECT_EQ(Values(t), std::vector<scalar_t>(6, 0.0f));
  EXPECT_FALSE(t->requires_grad());
}

TEST(TensorFactories, NewTensorsHaveNoGradBuffer) {
  EXPECT_FALSE(Tensor::zeros({2, 3})->has_grad());
  EXPECT_FALSE(Tensor::from_values({2}, {1, 2})->has_grad());
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

TEST(TensorFactories, RejectsShapeWhoseElementCountOverflows) {
  const size_t huge = size_t{1} << 32;

  EXPECT_THROW(static_cast<void>(Tensor::zeros({huge, huge})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Tensor::from_values({huge, huge}, {})),
               std::invalid_argument);
}

TEST(TensorFactories, RejectsShapeWhoseByteCountOverflows) {
  const size_t over_byte_limit =
      (std::numeric_limits<size_t>::max() / sizeof(scalar_t)) + 1;

  EXPECT_THROW(static_cast<void>(Tensor::zeros({over_byte_limit})),
               std::invalid_argument);
}

TEST(TensorFactories, ZeroDimensionMakesAnyShapeEmpty) {
  const size_t largest = std::numeric_limits<size_t>::max();

  auto t = Tensor::zeros({largest, largest, 0});

  EXPECT_EQ(t->size(), 0u);
  EXPECT_EQ(t->shape(), (std::vector<size_t>{largest, largest, 0}));
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

TEST(TensorGrad, ConstReadsDoNotAllocateAGradient) {
  auto t = Make({2, 3}, Arange(6));
  const Tensor &view = *t;

  EXPECT_FALSE(view.grad_if_present().has_value());
  EXPECT_EQ(view.grad_at({1, 2}), 0.0f);
  EXPECT_THROW(static_cast<void>(view.grad_at({0, 3})), std::out_of_range);
  EXPECT_FALSE(t->has_grad());
}

TEST(TensorGrad, GradIfPresentSeesAnExistingGradient) {
  auto t = Make({2, 3}, Arange(6));
  t->grad_at({1, 2}) = 7.0f;
  const Tensor &view = *t;

  const auto grad = view.grad_if_present();
  ASSERT_TRUE(grad.has_value());
  EXPECT_EQ(grad->size(), 6U);
  EXPECT_EQ((*grad)[5], 7.0f);
  EXPECT_EQ(view.grad_at({1, 2}), 7.0f);
}

TEST(TensorReshaping, Reshape) {
  auto t = Make({12}, Arange(12));

  auto reshaped = t->reshape({3, 4});

  EXPECT_EQ(reshaped->shape(), (std::vector<size_t>{3, 4}));
  EXPECT_EQ(Values(reshaped), Arange(12));
  EXPECT_EQ(reshaped->at({1, 2}), 6.0f);
  EXPECT_EQ(reshaped->at({2, 3}), 11.0f);
}

TEST(TensorReshaping, RejectsInvalidShapes) {
  auto t = Make({6}, Arange(6));

  EXPECT_THROW(t->reshape({-2, 3}), std::invalid_argument);
  EXPECT_THROW(t->reshape({-1, -1}), std::invalid_argument);
  EXPECT_THROW(t->reshape({2, 2}), std::invalid_argument);
  EXPECT_THROW(t->reshape({4, -1}), std::invalid_argument);
  EXPECT_THROW(t->reshape({0, -1}), std::invalid_argument);
}

TEST(TensorReshaping, NegativeDimensionErrorSaysWhatIsAllowed) {
  auto t = Make({6}, Arange(6));

  try {
    t->reshape({-2, 3});
    FAIL() << "reshape should reject -2";
  } catch (const std::invalid_argument &error) {
    EXPECT_STREQ(error.what(), "Reshape dimensions must be -1 or non-negative");
  }
}

TEST(TensorReshaping, EmptyTensorReshapesWithZeroDimension) {
  auto t = Tensor::zeros({0});

  EXPECT_EQ(t->reshape({0, 5})->shape(), (std::vector<size_t>{0, 5}));
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

TEST(TensorReductions, SumReturnsZeroDimScalar) {
  auto t = Make({2, 3}, {1, 2, 3, 4, 5, 6});

  auto total = t->sum();

  EXPECT_TRUE(total->shape().empty());
  EXPECT_EQ(total->size(), 1u);
  EXPECT_EQ(total->at({}), 21.0f);
}

TEST(TensorReductions, ReducingTheOnlyDimensionReturnsZeroDim) {
  auto t = Make({3}, {1, 5, 3});

  EXPECT_TRUE(t->sum(0)->shape().empty());
  EXPECT_EQ(t->sum(0)->at({}), 9.0f);
  EXPECT_TRUE(t->mean(0)->shape().empty());
  EXPECT_EQ(t->mean(0)->at({}), 3.0f);
  EXPECT_TRUE(t->max(0)->shape().empty());
  EXPECT_EQ(t->max(0)->at({}), 5.0f);
  EXPECT_TRUE(t->argmax(0)->shape().empty());
  EXPECT_EQ(t->argmax(0)->at({}), 1.0f);
}

TEST(TensorReductions, KeepdimOnTheOnlyDimensionKeepsOneDimension) {
  auto t = Make({3}, {1, 5, 3});

  EXPECT_EQ(t->sum(0, true)->shape(), (std::vector<size_t>{1}));
  EXPECT_EQ(t->sum(0, true)->at({0}), 9.0f);
}

TEST(TensorReductions, ReducesAlongEachDimension) {
  auto t = Make({2, 3}, {1, 2, 3, 4, 5, 6});

  EXPECT_EQ(Values(t->sum(0)), (std::vector<scalar_t>{5, 7, 9}));
  EXPECT_EQ(t->sum(0)->shape(), (std::vector<size_t>{3}));
  EXPECT_EQ(Values(t->sum(1)), (std::vector<scalar_t>{6, 15}));
  EXPECT_EQ(t->sum(1, true)->shape(), (std::vector<size_t>{2, 1}));
  EXPECT_EQ(Values(t->mean(1)), (std::vector<scalar_t>{2, 5}));
  EXPECT_EQ(Values(t->max(0)), (std::vector<scalar_t>{4, 5, 6}));
  EXPECT_EQ(Values(t->argmax(1)), (std::vector<scalar_t>{2, 2}));
}

TEST(TensorReductions, ZeroDimBroadcastsAgainstAnyShape) {
  auto t = Make({2, 3}, {1, 2, 3, 4, 5, 6});
  auto scalar_value = Tensor::zeros({});
  scalar_value->at({}) = 10.0f;

  auto left = t->add(scalar_value);
  auto right = scalar_value->add(t);

  EXPECT_EQ(left->shape(), (std::vector<size_t>{2, 3}));
  EXPECT_EQ(Values(left), (std::vector<scalar_t>{11, 12, 13, 14, 15, 16}));
  EXPECT_EQ(right->shape(), (std::vector<size_t>{2, 3}));
  EXPECT_EQ(Values(right), (std::vector<scalar_t>{11, 12, 13, 14, 15, 16}));
}

TEST(TensorReductions, ZeroDimReshapesToOneElement) {
  auto total = Make({2}, {1, 2})->sum();

  auto reshaped = total->reshape({1, 1});

  EXPECT_EQ(reshaped->shape(), (std::vector<size_t>{1, 1}));
  EXPECT_EQ(reshaped->at({0, 0}), 3.0f);
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

TEST(TensorClamp, ClampsToTheGivenRange) {
  EXPECT_EQ(Values(Make({4}, {-1.0f, 0.25f, 0.75f, 2.0f})->clamp(0.0f, 1.0f)),
            (std::vector<scalar_t>{0.0f, 0.25f, 0.75f, 1.0f}));
}

TEST(TensorClamp, GradientPassesInsideRangeAndAtBounds) {
  auto x = leaf({5}, {-1.0f, 0.0f, 0.5f, 1.0f, 2.0f});
  x->clamp(0.0f, 1.0f)->sum()->backward();
  const std::vector<scalar_t> grad(x->grad().begin(), x->grad().end());
  EXPECT_EQ(grad, (std::vector<scalar_t>{0.0f, 1.0f, 1.0f, 1.0f, 0.0f}));
}

TEST(TensorClamp, RejectsMinGreaterThanMax) {
  EXPECT_THROW(Make({1}, {0.0f})->clamp(1.0f, 0.0f), std::invalid_argument);
}

TEST(TensorReductions, MaxOverZeroLengthDimensionThrows) {
  auto t = Tensor::zeros({2, 0});

  EXPECT_THROW(t->max(1), std::invalid_argument);
  EXPECT_THROW(t->max(-1, true), std::invalid_argument);
  EXPECT_THROW(t->argmax(1), std::invalid_argument);
  EXPECT_EQ(t->max(0)->shape(), (std::vector<size_t>{0}));
}

TEST(TensorPow, ZeroExponentHasZeroGradientAtZero) {
  auto x = leaf({3}, {0.0f, 2.0f, -1.5f});
  x->pow(0.0f)->sum()->backward();

  const std::vector<scalar_t> grad(x->grad().begin(), x->grad().end());
  EXPECT_EQ(grad, (std::vector<scalar_t>{0.0f, 0.0f, 0.0f}));
}

TEST(TensorAddBias, RowLoopMatchesPerElementReferenceExactly) {
  const size_t rows = 5;
  const size_t cols = 7;
  std::vector<scalar_t> x_values(rows * cols);
  std::vector<scalar_t> b_values(cols);
  std::vector<scalar_t> seed(rows * cols);
  for (size_t i = 0; i < x_values.size(); i++) {
    x_values[i] = 0.1f * static_cast<scalar_t>(i) - 1.3f;
    seed[i] = 0.37f * static_cast<scalar_t>(i % 11) + 0.01f;
  }
  for (size_t c = 0; c < cols; c++) {
    b_values[c] = 0.7f - 0.3f * static_cast<scalar_t>(c);
  }

  auto x = leaf({rows, cols}, x_values);
  auto b = leaf({cols}, b_values);
  auto out = x->add_bias(b);
  out->backward(*Tensor::from_values({rows, cols}, seed));

  std::vector<scalar_t> expected_out(rows * cols);
  std::vector<scalar_t> expected_b_grad(cols, 0.0f);
  for (size_t i = 0; i < expected_out.size(); i++) {
    expected_out[i] = x_values[i] + b_values[i % cols];
    expected_b_grad[i % cols] += seed[i];
  }
  EXPECT_EQ(Values(out), expected_out);
  EXPECT_EQ(std::vector<scalar_t>(x->grad().begin(), x->grad().end()), seed);
  EXPECT_EQ(std::vector<scalar_t>(b->grad().begin(), b->grad().end()),
            expected_b_grad);
}
