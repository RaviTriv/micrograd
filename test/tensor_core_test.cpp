#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "test_helpers.h"

using namespace micrograd;
using namespace micrograd::test_helpers;

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

TEST(TensorCore, InvalidShapesThrow) {
  EXPECT_THROW(vec({1.0, 2.0})->add(vec({1.0, 2.0, 3.0})),
               std::invalid_argument);
  EXPECT_THROW(vec({1.0, 2.0})->mul(vec({1.0, 2.0, 3.0})),
               std::invalid_argument);
  EXPECT_THROW(vec({1.0, 2.0})->matmul(vec({1.0, 2.0})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(std::make_shared<Tensor>(
                   std::vector<size_t>{2, 2}, std::vector<scalar_t>{1.0})),
               std::invalid_argument);

  auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 3},
                                    std::vector<scalar_t>(6, 1.0));
  EXPECT_THROW(a->matmul(a), std::invalid_argument);
}
