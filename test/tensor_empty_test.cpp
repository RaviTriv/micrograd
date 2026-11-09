#include <gtest/gtest.h>

#include <vector>

#include "micrograd/Tensor.h"

using namespace micrograd;

TEST(TensorEmpty, HasRequestedShapeAndDevice) {
  auto tensor = Tensor::empty({2, 3, 4}, Device::CPU);

  EXPECT_EQ(tensor->shape(), (std::vector<size_t>{2, 3, 4}));
  EXPECT_EQ(tensor->size(), 24u);
  EXPECT_EQ(tensor->backend(), Device::CPU);
  EXPECT_EQ(tensor->data_storage().bytes(), 24u * sizeof(scalar_t));
}

TEST(TensorEmpty, CpuOpsStillStartFromZeroedOutputs) {
  auto a = Tensor::from_values({2, 2}, {1, 2, 3, 4});
  auto out = a->add(a);

  EXPECT_EQ(out->data()[0], 2.0f);
  EXPECT_EQ(out->data()[3], 8.0f);
}
