#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

#include "micrograd/NN.h"
#include "micrograd/Tensor.h"

using namespace micrograd;

TEST(Serialization, SaveLoadRoundTrip) {
  const std::string path = ::testing::TempDir() + "round_trip.bin";
  Linear saved(3, 2);
  save(path, saved);

  Linear loaded(3, 2);
  load(path, loaded);

  EXPECT_TRUE(
      std::ranges::equal(loaded.weights()->data(), saved.weights()->data()));
  EXPECT_TRUE(std::ranges::equal(loaded.bias()->data(), saved.bias()->data()));
}

TEST(Serialization, LoadRejectsShapeLargerThanFile) {
  const std::string path = ::testing::TempDir() + "huge_shape.bin";
  {
    std::ofstream file(path, std::ios::binary);
    const std::array<uint32_t, 4> header = {0x4d47534e, 1, 1, 1};
    file.write(reinterpret_cast<const char *>(header.data()), sizeof(header));
    file.write("w", 1);
    const uint32_t ndim = 1;
    file.write(reinterpret_cast<const char *>(&ndim), sizeof(ndim));
    const uint64_t huge_dim = uint64_t{1} << 40;
    file.write(reinterpret_cast<const char *>(&huge_dim), sizeof(huge_dim));
  }

  Linear layer(3, 2);
  EXPECT_THROW(load(path, layer), std::runtime_error);
}
