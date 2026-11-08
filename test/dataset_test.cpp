#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpt/Dataset.h"
#include "gpt/SampleOrder.h"

using micrograd::gpt::Dataset;
using micrograd::gpt::SampleOrder;
using micrograd::gpt::ShardedDataset;

namespace {

std::filesystem::path temp_path(const std::string &name) {
  return std::filesystem::temp_directory_path() / ("micrograd_" + name);
}

std::string write_shard(const std::string &name, uint16_t first,
                        size_t count) {
  std::array<int32_t, 256> header{};
  header[0] = 20240520;
  header[1] = 1;
  header[2] = static_cast<int32_t>(count);
  std::vector<uint16_t> tokens(count);
  for (size_t i = 0; i < count; i++) {
    tokens[i] = static_cast<uint16_t>(first + i);
  }
  const auto path = temp_path(name);
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char *>(header.data()),
             static_cast<std::streamsize>(sizeof(header)));
  file.write(reinterpret_cast<const char *>(tokens.data()),
             static_cast<std::streamsize>(tokens.size() * sizeof(uint16_t)));
  return path.string();
}

void expect_window(const Dataset::Batch &batch, size_t row, size_t block,
                   size_t first_token) {
  for (size_t col = 0; col < block; col++) {
    EXPECT_EQ(batch.inputs[(row * block) + col], first_token + col)
        << "row " << row << " col " << col;
    EXPECT_EQ(batch.targets[(row * block) + col], first_token + col + 1)
        << "row " << row << " col " << col;
  }
}

}  // namespace

TEST(Dataset, ValBatchesWalkTheValSplitInOrder) {
  const std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string text;
  for (int i = 0; i < 40; i++) {
    text += alphabet;
  }
  const auto path = temp_path("val_split_test.txt");
  {
    std::ofstream file(path, std::ios::binary);
    file << text;
  }
  Dataset dataset(path.string(), 0.25);
  std::filesystem::remove(path);

  const size_t block = 8;
  const size_t batch_size = 4;
  const size_t val_start = 780;
  ASSERT_EQ(dataset.val_windows(block), 32u);
  const std::vector<size_t> encoded = dataset.encode(text);

  for (size_t b = 0; b < 8; b++) {
    Dataset::Batch batch = dataset.val_batch(batch_size, block, b);
    for (size_t row = 0; row < batch_size; row++) {
      const size_t start = val_start + (((b * batch_size) + row) * block);
      for (size_t col = 0; col < block; col++) {
        EXPECT_EQ(batch.inputs[(row * block) + col], encoded[start + col]);
        EXPECT_EQ(batch.targets[(row * block) + col],
                  encoded[start + col + 1]);
      }
    }
  }
  EXPECT_THROW(dataset.val_batch(batch_size, block, 8), std::out_of_range);
}

TEST(ShardedDataset, ValBatchesCoverEveryShardInOrder) {
  const std::string train = write_shard("train_000.bin", 0, 200);
  const std::string val_a = write_shard("val_000.bin", 1000, 41);
  const std::string val_b = write_shard("val_001.bin", 2000, 17);
  {
    ShardedDataset dataset({train}, {val_a, val_b});
    const size_t block = 8;
    ASSERT_EQ(dataset.val_windows(block), 7u);
    EXPECT_EQ(dataset.token_count(Dataset::Split::kVal), 58u);

    Dataset::Batch all = dataset.val_batch(7, block, 0);
    for (size_t row = 0; row < 5; row++) {
      expect_window(all, row, block, 1000 + (row * block));
    }
    expect_window(all, 5, block, 2000);
    expect_window(all, 6, block, 2008);
    EXPECT_THROW(dataset.val_batch(7, block, 1), std::out_of_range);

    Dataset::Batch crossing = dataset.val_batch(3, block, 1);
    expect_window(crossing, 0, block, 1024);
    expect_window(crossing, 1, block, 1032);
    expect_window(crossing, 2, block, 2000);

    SampleOrder order(dataset.train_span_tokens(), block, 5);
    EXPECT_EQ(order.samples_per_epoch(), 24u);
    Dataset::Batch batch = dataset.train_batch(4, order);
    for (size_t row = 0; row < 4; row++) {
      expect_window(batch, row, block, batch.inputs[row * block]);
    }
  }
  std::filesystem::remove(train);
  std::filesystem::remove(val_a);
  std::filesystem::remove(val_b);
}
