#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpt/Dataset.h"
#include "gpt/SampleOrder.h"

using micrograd::gpt::Dataset;
using micrograd::gpt::SampleOrder;

namespace {

std::vector<std::pair<size_t, size_t>> draw(SampleOrder &order, size_t count) {
  std::vector<std::pair<size_t, size_t>> drawn;
  drawn.reserve(count);
  for (size_t i = 0; i < count; i++) {
    SampleOrder::Location location = order.next();
    drawn.emplace_back(location.span, location.offset);
  }
  return drawn;
}

}  // namespace

TEST(SampleOrder, VisitsEverySampleOncePerEpoch) {
  const std::vector<size_t> spans{100, 0, 35, 7};
  const size_t block = 8;
  SampleOrder order(spans, block, 42);
  ASSERT_EQ(order.samples_per_epoch(), 14u);

  for (size_t epoch = 0; epoch < 3; epoch++) {
    auto drawn = draw(order, order.samples_per_epoch());
    EXPECT_EQ(order.epoch(), epoch);
    EXPECT_EQ(order.position(), order.samples_per_epoch());

    std::set<std::pair<size_t, size_t>> unique(drawn.begin(), drawn.end());
    EXPECT_EQ(unique.size(), drawn.size());

    const size_t shift = drawn.front().second % block;
    for (const auto &[span, offset] : drawn) {
      ASSERT_TRUE(span == 0 || span == 2);
      EXPECT_EQ(offset % block, shift);
      EXPECT_LT(offset + block, spans[span]);
    }
  }
}

TEST(SampleOrder, SameSeedGivesSameOrder) {
  SampleOrder a({10000}, 4, 7);
  SampleOrder b({10000}, 4, 7);
  const size_t count = 3 * a.samples_per_epoch();
  EXPECT_EQ(draw(a, count), draw(b, count));
}

TEST(SampleOrder, DifferentSeedGivesDifferentOrder) {
  SampleOrder a({10000}, 4, 7);
  SampleOrder b({10000}, 4, 8);
  const size_t count = a.samples_per_epoch();
  EXPECT_NE(draw(a, count), draw(b, count));
}

TEST(SampleOrder, ReshufflesEachEpoch) {
  SampleOrder order({10000}, 4, 7);
  const size_t count = order.samples_per_epoch();
  auto first = draw(order, count);
  auto second = draw(order, count);
  EXPECT_EQ(order.epoch(), 1u);
  EXPECT_NE(first, second);
}

TEST(SampleOrder, RejectsUnusableInput) {
  EXPECT_THROW(SampleOrder({100}, 0, 1), std::invalid_argument);
  EXPECT_THROW(SampleOrder({15, 3}, 8, 1), std::runtime_error);
  EXPECT_THROW(SampleOrder({}, 8, 1), std::runtime_error);
}

TEST(SampleOrder, DatasetTrainBatchTargetsAreNextTokens) {
  const std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string text;
  for (int i = 0; i < 40; i++) {
    text += alphabet;
  }
  const auto path = std::filesystem::temp_directory_path() /
                    "micrograd_sample_order_test.txt";
  {
    std::ofstream file(path, std::ios::binary);
    file << text;
  }
  Dataset dataset(path.string(), 0.1);
  std::filesystem::remove(path);

  const size_t block = 8;
  const size_t batch_size = 4;
  SampleOrder order(dataset.train_span_tokens(), block, 3);
  Dataset::Batch batch = dataset.train_batch(batch_size, order);
  ASSERT_EQ(batch.inputs.size(), batch_size * block);
  ASSERT_EQ(batch.targets.size(), batch_size * block);

  for (size_t i = 0; i < batch.inputs.size(); i++) {
    const char input = dataset.decode(batch.inputs[i]);
    const char target = dataset.decode(batch.targets[i]);
    const size_t next = (alphabet.find(input) + 1) % alphabet.size();
    EXPECT_EQ(target, alphabet[next]);
    if (i % block != block - 1) {
      EXPECT_EQ(batch.targets[i], batch.inputs[i + 1]);
    }
  }

  SampleOrder wrong({0, 100}, block, 3);
  EXPECT_THROW(dataset.train_batch(1, wrong), std::out_of_range);
}
