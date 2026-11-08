#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
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

TEST(SampleOrder, ConsumedCountsAcrossEpochs) {
  SampleOrder order({100}, 8, 1);
  const size_t per_epoch = order.samples_per_epoch();
  EXPECT_EQ(order.consumed(), 0u);
  draw(order, per_epoch + 3);
  EXPECT_EQ(order.epoch(), 1u);
  EXPECT_EQ(order.consumed(), per_epoch + 3);
}

TEST(SampleOrder, SeekMatchesDrawing) {
  const size_t per_epoch = SampleOrder({100}, 8, 1).samples_per_epoch();
  for (size_t skip : {size_t{0}, size_t{5}, per_epoch, (2 * per_epoch) + 4}) {
    SampleOrder drawn({100}, 8, 1);
    draw(drawn, skip);
    SampleOrder sought({100}, 8, 1);
    sought.seek(skip);
    EXPECT_EQ(sought.consumed(), skip);
    EXPECT_EQ(draw(drawn, 2 * per_epoch), draw(sought, 2 * per_epoch))
        << "skip " << skip;
  }
}

TEST(SampleOrder, ResumeMatchesUninterruptedRun) {
  const std::vector<size_t> spans{500, 300};
  SampleOrder uninterrupted(spans, 8, 9);
  const size_t before = (2 * uninterrupted.samples_per_epoch()) + 7;
  const size_t after = 3 * uninterrupted.samples_per_epoch();
  draw(uninterrupted, before);
  auto expected = draw(uninterrupted, after);

  SampleOrder first(spans, 8, 9);
  draw(first, before);
  std::stringstream state;
  first.write_state(state);

  SampleOrder resumed(spans, 8, 9);
  resumed.read_state(state);
  EXPECT_EQ(resumed.consumed(), before);
  EXPECT_EQ(draw(resumed, after), expected);
}

TEST(SampleOrder, RejectsMismatchedState) {
  SampleOrder saved({500, 300}, 8, 9);
  draw(saved, 10);
  std::stringstream state;
  saved.write_state(state);
  const std::string bytes = state.str();

  auto load_into = [&](SampleOrder order, const std::string &data) {
    std::stringstream in(data);
    order.read_state(in);
  };
  EXPECT_NO_THROW(load_into(SampleOrder({500, 300}, 8, 9), bytes));
  EXPECT_THROW(load_into(SampleOrder({500, 300}, 8, 10), bytes),
               std::runtime_error);
  EXPECT_THROW(load_into(SampleOrder({500, 300}, 16, 9), bytes),
               std::runtime_error);
  EXPECT_THROW(load_into(SampleOrder({500, 400}, 8, 9), bytes),
               std::runtime_error);
  EXPECT_THROW(load_into(SampleOrder({500, 300}, 8, 9), bytes.substr(0, 20)),
               std::runtime_error);
  EXPECT_THROW(load_into(SampleOrder({500, 300}, 8, 9), "not a state file!!"),
               std::runtime_error);
}
