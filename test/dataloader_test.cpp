#include "micrograd/data/DataLoader.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "micrograd/Tensor.h"

using micrograd::scalar_t;
using micrograd::Tensor;
using micrograd::data::Batch;
using micrograd::data::class_indices;
using micrograd::data::DataLoader;
using micrograd::data::TensorDataset;

namespace {

TensorDataset numbered_dataset(size_t count) {
  std::vector<scalar_t> inputs(count * 3);
  std::vector<scalar_t> targets(count);
  for (size_t i = 0; i < count; i++) {
    for (size_t j = 0; j < 3; j++) {
      inputs[(i * 3) + j] = static_cast<scalar_t>((i * 10) + j);
    }
    targets[i] = static_cast<scalar_t>(i);
  }
  auto x = Tensor::from_values({count, 3}, inputs);
  auto y = Tensor::from_values({count}, targets);
  return {*x, *y};
}

std::vector<size_t> epoch_targets(const DataLoader &loader) {
  std::vector<size_t> seen;
  for (const Batch &batch : loader) {
    auto indices = class_indices(*batch.targets);
    seen.insert(seen.end(), indices.begin(), indices.end());
  }
  return seen;
}

}  // namespace

TEST(DataLoader, KeepsShortLastBatch) {
  TensorDataset dataset = numbered_dataset(10);
  DataLoader loader(dataset, 4, false);
  ASSERT_EQ(loader.size(), 3u);

  std::vector<size_t> rows;
  for (const Batch &batch : loader) {
    rows.push_back(batch.inputs->shape()[0]);
    EXPECT_EQ(batch.inputs->shape(), (std::vector<size_t>{rows.back(), 3}));
    EXPECT_EQ(batch.targets->shape(), (std::vector<size_t>{rows.back()}));
  }
  EXPECT_EQ(rows, (std::vector<size_t>{4, 4, 2}));
  EXPECT_THROW(loader.batch(3), std::out_of_range);
}

TEST(DataLoader, WithoutShuffleWalksInOrder) {
  TensorDataset dataset = numbered_dataset(7);
  DataLoader loader(dataset, 3, false);
  std::vector<size_t> expected(7);
  std::iota(expected.begin(), expected.end(), size_t{0});
  EXPECT_EQ(epoch_targets(loader), expected);
  loader.set_epoch(4);
  EXPECT_EQ(epoch_targets(loader), expected);
}

TEST(DataLoader, ShuffleVisitsEverySampleOncePerEpoch) {
  TensorDataset dataset = numbered_dataset(101);
  DataLoader loader(dataset, 8, true, 42);
  std::vector<size_t> expected(101);
  std::iota(expected.begin(), expected.end(), size_t{0});

  std::vector<std::vector<size_t>> epochs;
  for (size_t epoch = 0; epoch < 3; epoch++) {
    loader.set_epoch(epoch);
    std::vector<size_t> seen = epoch_targets(loader);
    epochs.push_back(seen);
    std::ranges::sort(seen);
    EXPECT_EQ(seen, expected) << "epoch " << epoch;
  }
  EXPECT_NE(epochs[0], expected);
  EXPECT_NE(epochs[0], epochs[1]);
  EXPECT_NE(epochs[1], epochs[2]);
}

TEST(DataLoader, OrderDependsOnlyOnSeedAndEpoch) {
  TensorDataset dataset = numbered_dataset(50);
  DataLoader a(dataset, 5, true, 7);
  DataLoader b(dataset, 5, true, 7);
  DataLoader other_seed(dataset, 5, true, 8);
  a.set_epoch(3);
  b.set_epoch(1);
  b.set_epoch(3);
  other_seed.set_epoch(3);
  EXPECT_EQ(a.order(), b.order());
  EXPECT_NE(a.order(), other_seed.order());
}

TEST(DataLoader, RowsKeepInputsAndTargetsAligned) {
  TensorDataset dataset = numbered_dataset(23);
  DataLoader loader(dataset, 6, true, 3);
  for (const Batch &batch : loader) {
    auto targets = class_indices(*batch.targets);
    for (size_t row = 0; row < targets.size(); row++) {
      for (size_t col = 0; col < 3; col++) {
        EXPECT_EQ(batch.inputs->at({row, col}),
                  static_cast<scalar_t>((targets[row] * 10) + col));
      }
    }
  }
}

TEST(DataLoader, KeepsPerSampleShapes) {
  std::vector<scalar_t> pixels(size_t{4} * 2 * 3);
  std::iota(pixels.begin(), pixels.end(), 0.0f);
  auto images = Tensor::from_values({4, 2, 3}, pixels);
  auto one_hot = Tensor::from_values({4, 2}, {1, 0, 0, 1, 1, 0, 0, 1});
  TensorDataset dataset(*images, *one_hot);
  EXPECT_EQ(dataset.input_shape(), (std::vector<size_t>{2, 3}));
  EXPECT_EQ(dataset.target_shape(), (std::vector<size_t>{2}));

  DataLoader loader(dataset, 3, false);
  Batch first = loader.batch(0);
  EXPECT_EQ(first.inputs->shape(), (std::vector<size_t>{3, 2, 3}));
  EXPECT_EQ(first.targets->shape(), (std::vector<size_t>{3, 2}));
  EXPECT_EQ(first.inputs->at({2, 1, 2}), 17.0f);
  Batch last = loader.batch(1);
  EXPECT_EQ(last.inputs->shape(), (std::vector<size_t>{1, 2, 3}));
  EXPECT_EQ(last.inputs->at({0, 0, 0}), 18.0f);
}

TEST(DataLoader, RejectsBadInput) {
  auto x = Tensor::from_values({3, 2}, {0, 1, 2, 3, 4, 5});
  auto y = Tensor::from_values({2}, {0, 1});
  EXPECT_THROW(TensorDataset(*x, *y), std::invalid_argument);

  TensorDataset dataset = numbered_dataset(4);
  EXPECT_THROW(DataLoader(dataset, 0, false), std::invalid_argument);
  EXPECT_THROW(dataset.get(4), std::out_of_range);

  auto fractional = Tensor::from_values({2}, {1.0f, 1.5f});
  auto negative = Tensor::from_values({1}, {-1.0f});
  EXPECT_THROW(class_indices(*fractional), std::invalid_argument);
  EXPECT_THROW(class_indices(*negative), std::invalid_argument);
}

TEST(DataLoader, RefusesTemporaryDatasets) {
  static_assert(
      !std::is_constructible_v<DataLoader, TensorDataset, size_t, bool>);
  static_assert(
      std::is_constructible_v<DataLoader, TensorDataset &, size_t, bool>);
}

TEST(DataLoader, CanBeReassigned) {
  TensorDataset first = numbered_dataset(4);
  TensorDataset second = numbered_dataset(6);
  DataLoader loader(first, 4, false);
  EXPECT_EQ(loader.size(), 1u);

  loader = DataLoader(second, 4, false);
  EXPECT_EQ(loader.size(), 2u);
  EXPECT_EQ(epoch_targets(loader), std::vector<size_t>({0, 1, 2, 3, 4, 5}));
}
