#include "gpt/Model.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "gpt/GPTConfig.h"
#include "micrograd/Random.h"
#include "micrograd/Tensor.h"
#include "micrograd/nn/Loss.h"
#include "micrograd/nn/Optim.h"

using namespace micrograd;
using micrograd::gpt::ArchOptions;
using micrograd::gpt::GPTConfig;
using micrograd::gpt::Model;

namespace {

constexpr size_t kVocab = 32;
constexpr size_t kSeq = 8;

GPTConfig TinyConfig() {
  GPTConfig config{};
  config.depth = 2;
  config.n_layer = 2;
  config.n_embd = 16;
  config.n_head = 2;
  config.n_kv_head = 2;
  config.block_size = kSeq;
  config.vocab_size = kVocab;
  return config;
}

struct Variant {
  std::string name;
  GPTConfig config;
};

std::vector<Variant> Variants() {
  std::vector<Variant> variants;
  variants.push_back({"Default", TinyConfig()});
  GPTConfig rotary = TinyConfig();
  rotary.arch.rotary = true;
  variants.push_back({"Rotary", rotary});
  GPTConfig qk_norm = TinyConfig();
  qk_norm.arch.qk_norm = true;
  variants.push_back({"QkNorm", qk_norm});
  GPTConfig relu_squared = TinyConfig();
  relu_squared.arch.relu_squared = true;
  variants.push_back({"ReluSquared", relu_squared});
  GPTConfig padded = TinyConfig();
  padded.padded_vocab_size = 40;
  variants.push_back({"PaddedVocab", padded});
  GPTConfig no_recompute = TinyConfig();
  no_recompute.recompute_activations = false;
  variants.push_back({"NoRecompute", no_recompute});
  GPTConfig all = TinyConfig();
  all.arch = {.rotary = true, .qk_norm = true, .relu_squared = true};
  all.padded_vocab_size = 40;
  variants.push_back({"AllSwitches", all});
  return variants;
}

std::vector<size_t> TokenIds(size_t batch, size_t offset) {
  std::vector<size_t> ids(batch * kSeq);
  for (size_t i = 0; i < ids.size(); i++) {
    ids[i] = ((i + offset) * 13 + 5) % kVocab;
  }
  return ids;
}

std::shared_ptr<Tensor> TokenTensor(const std::vector<size_t> &ids) {
  std::vector<scalar_t> values(ids.begin(), ids.end());
  return Tensor::from_values({ids.size() / kSeq, kSeq}, std::move(values));
}

std::vector<size_t> NextTokens(const std::vector<size_t> &ids) {
  std::vector<size_t> targets(ids.size());
  for (size_t i = 0; i < ids.size(); i++) {
    targets[i] = (ids[i] * 7 + 3) % kVocab;
  }
  return targets;
}

std::shared_ptr<Tensor> Loss(Model &model, const std::vector<size_t> &ids) {
  return cross_entropy(model.forward(TokenTensor(ids)), NextTokens(ids),
                       kVocab);
}

}  // namespace

class ModelVariantTest : public ::testing::TestWithParam<Variant> {};

TEST_P(ModelVariantTest, CausalLogitsIgnoreLaterTokens) {
  manual_seed(3);
  Model model(kVocab, GetParam().config);
  std::vector<size_t> ids = TokenIds(2, 0);
  const auto before = model.forward(TokenTensor(ids));
  constexpr size_t kChanged = 5;
  for (size_t b = 0; b < 2; b++) {
    ids[(b * kSeq) + kChanged] = (ids[(b * kSeq) + kChanged] + 1) % kVocab;
  }
  const auto after = model.forward(TokenTensor(ids));

  const size_t width = before->shape()[2];
  const auto x = before->data();
  const auto y = after->data();
  bool later_changed = false;
  for (size_t b = 0; b < 2; b++) {
    for (size_t t = 0; t < kSeq; t++) {
      for (size_t v = 0; v < width; v++) {
        const size_t i = (((b * kSeq) + t) * width) + v;
        if (t < kChanged) {
          ASSERT_EQ(x[i], y[i]) << "b=" << b << " t=" << t << " v=" << v;
        } else if (x[i] != y[i]) {
          later_changed = true;
        }
      }
    }
  }
  EXPECT_TRUE(later_changed);
}

TEST_P(ModelVariantTest, BackwardMatchesFiniteDifferences) {
  constexpr size_t kSamples = 3;
  constexpr scalar_t kEps = 3e-3f;
  manual_seed(5);
  Model model(kVocab, GetParam().config);
  const std::vector<size_t> ids = TokenIds(2, 1);
  Loss(model, ids)->backward();

  for (const auto &[name, parameter] : model.named_parameters()) {
    const std::vector<scalar_t> analytic(parameter->grad().begin(),
                                         parameter->grad().end());
    const std::span<scalar_t> values = parameter->data();
    for (size_t s = 0; s < kSamples; s++) {
      const size_t j = ((2 * s + 1) * values.size()) / (2 * kSamples);
      const scalar_t original = values[j];
      values[j] = original + kEps;
      const double up = Loss(model, ids)->at({});
      values[j] = original - kEps;
      const double down = Loss(model, ids)->at({});
      values[j] = original;

      const double numeric = (up - down) / (2.0 * kEps);
      const double expected = analytic[j];
      const double scale = std::max(std::abs(expected), std::abs(numeric));
      EXPECT_NEAR(expected, numeric, 2e-4 + (2e-2 * scale))
          << name << " index " << j;
    }
  }
}

TEST_P(ModelVariantTest, AccumulatedMicroBatchesMatchFullBatch) {
  constexpr size_t kMicroBatches = 2;
  constexpr size_t kMicroBatch = 2;
  manual_seed(7);
  Model model(kVocab, GetParam().config);
  const auto parameters = model.parameters();
  const std::vector<size_t> ids = TokenIds(kMicroBatches * kMicroBatch, 2);

  Loss(model, ids)->backward();
  std::vector<std::vector<scalar_t>> full;
  for (const auto &parameter : parameters) {
    full.emplace_back(parameter->grad().begin(), parameter->grad().end());
    parameter->zero_grad();
  }

  const size_t micro_size = kMicroBatch * kSeq;
  for (size_t m = 0; m < kMicroBatches; m++) {
    const std::vector<size_t> micro(
        ids.begin() + static_cast<std::ptrdiff_t>(m * micro_size),
        ids.begin() + static_cast<std::ptrdiff_t>((m + 1) * micro_size));
    Loss(model, micro)
        ->mul(1.0f / static_cast<scalar_t>(kMicroBatches))
        ->backward();
  }

  for (size_t p = 0; p < parameters.size(); p++) {
    const auto accumulated = parameters[p]->grad();
    for (size_t j = 0; j < accumulated.size(); j++) {
      EXPECT_NEAR(accumulated[j], full[p][j],
                  1e-6f + (1e-4f * std::abs(full[p][j])))
          << "parameter " << p << " index " << j;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Archs, ModelVariantTest,
                         ::testing::ValuesIn(Variants()),
                         [](const ::testing::TestParamInfo<Variant> &param) {
                           return param.param.name;
                         });

TEST(ModelTest, OverfitsOneBatch) {
  manual_seed(1);
  Model model(kVocab, TinyConfig());
  AdamW optimizer(model.parameters(), 1e-2f);
  const std::vector<size_t> ids = TokenIds(2, 3);

  scalar_t loss = 0.0f;
  for (size_t step = 0; step < 150; step++) {
    auto value = Loss(model, ids);
    loss = value->at({});
    optimizer.zero_grad();
    value->backward();
    optimizer.step();
  }
  EXPECT_LT(loss, 0.05f);
}

TEST(ModelTest, DISABLED_InitialLossIsNearLogVocab) {
  manual_seed(2);
  GPTConfig config = TinyConfig();
  config.n_layer = 1;
  config.n_embd = 256;
  config.n_head = 4;
  Model model(kVocab, config);
  const scalar_t loss = Loss(model, TokenIds(4, 0))->at({});
  const scalar_t expected = std::log(static_cast<scalar_t>(kVocab));
  EXPECT_NEAR(loss, expected, 0.05f * expected);
}
