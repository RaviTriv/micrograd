#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpt/GPTConfig.h"
#include "gpt/Model.h"
#include "gpt/ModelFile.h"
#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"
#include "micrograd/nn/Serialize.h"

using namespace micrograd;
using micrograd::gpt::architecture_metadata;
using micrograd::gpt::config_from_metadata;
using micrograd::gpt::ContinuationScore;
using micrograd::gpt::GPTConfig;
using micrograd::gpt::load_model;
using micrograd::gpt::Model;
using micrograd::gpt::read_model_config;
using micrograd::gpt::require_same_architecture;
using micrograd::gpt::save_model;
using micrograd::gpt::score_continuation;
using micrograd::gpt::score_request;

namespace {

constexpr size_t kVocab = 11;

GPTConfig TinyConfig() {
  GPTConfig config;
  config.depth = 1;
  config.n_layer = 1;
  config.n_embd = 16;
  config.n_head = 2;
  config.n_kv_head = 2;
  config.block_size = 8;
  config.vocab_size = kVocab;
  config.matrix_lr = 0.0f;
  config.embedding_lr = 0.0f;
  config.unembedding_lr = 0.0f;
  return config;
}

std::shared_ptr<Tensor> Tokens(const std::vector<int32_t> &tokens) {
  std::vector<scalar_t> values(tokens.begin(), tokens.end());
  return Tensor::from_values({1, tokens.size()}, std::move(values));
}

std::vector<scalar_t> Logits(Model &model, const std::vector<int32_t> &input) {
  const NoGradGuard no_grad;
  auto logits = model.forward(Tokens(input));
  auto data = logits->data();
  return {data.begin(), data.end()};
}

double DirectLogprob(Model &model, const std::vector<int32_t> &input,
                     size_t first_row, const std::vector<int32_t> &targets) {
  const NoGradGuard no_grad;
  auto log_probs = model.forward(Tokens(input))->log_softmax(2);
  double total = 0.0;
  for (size_t i = 0; i < targets.size(); i++) {
    total += static_cast<double>(
        log_probs->at({0, first_row + i, static_cast<size_t>(targets[i])}));
  }
  return total;
}

std::string ConfigError(const StateDictMetadata &metadata) {
  try {
    config_from_metadata(metadata);
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return "";
}

bool IsError(const std::string &response) {
  return response.starts_with(R"({"error": ")") && response.ends_with(R"("})");
}

}  // namespace

TEST(ModelFile, EveryArchitectureRoundTripsFromTheFileAlone) {
  const std::vector<int32_t> input = {3, 1, 4, 1, 5, 9, 2, 6};
  for (unsigned mask = 0; mask < 16; mask++) {
    GPTConfig config = TinyConfig();
    config.arch.rotary = (mask & 1U) != 0;
    config.arch.qk_norm = (mask & 2U) != 0;
    config.arch.relu_squared = (mask & 4U) != 0;
    config.padded_vocab_size = (mask & 8U) != 0 ? 16 : 0;
    Model saved(kVocab, config);
    const std::string path =
        ::testing::TempDir() + "model_file_" + std::to_string(mask) + ".bin";
    save_model(path, saved, config);

    EXPECT_EQ(architecture_metadata(read_model_config(path)),
              architecture_metadata(config));
    std::unique_ptr<Model> loaded = load_model(path, Device::CPU);
    EXPECT_EQ(Logits(*loaded, input), Logits(saved, input)) << "mask " << mask;
  }
}

TEST(ModelFile, RejectsCheckpointWithoutArchitecture) {
  Model model(kVocab, TinyConfig());
  const std::string path = ::testing::TempDir() + "model_file_bare.bin";
  save(path, model);
  try {
    load_model(path, Device::CPU);
    FAIL() << "load_model accepted a checkpoint without architecture";
  } catch (const std::runtime_error &error) {
    EXPECT_STREQ(error.what(),
                 "checkpoint predates architecture metadata; re-save it with "
                 "train_gpt");
  }
}

TEST(ModelFile, NamesEachMissingKey) {
  const StateDictMetadata full = architecture_metadata(TinyConfig());
  for (const auto &[key, value] : full) {
    StateDictMetadata metadata = full;
    metadata.erase(key);
    const std::string error = ConfigError(metadata);
    if (key == "format") {
      EXPECT_NE(error.find("predates architecture metadata"),
                std::string::npos);
    } else {
      EXPECT_EQ(error, "checkpoint architecture is missing key " + key);
    }
  }
}

TEST(ModelFile, NamesEachBadKey) {
  const std::vector<std::pair<std::string, std::string>> bad = {
      {"format", "gpt-arch-0"},   {"n_layer", "0"},
      {"n_embd", "abc"},          {"n_head", "3"},
      {"block_size", "-1"},       {"vocab_size", "0"},
      {"padded_vocab_size", "5"}, {"padded_vocab_size", "16x"},
      {"rotary", "yes"},          {"qk_norm", "1"},
      {"relu_squared", ""},       {"n_layer", "99999999999999999999999"},
  };
  for (const auto &[key, value] : bad) {
    StateDictMetadata metadata = architecture_metadata(TinyConfig());
    metadata[key] = value;
    std::string expected = "checkpoint architecture key " + key;
    expected += " is '" + value + "'";
    EXPECT_TRUE(ConfigError(metadata).starts_with(expected))
        << key << "=" << value << ": " << ConfigError(metadata);
  }
}

TEST(ModelFile, ArchitectureMismatchListsBothValues) {
  GPTConfig flags = TinyConfig();
  flags.arch.rotary = true;
  flags.n_layer = 2;
  try {
    require_same_architecture(TinyConfig(), flags);
    FAIL() << "mismatch was accepted";
  } catch (const std::runtime_error &error) {
    EXPECT_STREQ(error.what(),
                 "checkpoint architecture does not match the flags:\n"
                 "  n_layer: checkpoint 1, flags 2\n"
                 "  rotary: checkpoint false, flags true");
  }
  EXPECT_NO_THROW(require_same_architecture(TinyConfig(), TinyConfig()));
}

TEST(ModelFile, ScoreMatchesFullSequenceForward) {
  GPTConfig config = TinyConfig();
  Model model(kVocab, config);
  ContinuationScore score =
      score_continuation(model, config, Device::CPU, {1, 2, 3}, {4, 5});
  EXPECT_EQ(score.scored, 2U);
  EXPECT_NEAR(score.logprob, DirectLogprob(model, {1, 2, 3, 4, 5}, 2, {4, 5}),
              1e-5);
}

TEST(ModelFile, ScoreDropsContextFromTheLeft) {
  GPTConfig config = TinyConfig();
  Model model(kVocab, config);
  const std::vector<int32_t> context = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  ContinuationScore score =
      score_continuation(model, config, Device::CPU, context, {3, 7, 2});
  EXPECT_EQ(score.scored, 3U);
  EXPECT_NEAR(score.logprob,
              DirectLogprob(model, {4, 5, 6, 7, 8, 9, 3, 7}, 5, {3, 7, 2}),
              1e-5);

  ContinuationScore full = score_continuation(model, config, Device::CPU, {1},
                                              {2, 3, 4, 5, 6, 7, 8, 9});
  EXPECT_EQ(full.scored, 8U);
}

TEST(ModelFile, EmptyContextSkipsTheFirstToken) {
  GPTConfig config = TinyConfig();
  Model model(kVocab, config);
  ContinuationScore score =
      score_continuation(model, config, Device::CPU, {}, {4, 5, 6});
  EXPECT_EQ(score.scored, 2U);
  EXPECT_NEAR(score.logprob, DirectLogprob(model, {4, 5}, 0, {5, 6}), 1e-5);
  EXPECT_THROW(score_continuation(model, config, Device::CPU, {}, {4}),
               std::invalid_argument);
}

TEST(ModelFile, ScoreRejectsBadContinuations) {
  GPTConfig config = TinyConfig();
  Model model(kVocab, config);
  EXPECT_THROW(score_continuation(model, config, Device::CPU, {1}, {}),
               std::invalid_argument);
  EXPECT_THROW(score_continuation(model, config, Device::CPU, {1},
                                  {1, 2, 3, 4, 5, 6, 7, 8, 9}),
               std::invalid_argument);
  EXPECT_THROW(score_continuation(model, config, Device::CPU, {1},
                                  {static_cast<int32_t>(kVocab)}),
               std::invalid_argument);
  EXPECT_THROW(score_continuation(model, config, Device::CPU, {-1}, {1}),
               std::invalid_argument);
}

TEST(ModelFile, ScoreIgnoresPaddedVocabColumns) {
  GPTConfig config = TinyConfig();
  config.padded_vocab_size = 16;
  Model model(kVocab, config);
  const std::vector<int32_t> input = {1, 2, 3};
  const std::vector<scalar_t> logits = Logits(model, input);
  double expected = 0.0;
  const std::vector<int32_t> targets = {3, 4};
  for (size_t i = 0; i < targets.size(); i++) {
    const size_t row = 1 + i;
    double sum_exp = 0.0;
    for (size_t v = 0; v < kVocab; v++) {
      sum_exp += std::exp(static_cast<double>(logits[row * 16 + v]));
    }
    expected += static_cast<double>(
                    logits[row * 16 + static_cast<size_t>(targets[i])]) -
                std::log(sum_exp);
  }
  ContinuationScore score =
      score_continuation(model, config, Device::CPU, {1, 2}, targets);
  EXPECT_NEAR(score.logprob, expected, 1e-5);
}

TEST(ModelFile, GreedyOnlyForArgmaxContinuation) {
  GPTConfig config = TinyConfig();
  Model model(kVocab, config);
  std::vector<int32_t> sequence = {1, 2};
  std::vector<int32_t> continuation;
  for (size_t step = 0; step < 3; step++) {
    const std::vector<scalar_t> logits = Logits(model, sequence);
    const size_t row = sequence.size() - 1;
    int32_t best = 0;
    for (size_t v = 1; v < kVocab; v++) {
      if (logits[row * kVocab + v] >
          logits[row * kVocab + static_cast<size_t>(best)]) {
        best = static_cast<int32_t>(v);
      }
    }
    sequence.push_back(best);
    continuation.push_back(best);
  }

  EXPECT_TRUE(
      score_continuation(model, config, Device::CPU, {1, 2}, continuation)
          .greedy);
  continuation.back() =
      (continuation.back() + 1) % static_cast<int32_t>(kVocab);
  EXPECT_FALSE(
      score_continuation(model, config, Device::CPU, {1, 2}, continuation)
          .greedy);
}

TEST(ModelFile, RequestLineMatchesScore) {
  GPTConfig config = TinyConfig();
  Model model(kVocab, config);
  ContinuationScore score =
      score_continuation(model, config, Device::CPU, {1, 2}, {3, 4});
  const std::string response =
      score_request(model, config, Device::CPU,
                    R"(  { "continuation" : [ 3 ,4 ] , "context":[1,2] }  )");
  const std::string prefix = R"({"logprob": )";
  ASSERT_TRUE(response.starts_with(prefix)) << response;
  EXPECT_EQ(std::stod(response.substr(prefix.size())), score.logprob);
  EXPECT_TRUE(response.ends_with(std::string(R"(, "greedy": )") +
                                 (score.greedy ? "true" : "false") +
                                 R"(, "scored": 2})"))
      << response;
}

TEST(ModelFile, MalformedRequestsGetErrorLines) {
  GPTConfig config = TinyConfig();
  Model model(kVocab, config);
  const std::vector<std::string> lines = {
      "",
      "{",
      "not json",
      "[]",
      "{}",
      R"({"context": [1, 2]})",
      R"({"context": [1], "continuation": [-1]})",
      R"({"context": [1], "continuation": [1.5]})",
      R"({"context": [1], "continuation": [99]})",
      R"({"context": [1], "continuation": [99999999999]})",
      R"({"context": [1], "continuation": [1]} x)",
      R"({"context": [1], "context": [2], "continuation": [1]})",
      R"({"context": [1, ], "continuation": [2]})",
      R"({"context": [1], "continuation": []})",
      R"({"context": [], "continuation": [1]})",
      R"({"context": [1], "continuation": [1, 2, 3, 4, 5, 6, 7, 8, 9]})",
      R"({"id": 3, "context": [1], "continuation": [2]})",
      R"({"context": "ab", "continuation": [2]})",
  };
  for (const std::string &line : lines) {
    EXPECT_TRUE(IsError(score_request(model, config, Device::CPU, line)))
        << line << " -> " << score_request(model, config, Device::CPU, line);
  }
}
