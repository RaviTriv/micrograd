#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gpt/Dataset.h"
#include "gpt/GPTConfig.h"
#include "gpt/Model.h"
#include "gpt/SampleOrder.h"
#include "gpt/TrainConfig.h"
#include "gpt/TrainOptimizer.h"
#include "gpt/TrainState.h"
#include "micrograd/Random.h"
#include "micrograd/Tensor.h"
#include "micrograd/nn/Loss.h"
#include "micrograd/nn/Serialize.h"

using micrograd::scalar_t;
using micrograd::Tensor;
using micrograd::gpt::Dataset;
using micrograd::gpt::GPTConfig;
using micrograd::gpt::Model;
using micrograd::gpt::SampleOrder;
using micrograd::gpt::TrainConfig;
using micrograd::gpt::TrainOptimizer;

namespace {

constexpr size_t kBlock = 8;
constexpr size_t kBatch = 2;
constexpr uint64_t kSeed = 11;
constexpr size_t kFirstSteps = 4;
constexpr size_t kTotalSteps = 7;

struct ResumeCase {
  std::string name;
  scalar_t muon_lr;
  scalar_t dropout;
};

struct RunResult {
  std::vector<scalar_t> losses;
  std::vector<std::vector<scalar_t>> parameters;
};

TrainConfig schedule() {
  TrainConfig config;
  config.learning_rate = 3e-3f;
  config.min_lr = 3e-4f;
  config.warmup_iters = 2;
  config.max_iters = kTotalSteps;
  return config;
}

class Trainer {
 public:
  Trainer(const Dataset &dataset, const ResumeCase &resume_case)
      : dataset_(dataset),
        model_(make_model(dataset, resume_case)),
        optimizer_(model_->named_parameters(), 0.1f, 0.9f, 0.95f, 1e-8f,
                   schedule().learning_rate, resume_case.muon_lr),
        order_(dataset.train_span_tokens(), kBlock, kSeed) {}

  void train(size_t from, size_t to, RunResult &result) {
    for (size_t it = from; it < to; it++) {
      model_->train();
      optimizer_.zero_grad();
      Dataset::Batch batch = dataset_.train_batch(kBatch, order_);
      std::vector<scalar_t> tokens(batch.inputs.begin(), batch.inputs.end());
      auto input = Tensor::from_values({kBatch, kBlock}, std::move(tokens));
      auto loss = micrograd::cross_entropy(
          model_->forward(input), batch.targets, dataset_.vocab_size());
      loss->backward();
      result.losses.push_back(loss->at({}));
      optimizer_.step(micrograd::gpt::LrAt(it, schedule()));
    }
  }

  void save(const std::string &checkpoint) {
    const size_t step = optimizer_.step_count();
    micrograd::save(checkpoint, *model_);
    micrograd::gpt::save_optimizer_state(
        micrograd::gpt::optimizer_state_path(checkpoint), optimizer_);
    micrograd::gpt::save_data_state(micrograd::gpt::data_state_path(checkpoint),
                                    order_, step);
    micrograd::gpt::save_rng_state(micrograd::gpt::rng_state_path(checkpoint),
                                   step);
  }

  size_t load(const std::string &checkpoint) {
    const size_t step = micrograd::gpt::check_train_state(checkpoint);
    micrograd::load(checkpoint, *model_);
    micrograd::gpt::load_optimizer_state(
        micrograd::gpt::optimizer_state_path(checkpoint), optimizer_, step);
    micrograd::gpt::load_data_state(micrograd::gpt::data_state_path(checkpoint),
                                    step * kBatch, order_, step);
    micrograd::gpt::load_rng_state(micrograd::gpt::rng_state_path(checkpoint),
                                   step);
    return step;
  }

  size_t muon_count() const { return optimizer_.muon_count(); }

  std::vector<std::vector<scalar_t>> parameters() const {
    std::vector<std::vector<scalar_t>> values;
    for (const auto &p : model_->parameters()) {
      auto data = p->data();
      values.emplace_back(data.begin(), data.end());
    }
    return values;
  }

 private:
  static std::unique_ptr<Model> make_model(const Dataset &dataset,
                                           const ResumeCase &resume_case) {
    GPTConfig config{};
    config.depth = 2;
    config.n_layer = 2;
    config.n_embd = 16;
    config.n_head = 2;
    config.n_kv_head = 2;
    config.block_size = kBlock;
    config.vocab_size = dataset.vocab_size();
    config.dropout = resume_case.dropout;
    micrograd::manual_seed(kSeed);
    return std::make_unique<Model>(dataset.vocab_size(), config);
  }

  const Dataset &dataset_;
  std::unique_ptr<Model> model_;
  TrainOptimizer optimizer_;
  SampleOrder order_;
};

class ResumeTest : public ::testing::TestWithParam<ResumeCase> {
 protected:
  void SetUp() override {
    const std::string name = "resume_test_" + GetParam().name;
    data_path_ = (std::filesystem::temp_directory_path() / (name + ".txt"));
    checkpoint_ = (std::filesystem::temp_directory_path() / (name + ".bin"));
    std::ofstream file(data_path_);
    for (int i = 0; i < 20; i++) {
      file << "the quick brown fox jumps over the lazy dog. ";
    }
  }

  void TearDown() override {
    const std::string checkpoint = checkpoint_.string();
    std::filesystem::remove(data_path_);
    std::filesystem::remove(checkpoint);
    std::filesystem::remove(micrograd::gpt::optimizer_state_path(checkpoint));
    std::filesystem::remove(micrograd::gpt::data_state_path(checkpoint));
    std::filesystem::remove(micrograd::gpt::rng_state_path(checkpoint));
  }

  std::filesystem::path data_path_;
  std::filesystem::path checkpoint_;
};

TEST_P(ResumeTest, ResumedRunMatchesUninterruptedRun) {
  const Dataset dataset(data_path_.string(), 0.1);

  RunResult straight;
  {
    Trainer trainer(dataset, GetParam());
    EXPECT_EQ(trainer.muon_count() > 0, GetParam().muon_lr > 0.0f);
    trainer.train(0, kTotalSteps, straight);
    straight.parameters = trainer.parameters();
  }

  RunResult resumed;
  {
    Trainer trainer(dataset, GetParam());
    trainer.train(0, kFirstSteps, resumed);
    trainer.save(checkpoint_.string());
  }
  micrograd::manual_seed(kSeed + 1);
  {
    Trainer trainer(dataset, GetParam());
    ASSERT_EQ(trainer.load(checkpoint_.string()), kFirstSteps);
    trainer.train(kFirstSteps, kTotalSteps, resumed);
    resumed.parameters = trainer.parameters();
  }

  ASSERT_EQ(resumed.losses.size(), straight.losses.size());
  for (size_t i = 0; i < straight.losses.size(); i++) {
    EXPECT_EQ(resumed.losses[i], straight.losses[i]) << "step " << i;
  }
  ASSERT_EQ(resumed.parameters.size(), straight.parameters.size());
  for (size_t i = 0; i < straight.parameters.size(); i++) {
    EXPECT_EQ(resumed.parameters[i], straight.parameters[i]) << "tensor " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    Optimizers, ResumeTest,
    ::testing::Values(ResumeCase{"AdamW", 0.0f, 0.0f},
                      ResumeCase{"Muon", 0.02f, 0.0f},
                      ResumeCase{"AdamWDropout", 0.0f, 0.2f},
                      ResumeCase{"MuonDropout", 0.02f, 0.2f}),
    [](const auto &test_info) { return test_info.param.name; });

}  // namespace
