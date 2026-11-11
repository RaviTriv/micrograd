#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpt/TrainConfig.h"
#include "gpt/TrainOptimizer.h"
#include "gpt/TrainState.h"
#include "micrograd/Tensor.h"

using micrograd::scalar_t;
using micrograd::Tensor;
using micrograd::gpt::check_finite_loss;
using micrograd::gpt::format_flags;
using micrograd::gpt::LrAt;
using micrograd::gpt::NamedParameters;
using micrograd::gpt::TrainConfig;
using micrograd::gpt::TrainOptimizer;

namespace {

TrainConfig parse(std::vector<std::string> args) {
  args.insert(args.begin(), "train_gpt");
  std::vector<char *> argv;
  argv.reserve(args.size());
  for (auto &arg : args) {
    argv.push_back(arg.data());
  }
  return micrograd::gpt::parse_args(static_cast<int>(argv.size()), argv.data());
}

std::vector<std::string> split(const std::string &text) {
  std::istringstream in(text);
  std::vector<std::string> words;
  std::string word;
  while (in >> word) {
    words.push_back(word);
  }
  return words;
}

std::shared_ptr<Tensor> filled(std::vector<size_t> shape, scalar_t start) {
  size_t size = 1;
  for (size_t dim : shape) {
    size *= dim;
  }
  std::vector<scalar_t> values(size);
  for (size_t i = 0; i < size; i++) {
    values[i] = start + (0.1f * static_cast<scalar_t>(i));
  }
  return Tensor::from_values(std::move(shape), std::move(values));
}

NamedParameters make_parameters() {
  return {{"h.0.mlp.c_fc.weight", filled({4, 6}, 0.5f)},
          {"h.0.mlp.c_fc.bias", filled({6}, -0.2f)},
          {"wte.weight", filled({5, 4}, 0.1f)}};
}

void set_grads(const NamedParameters &parameters, scalar_t scale) {
  for (const auto &[name, p] : parameters) {
    auto grad = p->grad();
    for (size_t i = 0; i < grad.size(); i++) {
      grad[i] = scale * std::sin(static_cast<scalar_t>(i + 1));
    }
  }
}

std::vector<scalar_t> values(const Tensor &tensor) {
  auto data = tensor.data();
  return {data.begin(), data.end()};
}

TrainOptimizer make_optimizer(const NamedParameters &parameters,
                              scalar_t muon_lr) {
  return TrainOptimizer(parameters, 0.1f, 0.9f, 0.99f, 1e-8f, 1e-3f, muon_lr);
}

}  // namespace

TEST(TrainerSchedule, WarmupCosineDecay) {
  TrainConfig config;
  config.learning_rate = 1e-3f;
  config.min_lr = 1e-4f;
  config.warmup_iters = 10;
  config.max_iters = 100;

  EXPECT_FLOAT_EQ(LrAt(0, config), 1e-4f);
  EXPECT_FLOAT_EQ(LrAt(9, config), 1e-3f);
  EXPECT_FLOAT_EQ(LrAt(10, config), 1e-3f);
  const scalar_t last_progress = 89.0f / 90.0f;
  const scalar_t last_coeff =
      0.5f * (1.0f + std::cos(3.14159265f * last_progress));
  EXPECT_FLOAT_EQ(LrAt(99, config), 1e-4f + (last_coeff * 9e-4f));
  EXPECT_FLOAT_EQ(LrAt(100, config), 1e-4f);
}

TEST(TrainerOptimizer, StateRoundTripGivesIdenticalNextStep) {
  const std::string path =
      (std::filesystem::temp_directory_path() / "trainer_test_state.opt")
          .string();
  NamedParameters original = make_parameters();
  TrainOptimizer first = make_optimizer(original, 0.02f);
  for (int step = 0; step < 3; step++) {
    first.zero_grad();
    set_grads(original, 1.0f + static_cast<scalar_t>(step));
    first.step(1e-3f);
  }
  micrograd::gpt::save_optimizer_state(path, first);

  NamedParameters copy = make_parameters();
  for (size_t i = 0; i < copy.size(); i++) {
    auto source = original[i].second->data();
    auto target = copy[i].second->data();
    std::copy(source.begin(), source.end(), target.begin());
  }
  TrainOptimizer second = make_optimizer(copy, 0.02f);
  micrograd::gpt::load_optimizer_state(path, second);
  std::filesystem::remove(path);
  EXPECT_EQ(second.step_count(), 3u);

  first.zero_grad();
  second.zero_grad();
  set_grads(original, -2.0f);
  set_grads(copy, -2.0f);
  first.step(5e-4f);
  second.step(5e-4f);
  for (size_t i = 0; i < copy.size(); i++) {
    EXPECT_EQ(values(*original[i].second), values(*copy[i].second))
        << original[i].first;
  }
}

TEST(TrainerOptimizer, MuonParameterChangesAfterCpuStep) {
  NamedParameters parameters = make_parameters();
  TrainOptimizer optimizer = make_optimizer(parameters, 0.02f);
  ASSERT_EQ(optimizer.muon_count(), 1u);
  const std::vector<scalar_t> before = values(*parameters[0].second);
  set_grads(parameters, 1.0f);
  optimizer.step(1e-3f);
  EXPECT_NE(values(*parameters[0].second), before);
}

TEST(TrainerOptimizer, DecayedCountExcludesMuonMatrices) {
  NamedParameters parameters = make_parameters();
  EXPECT_EQ(make_optimizer(parameters, 0.0f).decayed_count(), 2u);
  EXPECT_EQ(make_optimizer(parameters, 0.02f).decayed_count(), 1u);
}

TEST(TrainerArgs, RejectsInvalidValues) {
  const std::vector<std::vector<std::string>> invalid = {
      {"--batch-size", "0"},
      {"--batch-size", "-1"},
      {"--batch-size", "4x"},
      {"--grad-accum-steps", "0"},
      {"--grad-accum-steps", "-2"},
      {"--block-size", "0"},
      {"--n-layer", "0"},
      {"--n-head", "0"},
      {"--n-embd", "0"},
      {"--n-embd", "30", "--n-head", "4"},
      {"--eval-iters", "0"},
      {"--max-iters", "-5"},
      {"--muon-lr", "0.02", "--lr", "0"},
  };
  for (const auto &args : invalid) {
    EXPECT_THROW(parse(args), std::invalid_argument)
        << args[0] << " " << args[1];
  }
}

TEST(TrainerArgs, AcceptsValidValues) {
  TrainConfig config = parse({"--batch-size", "4", "--grad-accum-steps", "2",
                              "--n-embd", "32", "--n-head", "2"});
  EXPECT_EQ(config.batch_size, 4u);
  EXPECT_EQ(config.grad_accum_steps, 2u);
}

TEST(TrainerArgs, FormattedFlagsParseBackToSameConfig) {
  TrainConfig config =
      parse({"--batch-size", "8", "--lr", "0.003", "--muon-lr", "0.02",
             "--rotary", "--no-recompute", "--device", "cpu", "--seed", "7"});
  const std::string flags = format_flags(config);
  EXPECT_EQ(format_flags(parse(split(flags))), flags);
}

TEST(TrainerLoss, NonFiniteLossThrowsWithStep) {
  EXPECT_NO_THROW(check_finite_loss(2.5, 3));
  try {
    check_finite_loss(std::numeric_limits<double>::quiet_NaN(), 42);
    FAIL() << "expected a throw";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("step 42"), std::string::npos);
  }
  EXPECT_THROW(check_finite_loss(std::numeric_limits<double>::infinity(), 1),
               std::runtime_error);
}
