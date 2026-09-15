#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include "examples/gpt/GRPO.h"
#include "examples/gpt/KVCache.h"
#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"

namespace {

using micrograd::Tensor;
using micrograd::gpt::exact_match_reward;
using micrograd::gpt::generate_rollout_group;
using micrograd::gpt::group_relative_advantages;
using micrograd::gpt::KVCache;
using micrograd::gpt::NextTokenLogits;
using micrograd::gpt::Rollout;
using micrograd::gpt::RolloutConfig;

bool Close(double a, double b) { return std::fabs(a - b) < 1e-3; }

NextTokenLogits CyclicPolicy(size_t vocab) {
  return [vocab](int32_t token, size_t, KVCache &) {
    std::vector<micrograd::scalar_t> values(vocab, 0.0f);
    size_t next = static_cast<size_t>(token + 1) % vocab;
    values[next] = 10.0f;
    return std::make_shared<Tensor>(std::vector<size_t>{vocab}, values);
  };
}

size_t CheckRolloutGroupStopsOnStopToken() {
  RolloutConfig config{.group_size = 3, .max_new_tokens = 8, .top_k = 1};
  KVCache cache(1, 1, 1);
  std::mt19937_64 rng(0);
  auto reward = [](const std::vector<int32_t> &tokens) {
    return (tokens == std::vector<int32_t>{1, 2}) ? 1.0 : 0.0;
  };

  std::vector<Rollout> rollouts = generate_rollout_group(
      config, {0}, 3, CyclicPolicy(4), cache, reward, rng);

  size_t failed = 0;
  if (rollouts.size() != config.group_size) {
    std::printf("FAIL  rollout group size %zu, want %zu\n", rollouts.size(),
                config.group_size);
    failed++;
  }
  for (const Rollout &rollout : rollouts) {
    if (rollout.tokens != std::vector<int32_t>{1, 2} || rollout.reward != 1.0) {
      std::printf("FAIL  rollout did not stop at the stop token\n");
      failed++;
    }
  }
  if (failed == 0) {
    std::printf("ok    rollout_group_stops_on_stop_token\n");
  }
  return failed;
}

size_t CheckRolloutGroupRespectsMaxNewTokens() {
  RolloutConfig config{.group_size = 1, .max_new_tokens = 1, .top_k = 1};
  KVCache cache(1, 1, 1);
  std::mt19937_64 rng(0);
  auto reward = [](const std::vector<int32_t> &) { return 0.0; };

  std::vector<Rollout> rollouts = generate_rollout_group(
      config, {0}, 3, CyclicPolicy(4), cache, reward, rng);

  if (rollouts.size() == 1 && rollouts[0].tokens == std::vector<int32_t>{1}) {
    std::printf("ok    rollout_group_respects_max_new_tokens\n");
    return 0;
  }
  std::printf("FAIL  rollout did not stop at max_new_tokens\n");
  return 1;
}

size_t CheckGroupRelativeAdvantagesNormalize() {
  std::vector<micrograd::scalar_t> advantages =
      group_relative_advantages({0.0, 1.0});

  size_t failed = 0;
  if (!Close(advantages[0], -0.9998) || !Close(advantages[1], 0.9998)) {
    std::printf("FAIL  group_relative_advantages [%f, %f]\n",
                static_cast<double>(advantages[0]),
                static_cast<double>(advantages[1]));
    failed++;
  }
  if (failed == 0) {
    std::printf("ok    group_relative_advantages_normalize\n");
  }
  return failed;
}

size_t CheckGroupRelativeAdvantagesZeroVariance() {
  std::vector<micrograd::scalar_t> advantages =
      group_relative_advantages({1.0, 1.0, 1.0});

  for (micrograd::scalar_t advantage : advantages) {
    if (!Close(advantage, 0.0)) {
      std::printf("FAIL  group_relative_advantages zero variance group\n");
      return 1;
    }
  }
  std::printf("ok    group_relative_advantages_zero_variance\n");
  return 0;
}

size_t CheckGroupRelativeAdvantagesThrowsOnEmpty() {
  try {
    group_relative_advantages({});
  } catch (const std::exception &) {
    std::printf("ok    group_relative_advantages_throws_on_empty\n");
    return 0;
  }
  std::printf("FAIL  group_relative_advantages did not throw on empty group\n");
  return 1;
}

size_t CheckExactMatchReward() {
  struct Case {
    std::string completion;
    std::string gold;
    double expected;
  };
  std::vector<Case> cases = {
      {"The answer is 42.", "#### 42", 1.0},
      {"That totals 42,000 dollars", "42000", 1.0},
      {"The change owed is -7.5", "-7.5", 1.0},
      {"The answer is 13", "31", 0.0},
      {"I am not sure", "5", 0.0},
  };

  size_t failed = 0;
  for (const Case &c : cases) {
    double actual = exact_match_reward(c.completion, c.gold);
    if (!Close(actual, c.expected)) {
      std::printf("FAIL  exact_match_reward(%s, %s) = %f, want %f\n",
                  c.completion.c_str(), c.gold.c_str(), actual, c.expected);
      failed++;
    }
  }
  if (failed == 0) {
    std::printf("ok    exact_match_reward\n");
  }
  return failed;
}

}  // namespace

int main() {
  try {
    size_t failed = 0;
    failed += CheckRolloutGroupStopsOnStopToken();
    failed += CheckRolloutGroupRespectsMaxNewTokens();
    failed += CheckGroupRelativeAdvantagesNormalize();
    failed += CheckGroupRelativeAdvantagesZeroVariance();
    failed += CheckGroupRelativeAdvantagesThrowsOnEmpty();
    failed += CheckExactMatchReward();

    std::printf("%zu check(s) failed\n", failed);
    return failed == 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::printf("grpo_check raised: %s\n", error.what());
    return 1;
  }
}
