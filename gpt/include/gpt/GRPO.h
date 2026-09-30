#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "gpt/Chat.h"
#include "gpt/KVCache.h"
#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"

namespace micrograd::gpt {

std::shared_ptr<Tensor> grpo_loss(
    const std::shared_ptr<Tensor> &logits,
    const std::vector<size_t> &target_indices,
    const std::vector<scalar_t> &advantages,
    const std::vector<scalar_t> &reference_log_probs, scalar_t kl_coeff);

struct RolloutConfig {
  size_t group_size;
  size_t max_new_tokens;
  scalar_t temperature = 1.0f;
  size_t top_k = 0;
};

struct Rollout {
  std::vector<int32_t> tokens;
  double reward;
};

using RewardFn = std::function<double(const std::vector<int32_t> &)>;

std::vector<Rollout> generate_rollout_group(
    const RolloutConfig &config, const std::vector<int32_t> &prompt_tokens,
    int32_t stop_token, const NextTokenLogits &next_logits, KVCache &cache,
    const RewardFn &reward, std::mt19937_64 &rng);

std::vector<scalar_t> group_relative_advantages(
    const std::vector<double> &rewards, scalar_t eps = 1e-4f);

double exact_match_reward(const std::string &completion,
                          const std::string &gold_answer);

}  // namespace micrograd::gpt
