#include "examples/gpt/GRPO.h"

#include <cmath>
#include <stdexcept>

namespace micrograd::gpt {

namespace {

std::shared_ptr<Tensor> feed(const NextTokenLogits &next_logits, int32_t token,
                             KVCache &cache) {
  return next_logits(token, cache.length(0), cache);
}

bool is_number_char(char c) { return (c >= '0' && c <= '9') || c == '.'; }

std::string extract_final_number(const std::string &text) {
  std::string cleaned;
  cleaned.reserve(text.size());
  for (char c : text) {
    if (c != ',') {
      cleaned.push_back(c);
    }
  }

  int64_t end = static_cast<int64_t>(cleaned.size()) - 1;
  while (end >= 0 && !is_number_char(cleaned[static_cast<size_t>(end)])) {
    end--;
  }
  if (end < 0) {
    return "";
  }

  int64_t start = end;
  while (start > 0 && is_number_char(cleaned[static_cast<size_t>(start - 1)])) {
    start--;
  }
  if (start > 0 && cleaned[static_cast<size_t>(start - 1)] == '-') {
    start--;
  }

  std::string number = cleaned.substr(static_cast<size_t>(start),
                                      static_cast<size_t>(end - start + 1));
  while (!number.empty() && number.back() == '.') {
    number.pop_back();
  }
  return number;
}

}  // namespace

std::vector<Rollout> generate_rollout_group(
    const RolloutConfig &config, const std::vector<int32_t> &prompt_tokens,
    int32_t stop_token, const NextTokenLogits &next_logits, KVCache &cache,
    const RewardFn &reward, std::mt19937_64 &rng) {
  if (config.group_size == 0) {
    throw std::invalid_argument(
        "generate_rollout_group requires a positive group size");
  }
  if (prompt_tokens.empty()) {
    throw std::invalid_argument(
        "generate_rollout_group requires at least one prompt token");
  }

  std::vector<Rollout> rollouts;
  rollouts.reserve(config.group_size);

  for (size_t i = 0; i < config.group_size; i++) {
    cache.reset();

    std::shared_ptr<Tensor> logits;
    for (int32_t token : prompt_tokens) {
      logits = feed(next_logits, token, cache);
    }

    std::vector<int32_t> tokens;
    for (size_t step = 0; step < config.max_new_tokens; step++) {
      int32_t token =
          sample_token(logits, config.temperature, config.top_k, rng);
      if (token == stop_token) {
        break;
      }
      tokens.push_back(token);
      logits = feed(next_logits, token, cache);
    }

    rollouts.push_back({tokens, reward(tokens)});
  }

  return rollouts;
}

std::vector<scalar_t> group_relative_advantages(
    const std::vector<double> &rewards, scalar_t eps) {
  if (rewards.empty()) {
    throw std::invalid_argument("group_relative_advantages: empty group");
  }

  double mean = 0.0;
  for (double reward : rewards) {
    mean += reward;
  }
  mean /= static_cast<double>(rewards.size());

  double variance = 0.0;
  for (double reward : rewards) {
    variance += (reward - mean) * (reward - mean);
  }
  variance /= static_cast<double>(rewards.size());
  double stddev = std::sqrt(variance);

  std::vector<scalar_t> advantages(rewards.size());
  for (size_t i = 0; i < rewards.size(); i++) {
    advantages[i] = static_cast<scalar_t>((rewards[i] - mean) / (stddev + eps));
  }
  return advantages;
}

double exact_match_reward(const std::string &completion,
                          const std::string &gold_answer) {
  std::string predicted = extract_final_number(completion);
  std::string gold = extract_final_number(gold_answer);
  return (!predicted.empty() && predicted == gold) ? 1.0 : 0.0;
}

}  // namespace micrograd::gpt
