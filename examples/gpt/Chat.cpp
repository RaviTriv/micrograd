#include "examples/gpt/Chat.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "examples/gpt/Conversation.h"

namespace micrograd::gpt {

int32_t sample_token(const std::shared_ptr<Tensor> &logits,
                     scalar_t temperature, size_t top_k, std::mt19937_64 &rng) {
  if (logits->shape().size() != 1) {
    throw std::invalid_argument("sample_token expects rank 1 logits");
  }
  if (temperature <= 0.0f) {
    throw std::invalid_argument("sample_token requires a positive temperature");
  }

  std::span<const scalar_t> values = logits->data();
  size_t vocab_size = values.size();
  size_t k = (top_k == 0 || top_k > vocab_size) ? vocab_size : top_k;

  std::vector<size_t> indices(vocab_size);
  std::iota(indices.begin(), indices.end(), size_t{0});
  std::partial_sort(
      indices.begin(), indices.begin() + static_cast<int64_t>(k), indices.end(),
      [&values](size_t a, size_t b) { return values[a] > values[b]; });

  scalar_t max_logit = values[indices[0]];
  std::vector<double> weights(k);
  for (size_t i = 0; i < k; i++) {
    weights[i] = std::exp(
        static_cast<double>((values[indices[i]] - max_logit) / temperature));
  }

  std::discrete_distribution<size_t> pick(weights.begin(), weights.end());
  return static_cast<int32_t>(indices[pick(rng)]);
}

namespace {

bool is_prefix(const std::vector<int32_t> &prefix,
               const std::vector<int32_t> &whole) {
  if (whole.size() < prefix.size()) {
    return false;
  }
  return std::equal(prefix.begin(), prefix.end(), whole.begin());
}

std::shared_ptr<Tensor> feed(const NextTokenLogits &next_logits, int32_t token,
                             KVCache &cache) {
  return next_logits(token, cache.length(0), cache);
}

}  // namespace

void run_chat(const BPE &bpe, const ChatConfig &config,
              const NextTokenLogits &next_logits, KVCache &cache,
              std::mt19937_64 &rng, std::istream &in, std::ostream &out) {
  int32_t assistant_start = bpe.special_token_id("<|assistant_start|>");
  int32_t assistant_end = bpe.special_token_id("<|assistant_end|>");

  std::vector<Message> history;
  std::vector<int32_t> committed_tokens;

  std::string line;
  while (out << "> ", std::getline(in, line)) {
    history.push_back({Role::kUser, line});
    RenderedConversation rendered = render_conversation(bpe, history);
    if (!is_prefix(committed_tokens, rendered.tokens)) {
      throw std::runtime_error(
          "run_chat: rendered history diverged from the cached prefix");
    }

    std::shared_ptr<Tensor> logits;
    for (size_t i = committed_tokens.size(); i < rendered.tokens.size(); i++) {
      logits = feed(next_logits, rendered.tokens[i], cache);
    }
    logits = feed(next_logits, assistant_start, cache);

    std::vector<int32_t> reply_tokens;
    for (size_t i = 0; i < config.max_new_tokens; i++) {
      int32_t token =
          sample_token(logits, config.temperature, config.top_k, rng);
      if (token == assistant_end) {
        break;
      }
      reply_tokens.push_back(token);
      logits = feed(next_logits, token, cache);
    }
    feed(next_logits, assistant_end, cache);

    std::string reply = bpe.decode(reply_tokens);
    out << reply << "\n";
    history.push_back({Role::kAssistant, reply});

    committed_tokens = rendered.tokens;
    committed_tokens.push_back(assistant_start);
    committed_tokens.insert(committed_tokens.end(), reply_tokens.begin(),
                            reply_tokens.end());
    committed_tokens.push_back(assistant_end);
  }
}

}  // namespace micrograd::gpt
