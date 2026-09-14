#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <random>

#include "examples/gpt/BPE.h"
#include "examples/gpt/KVCache.h"
#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"

namespace micrograd::gpt {

struct ChatConfig {
  size_t max_new_tokens = 256;
  scalar_t temperature = 1.0f;
  size_t top_k = 0;
};

using NextTokenLogits = std::function<std::shared_ptr<Tensor>(
    int32_t token, size_t position, KVCache &cache)>;

int32_t sample_token(const std::shared_ptr<Tensor> &logits,
                     scalar_t temperature, size_t top_k, std::mt19937_64 &rng);

void run_chat(const BPE &bpe, const ChatConfig &config,
              const NextTokenLogits &next_logits, KVCache &cache,
              std::mt19937_64 &rng, std::istream &in, std::ostream &out);

}  // namespace micrograd::gpt
