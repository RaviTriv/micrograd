#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "examples/gpt/BPE.h"
#include "examples/gpt/Conversation.h"

namespace micrograd::gpt {

class ConversationDataset {
 public:
  struct Batch {
    std::vector<size_t> inputs;
    std::vector<size_t> targets;
    std::vector<bool> loss_mask;
  };

  ConversationDataset(const BPE &bpe,
                      const std::vector<std::vector<Message>> &conversations);

  Batch sample(size_t batch_size, size_t block_size,
               std::mt19937_64 &rng) const;

  size_t token_count() const { return tokens_.size(); }

 private:
  std::vector<int32_t> tokens_;
  std::vector<bool> loss_mask_;
};

}  // namespace micrograd::gpt
