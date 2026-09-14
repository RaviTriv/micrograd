#include "examples/gpt/ConversationDataset.h"

#include <stdexcept>

namespace micrograd::gpt {

ConversationDataset::ConversationDataset(
    const BPE &bpe, const std::vector<std::vector<Message>> &conversations) {
  if (conversations.empty()) {
    throw std::invalid_argument("ConversationDataset: no conversations given");
  }

  for (const std::vector<Message> &messages : conversations) {
    RenderedConversation rendered = render_conversation(bpe, messages);
    tokens_.insert(tokens_.end(), rendered.tokens.begin(),
                   rendered.tokens.end());
    loss_mask_.insert(loss_mask_.end(), rendered.loss_mask.begin(),
                      rendered.loss_mask.end());
  }
}

ConversationDataset::Batch ConversationDataset::sample(
    size_t batch_size, size_t block_size, std::mt19937_64 &rng) const {
  if (tokens_.size() <= block_size) {
    throw std::runtime_error(
        "ConversationDataset: not enough tokens for the requested block "
        "size");
  }
  size_t window_count = tokens_.size() - block_size;

  Batch batch;
  batch.inputs.resize(batch_size * block_size);
  batch.targets.resize(batch_size * block_size);
  batch.loss_mask.resize(batch_size * block_size);

  std::uniform_int_distribution<size_t> pick(0, window_count - 1);
  for (size_t row = 0; row < batch_size; row++) {
    size_t start = pick(rng);
    for (size_t col = 0; col < block_size; col++) {
      size_t index = (row * block_size) + col;
      batch.inputs[index] = static_cast<size_t>(tokens_[start + col]);
      batch.targets[index] = static_cast<size_t>(tokens_[start + col + 1]);
      batch.loss_mask[index] = loss_mask_[start + col + 1];
    }
  }
  return batch;
}

}  // namespace micrograd::gpt
