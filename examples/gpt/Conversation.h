#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "examples/gpt/BPE.h"

namespace micrograd::gpt {

enum class Role { kUser, kAssistant };

struct Message {
  Role role;
  std::string content;
};

struct RenderedConversation {
  std::vector<int32_t> tokens;
  std::vector<bool> loss_mask;
};

RenderedConversation render_conversation(const BPE &bpe,
                                         const std::vector<Message> &messages);

}  // namespace micrograd::gpt
