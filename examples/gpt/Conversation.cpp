#include "examples/gpt/Conversation.h"

namespace micrograd::gpt {

namespace {

void push(RenderedConversation &rendered, int32_t token, bool loss) {
  rendered.tokens.push_back(token);
  rendered.loss_mask.push_back(loss);
}

void push_all(RenderedConversation &rendered,
              const std::vector<int32_t> &tokens, bool loss) {
  for (int32_t token : tokens) {
    push(rendered, token, loss);
  }
}

}  // namespace

RenderedConversation render_conversation(const BPE &bpe,
                                         const std::vector<Message> &messages) {
  RenderedConversation rendered;
  push(rendered, bpe.special_token_id("<|bos|>"), false);

  for (const Message &message : messages) {
    switch (message.role) {
      case Role::kUser:
        push(rendered, bpe.special_token_id("<|user_start|>"), false);
        push_all(rendered, bpe.encode(message.content), false);
        push(rendered, bpe.special_token_id("<|user_end|>"), false);
        break;
      case Role::kAssistant:
        push(rendered, bpe.special_token_id("<|assistant_start|>"), false);
        push_all(rendered, bpe.encode(message.content), true);
        push(rendered, bpe.special_token_id("<|assistant_end|>"), true);
        break;
    }
  }

  return rendered;
}

}  // namespace micrograd::gpt
