#include "gpt/Block.h"

#include <utility>

namespace micrograd::gpt {

namespace {

constexpr size_t kMlpRatio = 4;

}  // namespace

Block::Block(size_t n_embd, size_t n_head, scalar_t dropout) {
  ln1_ = std::make_shared<LayerNorm>(std::vector<size_t>{n_embd});
  attn_ = std::make_shared<CausalSelfAttention>(n_embd, n_head, dropout);
  ln2_ = std::make_shared<LayerNorm>(std::vector<size_t>{n_embd});
  mlp_fc_ = std::make_shared<Linear>(n_embd, n_embd * kMlpRatio);
  mlp_proj_ = std::make_shared<Linear>(n_embd * kMlpRatio, n_embd);
  mlp_dropout_ = std::make_shared<Dropout>(dropout);

  register_module("ln1", ln1_);
  register_module("attn", attn_);
  register_module("ln2", ln2_);
  register_module("mlp_fc", mlp_fc_);
  register_module("mlp_proj", mlp_proj_);
  register_module("mlp_dropout", mlp_dropout_);

  set_checkpointing();
}

std::shared_ptr<Tensor> Block::forward(const std::shared_ptr<Tensor> &input) {
  return maybe_checkpoint(input, [this](const std::shared_ptr<Tensor> &x) {
    std::shared_ptr<Tensor> residual = x->add(attn_->forward(ln1_->forward(x)));

    std::shared_ptr<Tensor> mlp_out = mlp_dropout_->forward(
        mlp_proj_->forward(gelu(mlp_fc_->forward(ln2_->forward(residual)))));

    return residual->add(mlp_out);
  });
}

}  // namespace micrograd::gpt
