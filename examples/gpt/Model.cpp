#include "examples/gpt/Model.h"

#include <stdexcept>
#include <utility>
#include <vector>

#include "examples/gpt/Block.h"

namespace micrograd::gpt {

namespace {

std::shared_ptr<Tensor> position_ids(size_t seq_len, Backend backend) {
  std::vector<scalar_t> values(seq_len);
  for (size_t i = 0; i < seq_len; i++) {
    values[i] = static_cast<scalar_t>(i);
  }

  auto positions =
      std::make_shared<Tensor>(std::vector<size_t>{seq_len}, std::move(values));
  positions->to(backend);
  return positions;
}

}  // namespace

Model::Model(size_t vocab_size, const GPTConfig &config) : config_(config) {
  token_embedding_ = std::make_shared<Embedding>(vocab_size, config_.n_embd);
  position_embedding_ =
      std::make_shared<Embedding>(config_.block_size, config_.n_embd);

  std::vector<std::shared_ptr<nn::Module>> layers;
  for (size_t i = 0; i < config_.n_layer; i++) {
    layers.push_back(std::make_shared<Block>(config_.n_embd, config_.n_head,
                                             config_.dropout));
  }
  blocks_ = std::make_shared<Sequential>(std::move(layers));

  ln_f_ = std::make_shared<LayerNorm>(std::vector<size_t>{config_.n_embd});
  lm_head_ = std::make_shared<LMHead>(token_embedding_->weight(), true);

  register_module("token_embedding", token_embedding_);
  register_module("position_embedding", position_embedding_);
  register_module("h", blocks_);
  register_module("ln_f", ln_f_);
  register_module("lm_head", lm_head_);
}

std::shared_ptr<Tensor> Model::forward(const std::shared_ptr<Tensor> &input) {
  const std::vector<size_t> &shape = input->shape();
  if (shape.size() != 2) {
    throw std::invalid_argument("Model expects a (batch, seq) input");
  }

  size_t seq_len = shape[1];
  if (seq_len > config_.block_size) {
    throw std::invalid_argument("Model input exceeds block_size");
  }

  std::shared_ptr<Tensor> x = token_embedding_->forward(input)->add(
      position_embedding_->forward(position_ids(seq_len, input->backend())));

  x = blocks_->forward(x);
  x = ln_f_->forward(x);

  return lm_head_->forward(x);
}

}  // namespace micrograd::gpt
