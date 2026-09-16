#pragma once

#include <cstddef>
#include <memory>

#include "examples/gpt/Attention.h"
#include "micrograd/NN.h"
#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"
#include "micrograd/nn/Module.h"

namespace micrograd::gpt {

class Block : public nn::Module {
 public:
  Block(size_t n_embd, size_t n_head, scalar_t dropout);

  std::shared_ptr<Tensor> forward(
      const std::shared_ptr<Tensor> &input) override;

 private:
  std::shared_ptr<LayerNorm> ln1_;
  std::shared_ptr<CausalSelfAttention> attn_;
  std::shared_ptr<LayerNorm> ln2_;
  std::shared_ptr<Linear> mlp_fc_;
  std::shared_ptr<Linear> mlp_proj_;
  std::shared_ptr<Dropout> mlp_dropout_;
};

}  // namespace micrograd::gpt
