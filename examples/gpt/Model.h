#pragma once

#include <cstddef>
#include <memory>

#include "examples/gpt/GPTConfig.h"
#include "micrograd/NN.h"
#include "micrograd/Tensor.h"
#include "micrograd/nn/Module.h"

namespace micrograd::gpt {

class Model : public nn::Module {
 public:
  Model(size_t vocab_size, const GPTConfig &config);

  std::shared_ptr<Tensor> forward(
      const std::shared_ptr<Tensor> &input) override;

 private:
  GPTConfig config_;
  std::shared_ptr<Embedding> token_embedding_;
  std::shared_ptr<Embedding> position_embedding_;
  std::shared_ptr<Sequential> blocks_;
  std::shared_ptr<LayerNorm> ln_f_;
  std::shared_ptr<LMHead> lm_head_;
};

}  // namespace micrograd::gpt
