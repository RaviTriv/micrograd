#pragma once

#include <cstddef>
#include <memory>

#include "micrograd/NN.h"
#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"
#include "micrograd/nn/Module.h"

namespace micrograd::gpt {

class CausalSelfAttention : public nn::Module {
 public:
  CausalSelfAttention(size_t n_embd, size_t n_head, scalar_t dropout);

  std::shared_ptr<Tensor> forward(
      const std::shared_ptr<Tensor> &input) override;

 private:
  size_t n_embd_;
  size_t n_head_;
  std::shared_ptr<Linear> query_;
  std::shared_ptr<Linear> key_;
  std::shared_ptr<Linear> value_;
  std::shared_ptr<Linear> out_proj_;
  std::shared_ptr<Dropout> attn_dropout_;
  std::shared_ptr<Dropout> resid_dropout_;
};

}  // namespace micrograd::gpt
