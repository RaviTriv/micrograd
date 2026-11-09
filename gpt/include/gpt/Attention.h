#pragma once

#include <cstddef>
#include <memory>

#include "micrograd/NN.h"
#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"
#include "micrograd/nn/Module.h"

namespace micrograd::gpt {

std::shared_ptr<Tensor> causal_mask(size_t seq_len, Device device);

std::shared_ptr<Tensor> causal_attention(const std::shared_ptr<Tensor> &q,
                                         const std::shared_ptr<Tensor> &k,
                                         const std::shared_ptr<Tensor> &v,
                                         scalar_t scale,
                                         const std::shared_ptr<Tensor> &mask);

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
  std::shared_ptr<Dropout> resid_dropout_;
};

}  // namespace micrograd::gpt
