#pragma once

#include <cstddef>

#include "micrograd/Scalar.h"

namespace micrograd::gpt {

struct GPTConfig {
  size_t depth;
  size_t n_layer = 6;
  size_t n_embd = 384;
  size_t n_head = 6;
  size_t n_kv_head;
  size_t block_size = 256;
  size_t vocab_size = 65536;
  scalar_t dropout = 0.0f;
  scalar_t matrix_lr;
  scalar_t embedding_lr;
  scalar_t unembedding_lr;
};

GPTConfig config_for_depth(size_t depth);
GPTConfig gpt2_124m();

}  // namespace micrograd::gpt
