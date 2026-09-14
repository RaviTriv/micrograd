#pragma once

#include <cstddef>

#include "micrograd/Scalar.h"

namespace micrograd::gpt {

struct GPTConfig {
  size_t depth;
  size_t n_layer;
  size_t n_embd;
  size_t n_head;
  size_t n_kv_head;
  scalar_t matrix_lr;
  scalar_t embedding_lr;
  scalar_t unembedding_lr;
};

GPTConfig config_for_depth(size_t depth);

}  // namespace micrograd::gpt
