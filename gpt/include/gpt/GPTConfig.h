#pragma once

#include <cstddef>

#include "micrograd/DType.h"
#include "micrograd/Scalar.h"
#include "micrograd/nn/Serialize.h"

namespace micrograd::gpt {

struct ArchOptions {
  bool rotary = false;
  bool qk_norm = false;
  bool relu_squared = false;
};

struct GPTConfig {
  size_t depth;
  size_t n_layer = 6;
  size_t n_embd = 384;
  size_t n_head = 6;
  size_t n_kv_head;
  size_t block_size = 256;
  size_t vocab_size = 65536;
  size_t padded_vocab_size = 0;
  scalar_t dropout = 0.0f;
  scalar_t matrix_lr;
  scalar_t embedding_lr;
  scalar_t unembedding_lr;
  bool recompute_activations = true;
  DType activation_dtype = DType::kFloat32;
  ArchOptions arch;
};

GPTConfig config_for_depth(size_t depth);
GPTConfig gpt2_124m();

StateDictMetadata architecture_metadata(const GPTConfig &config);
GPTConfig config_from_metadata(const StateDictMetadata &metadata);
void require_same_architecture(const GPTConfig &checkpoint,
                               const GPTConfig &flags);

}  // namespace micrograd::gpt
