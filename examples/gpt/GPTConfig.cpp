#include "examples/gpt/GPTConfig.h"

#include <algorithm>
#include <stdexcept>

namespace micrograd::gpt {

namespace {

constexpr size_t kChannelsPerLayer = 64;
constexpr size_t kHeadDim = 128;
constexpr size_t kReferenceEmbd = 768;
constexpr scalar_t kMatrixLr = 0.02f;
constexpr scalar_t kEmbeddingLr = 0.2f;
constexpr scalar_t kUnembeddingLr = 0.004f;

}  // namespace

GPTConfig config_for_depth(size_t depth) {
  if (depth == 0) {
    throw std::invalid_argument("depth must be positive");
  }

  GPTConfig config;
  config.depth = depth;
  config.n_layer = depth;
  config.n_embd = depth * kChannelsPerLayer;
  config.n_head = std::max<size_t>(1, config.n_embd / kHeadDim);
  config.n_kv_head = config.n_head;

  scalar_t width_scale = static_cast<scalar_t>(kReferenceEmbd) /
                         static_cast<scalar_t>(config.n_embd);
  config.matrix_lr = kMatrixLr;
  config.embedding_lr = kEmbeddingLr * width_scale;
  config.unembedding_lr = kUnembeddingLr * width_scale;

  return config;
}

}  // namespace micrograd::gpt
