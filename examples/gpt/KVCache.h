#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "micrograd/Tensor.h"

namespace micrograd::gpt {

class KVCache {
 public:
  KVCache(size_t n_layer, size_t n_kv_head, size_t head_dim);

  std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> append(
      size_t layer, const std::shared_ptr<Tensor> &keys,
      const std::shared_ptr<Tensor> &values);

  size_t length(size_t layer) const;
  void reset();

 private:
  size_t n_kv_head_;
  size_t head_dim_;
  std::vector<std::shared_ptr<Tensor>> keys_;
  std::vector<std::shared_ptr<Tensor>> values_;
};

}  // namespace micrograd::gpt
