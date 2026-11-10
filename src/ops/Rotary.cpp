#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {

namespace {

std::shared_ptr<Tensor> MakeRotaryTable(size_t seq_len, size_t head_dim,
                                        scalar_t base, bool cosine) {
  const size_t half = head_dim / 2;
  auto table = Tensor::zeros(std::vector<size_t>{seq_len, half});
  auto values = table->data();
  for (size_t t = 0; t < seq_len; t++) {
    for (size_t j = 0; j < half; j++) {
      scalar_t exponent =
          static_cast<scalar_t>(2 * j) / static_cast<scalar_t>(head_dim);
      scalar_t freq = 1.0f / std::pow(base, exponent);
      scalar_t angle = static_cast<scalar_t>(t) * freq;
      values[(t * half) + j] = cosine ? std::cos(angle) : std::sin(angle);
    }
  }
  return table;
}

}  // namespace

std::shared_ptr<Tensor> Tensor::rotary_embedding(scalar_t base) {
  if (shape_.size() < 2) {
    throw std::invalid_argument(
        "rotary_embedding expects an input of rank 2 or higher");
  }

  size_t head_dim = shape_.back();
  if (head_dim == 0 || head_dim % 2 != 0) {
    throw std::invalid_argument(
        "rotary_embedding expects an even, non-zero head dimension");
  }

  size_t seq_len = shape_[shape_.size() - 2];

  struct Tables {
    size_t seq_len = 0;
    size_t head_dim = 0;
    scalar_t base = 0.0f;
    Device device = Device::CPU;
    std::shared_ptr<Tensor> cos_table;
    std::shared_ptr<Tensor> sin_table;
  };
  static thread_local Tables cached;
  if (cached.cos_table == nullptr || cached.seq_len != seq_len ||
      cached.head_dim != head_dim || cached.base != base ||
      cached.device != backend()) {
    cached = Tables{.seq_len = seq_len,
                    .head_dim = head_dim,
                    .base = base,
                    .device = backend(),
                    .cos_table = MakeRotaryTable(seq_len, head_dim, base, true),
                    .sin_table =
                        MakeRotaryTable(seq_len, head_dim, base, false)};
    cached.cos_table->to(backend());
    cached.sin_table->to(backend());
  }

  return apply(OpId::kRotaryEmbedding, shape_,
               {.lhs = shared_from_this(),
                .rhs = cached.cos_table,
                .extra = cached.sin_table,
                .grad_through_lhs_only = true});
}

}  // namespace micrograd
