#include <cmath>
#include <cstddef>
#include <cstdint>
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

struct RotaryTables {
  size_t seq_len = 0;
  size_t head_dim = 0;
  scalar_t base = 0.0f;
  Device device = Device::CPU;
  std::shared_ptr<Tensor> cos_table;
  std::shared_ptr<Tensor> sin_table;
};

const RotaryTables &CachedRotaryTables(size_t seq_len, size_t head_dim,
                                       scalar_t base, Device device) {
  static thread_local RotaryTables cached;
  if (cached.cos_table == nullptr || cached.seq_len != seq_len ||
      cached.head_dim != head_dim || cached.base != base ||
      cached.device != device) {
    cached = RotaryTables{
        .seq_len = seq_len,
        .head_dim = head_dim,
        .base = base,
        .device = device,
        .cos_table = MakeRotaryTable(seq_len, head_dim, base, true),
        .sin_table = MakeRotaryTable(seq_len, head_dim, base, false)};
    cached.cos_table->to(device);
    cached.sin_table->to(device);
  }
  return cached;
}

}  // namespace

std::shared_ptr<Tensor> Tensor::rotary_embedding(scalar_t base,
                                                 int64_t seq_dim) {
  if (shape_.size() < 2) {
    throw std::invalid_argument(
        "rotary_embedding expects an input of rank 2 or higher");
  }
  const auto rank = static_cast<int64_t>(shape_.size());
  if (seq_dim < 0) {
    seq_dim += rank;
  }
  if (seq_dim < 0 || seq_dim >= rank - 1) {
    throw std::invalid_argument(
        "rotary_embedding sequence dimension must precede the head dimension");
  }

  size_t head_dim = shape_.back();
  if (head_dim == 0 || head_dim % 2 != 0) {
    throw std::invalid_argument(
        "rotary_embedding expects an even, non-zero head dimension");
  }

  size_t seq_len = shape_[static_cast<size_t>(seq_dim)];
  const RotaryTables &tables =
      CachedRotaryTables(seq_len, head_dim, base, backend());
  return apply(OpId::kRotaryEmbedding, shape_,
               {.lhs = shared_from_this(),
                .rhs = tables.cos_table,
                .extra = tables.sin_table,
                .dim = seq_dim,
                .grad_through_lhs_only = true});
}

std::shared_ptr<Tensor> Tensor::head_norm_rotary(
    size_t n_head, const std::shared_ptr<Tensor> &gain, scalar_t eps,
    bool rotate, scalar_t base) {
  if (shape_.size() < 2 || n_head == 0 || shape_.back() % n_head != 0) {
    throw std::invalid_argument(
        "head_norm_rotary expects a (..., seq, n_head * head_dim) input");
  }
  const size_t head_dim = shape_.back() / n_head;
  if (gain && gain->size() != head_dim) {
    throw std::invalid_argument("head_norm_rotary gain must match head_dim");
  }
  if (rotate && head_dim % 2 != 0) {
    throw std::invalid_argument(
        "head_norm_rotary needs an even head dimension to rotate");
  }
  if (!gain && !rotate) {
    return shared_from_this();
  }
  OpInputs inputs{.lhs = shared_from_this(),
                  .rhs = gain,
                  .scalar = eps,
                  .dim = static_cast<int64_t>(n_head)};
  if (rotate) {
    const RotaryTables &tables = CachedRotaryTables(shape_[shape_.size() - 2],
                                                    head_dim, base, backend());
    inputs.extra = tables.cos_table;
    inputs.aux = tables.sin_table;
  }
  return apply(OpId::kHeadNormRotary, shape_, inputs);
}

}  // namespace micrograd
