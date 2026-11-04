#include <cstddef>
#include <functional>
#include <span>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cpu/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::ops::cpu {
namespace {

template <bool IsBackward>
void RotateHalf(std::span<const scalar_t> input,
                std::span<const scalar_t> cos_table,
                std::span<const scalar_t> sin_table, std::span<scalar_t> output,
                size_t outer, size_t seq_len, size_t half) {
  size_t head_dim = half * 2;
  scalar_t sign = IsBackward ? -1.0f : 1.0f;

  for (size_t o = 0; o < outer; o++) {
    for (size_t t = 0; t < seq_len; t++) {
      const scalar_t *row = input.data() + (((o * seq_len) + t) * head_dim);
      scalar_t *out_row = output.data() + (((o * seq_len) + t) * head_dim);
      const scalar_t *cos_row = cos_table.data() + (t * half);
      const scalar_t *sin_row = sin_table.data() + (t * half);
      for (size_t j = 0; j < half; j++) {
        scalar_t x1 = row[j];
        scalar_t x2 = row[j + half];
        scalar_t y1 = (x1 * cos_row[j]) + (sign * x2 * sin_row[j]);
        scalar_t y2 = (x2 * cos_row[j]) - (sign * x1 * sin_row[j]);
        if constexpr (IsBackward) {
          out_row[j] += y1;
          out_row[j + half] += y2;
        } else {
          out_row[j] = y1;
          out_row[j + half] = y2;
        }
      }
    }
  }
}

void RotaryEmbedding(const OpArgs &args) {
  size_t head_dim = args.lhs->shape().back();
  size_t seq_len = args.lhs->shape()[args.lhs->shape().size() - 2];
  size_t half = head_dim / 2;
  size_t outer = args.lhs->size() / (seq_len * head_dim);

  RotateHalf<false>(args.lhs->data(), args.rhs->data(), args.extra->data(),
                    args.out->data(), outer, seq_len, half);
}

std::function<void()> RotaryEmbeddingBackward(const GradArgs &args) {
  return [lhs = args.lhs, cos_table = args.rhs, sin_table = args.extra,
          out = args.out]() {
    size_t head_dim = lhs->shape().back();
    size_t seq_len = lhs->shape()[lhs->shape().size() - 2];
    size_t half = head_dim / 2;
    size_t outer = lhs->size() / (seq_len * head_dim);

    RotateHalf<true>(out->grad(), cos_table->data(), sin_table->data(),
                     lhs->grad(), outer, seq_len, half);
  };
}

}  // namespace

void RegisterRotaryOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kRotaryEmbedding, Device::CPU, RotaryEmbedding);
  registry.RegisterBackward(OpId::kRotaryEmbedding, Device::CPU,
                            RotaryEmbeddingBackward);
}

}  // namespace micrograd::ops::cpu
