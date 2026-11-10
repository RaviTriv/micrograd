#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cpu/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::ops::cpu {
namespace {

struct RotaryLayout {
  size_t outer = 1;
  size_t seq_len = 1;
  size_t inner = 1;
  size_t half = 0;
};

RotaryLayout LayoutFor(const std::vector<size_t> &shape, int64_t seq_dim) {
  RotaryLayout layout;
  const auto seq = static_cast<size_t>(seq_dim);
  for (size_t i = 0; i < seq; i++) {
    layout.outer *= shape[i];
  }
  layout.seq_len = shape[seq];
  for (size_t i = seq + 1; i + 1 < shape.size(); i++) {
    layout.inner *= shape[i];
  }
  layout.half = shape.back() / 2;
  return layout;
}

template <bool IsBackward>
void RotateHalf(std::span<const scalar_t> input,
                std::span<const scalar_t> cos_table,
                std::span<const scalar_t> sin_table, std::span<scalar_t> output,
                const RotaryLayout &layout) {
  const size_t half = layout.half;
  const size_t head_dim = half * 2;
  scalar_t sign = IsBackward ? -1.0f : 1.0f;

  for (size_t o = 0; o < layout.outer; o++) {
    for (size_t t = 0; t < layout.seq_len; t++) {
      const scalar_t *cos_row = cos_table.data() + (t * half);
      const scalar_t *sin_row = sin_table.data() + (t * half);
      for (size_t h = 0; h < layout.inner; h++) {
        const size_t offset =
            (((o * layout.seq_len) + t) * layout.inner + h) * head_dim;
        const scalar_t *row = input.data() + offset;
        scalar_t *out_row = output.data() + offset;
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
}

void RotaryEmbedding(const OpArgs &args) {
  RotateHalf<false>(args.lhs->data(), args.rhs->data(), args.extra->data(),
                    args.out->data(), LayoutFor(args.lhs->shape(), args.dim));
}

std::function<void()> RotaryEmbeddingBackward(const GradArgs &args) {
  return [lhs = args.lhs, cos_table = args.rhs, sin_table = args.extra,
          out = args.out, seq_dim = args.dim]() {
    RotateHalf<true>(out->grad(), cos_table->data(), sin_table->data(),
                     lhs->grad(), LayoutFor(lhs->shape(), seq_dim));
  };
}

struct HeadRows {
  size_t rows = 0;
  size_t n_head = 1;
  size_t seq_len = 1;
  size_t head_dim = 0;
};

HeadRows HeadRowsFor(const Tensor &input, int64_t n_head) {
  const std::vector<size_t> &shape = input.shape();
  HeadRows layout;
  layout.n_head = static_cast<size_t>(n_head);
  layout.head_dim = shape.back() / layout.n_head;
  layout.seq_len = shape[shape.size() - 2];
  layout.rows = input.size() / layout.head_dim;
  return layout;
}

scalar_t RowRstd(const scalar_t *row, size_t n, scalar_t eps) {
  scalar_t mean_square = 0.0f;
  for (size_t i = 0; i < n; i++) {
    mean_square += row[i] * row[i];
  }
  mean_square /= static_cast<scalar_t>(n);
  return 1.0f / std::sqrt(mean_square + eps);
}

void HeadNormRotary(const OpArgs &args) {
  const HeadRows layout = HeadRowsFor(*args.lhs, args.dim);
  const size_t d = layout.head_dim;
  const size_t half = d / 2;
  auto input = args.lhs->data();
  auto out = args.out->data();
  std::vector<scalar_t> normed(d);
  for (size_t r = 0; r < layout.rows; r++) {
    const scalar_t *x = input.data() + (r * d);
    scalar_t *y = out.data() + (r * d);
    if (args.rhs != nullptr) {
      auto gain = args.rhs->data();
      const scalar_t rstd = RowRstd(x, d, args.scalar);
      for (size_t i = 0; i < d; i++) {
        normed[i] = x[i] * rstd * gain[i];
      }
    } else {
      std::copy(x, x + d, normed.begin());
    }
    if (args.extra == nullptr) {
      std::copy(normed.begin(), normed.end(), y);
      continue;
    }
    const size_t t = (r / layout.n_head) % layout.seq_len;
    const scalar_t *cos_row = args.extra->data().data() + (t * half);
    const scalar_t *sin_row = args.aux->data().data() + (t * half);
    for (size_t j = 0; j < half; j++) {
      y[j] = (normed[j] * cos_row[j]) + (normed[j + half] * sin_row[j]);
      y[j + half] = (normed[j + half] * cos_row[j]) - (normed[j] * sin_row[j]);
    }
  }
}

std::function<void()> HeadNormRotaryBackward(const GradArgs &args) {
  return [lhs = args.lhs, gain = args.rhs, cos_table = args.extra,
          sin_table = args.aux, out = args.out, eps = args.scalar,
          n_head = args.dim]() {
    const HeadRows layout = HeadRowsFor(*lhs, n_head);
    const size_t d = layout.head_dim;
    const size_t half = d / 2;
    auto input = lhs->data();
    auto incoming = out->grad();
    auto input_grad = lhs->grad();
    std::vector<scalar_t> dn(d);
    for (size_t r = 0; r < layout.rows; r++) {
      const scalar_t *dy = incoming.data() + (r * d);
      if (cos_table) {
        const size_t t = (r / layout.n_head) % layout.seq_len;
        const scalar_t *cos_row = cos_table->data().data() + (t * half);
        const scalar_t *sin_row = sin_table->data().data() + (t * half);
        for (size_t j = 0; j < half; j++) {
          dn[j] = (dy[j] * cos_row[j]) - (dy[j + half] * sin_row[j]);
          dn[j + half] = (dy[j + half] * cos_row[j]) + (dy[j] * sin_row[j]);
        }
      } else {
        std::copy(dy, dy + d, dn.begin());
      }
      scalar_t *dx = input_grad.data() + (r * d);
      if (!gain) {
        for (size_t i = 0; i < d; i++) {
          dx[i] += dn[i];
        }
        continue;
      }
      auto gain_values = gain->data();
      auto gain_grad = gain->grad();
      const scalar_t *x = input.data() + (r * d);
      const scalar_t rstd = RowRstd(x, d, eps);
      scalar_t mean_dot = 0.0f;
      for (size_t i = 0; i < d; i++) {
        mean_dot += dn[i] * gain_values[i] * x[i] * rstd;
        gain_grad[i] += dn[i] * x[i] * rstd;
      }
      mean_dot /= static_cast<scalar_t>(d);
      for (size_t i = 0; i < d; i++) {
        dx[i] += rstd * ((dn[i] * gain_values[i]) - (x[i] * rstd * mean_dot));
      }
    }
  };
}

}  // namespace

void RegisterRotaryOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kRotaryEmbedding, Device::CPU, RotaryEmbedding);
  registry.RegisterBackward(OpId::kRotaryEmbedding, Device::CPU,
                            RotaryEmbeddingBackward);
  registry.Register(OpId::kHeadNormRotary, Device::CPU, HeadNormRotary);
  registry.RegisterBackward(OpId::kHeadNormRotary, Device::CPU,
                            HeadNormRotaryBackward);
}

}  // namespace micrograd::ops::cpu
