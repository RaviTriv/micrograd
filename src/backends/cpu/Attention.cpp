#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cpu/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::ops::cpu {
namespace {

constexpr scalar_t kNegInf = -std::numeric_limits<scalar_t>::infinity();

void FlashAttentionForward(const OpArgs &args) {
  const std::vector<size_t> &shape = args.lhs->shape();
  size_t head_dim = shape.back();
  size_t seq_len = shape[shape.size() - 2];
  size_t outer = args.lhs->size() / (seq_len * head_dim);
  scalar_t scale = args.scalar;

  std::span<const scalar_t> q = args.lhs->data();
  std::span<const scalar_t> k = args.rhs->data();
  std::span<const scalar_t> v = args.extra->data();
  std::span<scalar_t> out = args.out->data();

  std::vector<scalar_t> acc(head_dim);
  for (size_t n = 0; n < outer; n++) {
    const scalar_t *q_base = q.data() + (n * seq_len * head_dim);
    const scalar_t *k_base = k.data() + (n * seq_len * head_dim);
    const scalar_t *v_base = v.data() + (n * seq_len * head_dim);
    scalar_t *out_base = out.data() + (n * seq_len * head_dim);

    for (size_t i = 0; i < seq_len; i++) {
      const scalar_t *q_row = q_base + (i * head_dim);
      std::ranges::fill(acc, 0.0f);
      scalar_t m = kNegInf;
      scalar_t l = 0.0f;

      for (size_t j = 0; j <= i; j++) {
        const scalar_t *k_row = k_base + (j * head_dim);
        scalar_t s = 0.0f;
        for (size_t d = 0; d < head_dim; d++) {
          s += q_row[d] * k_row[d];
        }
        s *= scale;

        scalar_t m_new = std::max(m, s);
        scalar_t correction = std::exp(m - m_new);
        scalar_t p = std::exp(s - m_new);
        l = (l * correction) + p;

        const scalar_t *v_row = v_base + (j * head_dim);
        for (size_t d = 0; d < head_dim; d++) {
          acc[d] = (acc[d] * correction) + (p * v_row[d]);
        }
        m = m_new;
      }

      scalar_t *out_row = out_base + (i * head_dim);
      for (size_t d = 0; d < head_dim; d++) {
        out_row[d] = acc[d] / l;
      }
    }
  }
}

std::function<void()> FlashAttentionBackward(const GradArgs &args) {
  return [q = args.lhs, k = args.rhs, v = args.extra, out = args.out,
          scale = args.scalar]() {
    const std::vector<size_t> &shape = q->shape();
    size_t head_dim = shape.back();
    size_t seq_len = shape[shape.size() - 2];
    size_t outer = q->size() / (seq_len * head_dim);

    std::span<const scalar_t> q_data = q->data();
    std::span<const scalar_t> k_data = k->data();
    std::span<const scalar_t> v_data = v->data();
    std::span<const scalar_t> out_data = out->data();
    std::span<const scalar_t> out_grad = out->grad();
    std::span<scalar_t> q_grad = q->grad();
    std::span<scalar_t> k_grad = k->grad();
    std::span<scalar_t> v_grad = v->grad();

    std::vector<scalar_t> row_max(seq_len);
    std::vector<scalar_t> row_sum(seq_len);
    std::vector<scalar_t> row_delta(seq_len);

    for (size_t n = 0; n < outer; n++) {
      const scalar_t *q_base = q_data.data() + (n * seq_len * head_dim);
      const scalar_t *k_base = k_data.data() + (n * seq_len * head_dim);
      const scalar_t *v_base = v_data.data() + (n * seq_len * head_dim);
      const scalar_t *o_base = out_data.data() + (n * seq_len * head_dim);
      const scalar_t *do_base = out_grad.data() + (n * seq_len * head_dim);
      scalar_t *dq_base = q_grad.data() + (n * seq_len * head_dim);
      scalar_t *dk_base = k_grad.data() + (n * seq_len * head_dim);
      scalar_t *dv_base = v_grad.data() + (n * seq_len * head_dim);

      for (size_t i = 0; i < seq_len; i++) {
        const scalar_t *q_row = q_base + (i * head_dim);
        scalar_t m = kNegInf;
        scalar_t l = 0.0f;
        for (size_t j = 0; j <= i; j++) {
          const scalar_t *k_row = k_base + (j * head_dim);
          scalar_t s = 0.0f;
          for (size_t d = 0; d < head_dim; d++) {
            s += q_row[d] * k_row[d];
          }
          s *= scale;
          scalar_t m_new = std::max(m, s);
          l = (l * std::exp(m - m_new)) + std::exp(s - m_new);
          m = m_new;
        }
        row_max[i] = m;
        row_sum[i] = l;

        const scalar_t *do_row = do_base + (i * head_dim);
        const scalar_t *o_row = o_base + (i * head_dim);
        scalar_t delta = 0.0f;
        for (size_t d = 0; d < head_dim; d++) {
          delta += do_row[d] * o_row[d];
        }
        row_delta[i] = delta;
      }

      for (size_t i = 0; i < seq_len; i++) {
        const scalar_t *q_row = q_base + (i * head_dim);
        const scalar_t *do_row = do_base + (i * head_dim);
        scalar_t *dq_row = dq_base + (i * head_dim);
        scalar_t m = row_max[i];
        scalar_t l = row_sum[i];
        scalar_t delta = row_delta[i];

        for (size_t j = 0; j <= i; j++) {
          const scalar_t *k_row = k_base + (j * head_dim);
          const scalar_t *v_row = v_base + (j * head_dim);
          scalar_t *dk_row = dk_base + (j * head_dim);
          scalar_t *dv_row = dv_base + (j * head_dim);

          scalar_t s = 0.0f;
          for (size_t d = 0; d < head_dim; d++) {
            s += q_row[d] * k_row[d];
          }
          s *= scale;
          scalar_t p = std::exp(s - m) / l;

          scalar_t dp = 0.0f;
          for (size_t d = 0; d < head_dim; d++) {
            dp += do_row[d] * v_row[d];
          }
          scalar_t dl = p * (dp - delta);

          for (size_t d = 0; d < head_dim; d++) {
            dq_row[d] += scale * dl * k_row[d];
            dk_row[d] += scale * dl * q_row[d];
            dv_row[d] += p * do_row[d];
          }
        }
      }
    }
  };
}

}  // namespace

void RegisterAttentionOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kFlashAttention, Device::CPU, FlashAttentionForward);
  registry.RegisterBackward(OpId::kFlashAttention, Device::CPU,
                            FlashAttentionBackward);
}

}  // namespace micrograd::ops::cpu
