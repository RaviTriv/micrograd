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

scalar_t Dot(const scalar_t *a, const scalar_t *b, size_t n) {
  scalar_t sum = 0.0f;
  for (size_t i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

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
        scalar_t s = Dot(q_row, k_row, head_dim) * scale;

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
          scalar_t s = Dot(q_row, k_row, head_dim) * scale;
          scalar_t m_new = std::max(m, s);
          l = (l * std::exp(m - m_new)) + std::exp(s - m_new);
          m = m_new;
        }
        row_max[i] = m;
        row_sum[i] = l;

        const scalar_t *do_row = do_base + (i * head_dim);
        const scalar_t *o_row = o_base + (i * head_dim);
        row_delta[i] = Dot(do_row, o_row, head_dim);
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

          scalar_t s = Dot(q_row, k_row, head_dim) * scale;
          scalar_t p = std::exp(s - m) / l;

          scalar_t dp = Dot(do_row, v_row, head_dim);
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

struct HeadLayout {
  size_t batch;
  size_t heads;
  size_t seq;
  size_t keys;
  size_t embed;
  size_t head_dim;
};

void CausalAttentionProbs(const OpArgs &args) {
  const std::vector<size_t> &shape = args.lhs->shape();
  const size_t batch = shape[0];
  const size_t seq = shape[1];
  const size_t embed = shape[2];
  const auto heads = static_cast<size_t>(args.dim);
  const size_t head_dim = embed / heads;
  const scalar_t scale = args.scalar;

  std::span<const scalar_t> q = args.lhs->data();
  std::span<const scalar_t> k = args.rhs->data();
  std::span<scalar_t> out = args.out->data();

  for (size_t b = 0; b < batch; b++) {
    for (size_t h = 0; h < heads; h++) {
      for (size_t t = 0; t < seq; t++) {
        scalar_t *row = out.data() + ((((b * heads) + h) * seq) + t) * seq;
        const scalar_t *q_row =
            q.data() + (((b * seq) + t) * embed) + (h * head_dim);
        scalar_t row_max = kNegInf;
        for (size_t s = 0; s <= t; s++) {
          const scalar_t *k_row =
              k.data() + (((b * seq) + s) * embed) + (h * head_dim);
          row[s] = Dot(q_row, k_row, head_dim) * scale;
          row_max = std::max(row_max, row[s]);
        }
        scalar_t total = 0.0f;
        for (size_t s = 0; s <= t; s++) {
          row[s] = std::exp(row[s] - row_max);
          total += row[s];
        }
        for (size_t s = 0; s <= t; s++) {
          row[s] /= total;
        }
        for (size_t s = t + 1; s < seq; s++) {
          row[s] = 0.0f;
        }
      }
    }
  }
}

std::function<void()> CausalAttentionProbsBackward(const GradArgs &args) {
  return [q = args.lhs, k = args.rhs, out = args.out, scale = args.scalar]() {
    const std::vector<size_t> &shape = q->shape();
    const size_t batch = shape[0];
    const size_t seq = shape[1];
    const size_t embed = shape[2];
    const size_t heads = out->shape()[1];
    const size_t head_dim = embed / heads;

    std::span<const scalar_t> q_data = q->data();
    std::span<const scalar_t> k_data = k->data();
    std::span<const scalar_t> probs = out->data();
    std::span<const scalar_t> probs_grad = out->grad();
    std::span<scalar_t> q_grad = q->grad();
    std::span<scalar_t> k_grad = k->grad();

    for (size_t b = 0; b < batch; b++) {
      for (size_t h = 0; h < heads; h++) {
        for (size_t t = 0; t < seq; t++) {
          const size_t row_offset = ((((b * heads) + h) * seq) + t) * seq;
          const scalar_t *p_row = probs.data() + row_offset;
          const scalar_t *dp_row = probs_grad.data() + row_offset;
          const scalar_t dot = Dot(p_row, dp_row, t + 1);
          const size_t q_offset = (((b * seq) + t) * embed) + (h * head_dim);
          for (size_t s = 0; s <= t; s++) {
            const scalar_t ds = scale * p_row[s] * (dp_row[s] - dot);
            const size_t k_offset = (((b * seq) + s) * embed) + (h * head_dim);
            for (size_t d = 0; d < head_dim; d++) {
              q_grad[q_offset + d] += ds * k_data[k_offset + d];
              k_grad[k_offset + d] += ds * q_data[q_offset + d];
            }
          }
        }
      }
    }
  };
}

HeadLayout CombineLayout(const Tensor &probs, const Tensor &value) {
  const std::vector<size_t> &shape = probs.shape();
  const size_t embed = value.shape()[2];
  return {.batch = shape[0],
          .heads = shape[1],
          .seq = shape[2],
          .keys = shape[3],
          .embed = embed,
          .head_dim = embed / shape[1]};
}

void AttentionCombine(const OpArgs &args) {
  const HeadLayout l = CombineLayout(*args.lhs, *args.rhs);
  std::span<const scalar_t> probs = args.lhs->data();
  std::span<const scalar_t> value = args.rhs->data();
  std::span<scalar_t> out = args.out->data();

  for (size_t b = 0; b < l.batch; b++) {
    for (size_t h = 0; h < l.heads; h++) {
      for (size_t t = 0; t < l.seq; t++) {
        const scalar_t *p_row =
            probs.data() + ((((b * l.heads) + h) * l.seq) + t) * l.keys;
        scalar_t *out_row =
            out.data() + (((b * l.seq) + t) * l.embed) + (h * l.head_dim);
        for (size_t d = 0; d < l.head_dim; d++) {
          out_row[d] = 0.0f;
        }
        for (size_t s = 0; s < l.keys; s++) {
          const scalar_t *v_row =
              value.data() + (((b * l.keys) + s) * l.embed) + (h * l.head_dim);
          for (size_t d = 0; d < l.head_dim; d++) {
            out_row[d] += p_row[s] * v_row[d];
          }
        }
      }
    }
  }
}

std::function<void()> AttentionCombineBackward(const GradArgs &args) {
  return [probs = args.lhs, value = args.rhs, out = args.out]() {
    const HeadLayout l = CombineLayout(*probs, *value);
    std::span<const scalar_t> p_data = probs->data();
    std::span<const scalar_t> v_data = value->data();
    std::span<const scalar_t> out_grad = out->grad();
    std::span<scalar_t> p_grad = probs->grad();
    std::span<scalar_t> v_grad = value->grad();

    for (size_t b = 0; b < l.batch; b++) {
      for (size_t h = 0; h < l.heads; h++) {
        for (size_t t = 0; t < l.seq; t++) {
          const size_t p_offset = ((((b * l.heads) + h) * l.seq) + t) * l.keys;
          const scalar_t *do_row = out_grad.data() +
                                   (((b * l.seq) + t) * l.embed) +
                                   (h * l.head_dim);
          for (size_t s = 0; s < l.keys; s++) {
            const size_t v_offset =
                (((b * l.keys) + s) * l.embed) + (h * l.head_dim);
            scalar_t dot = 0.0f;
            for (size_t d = 0; d < l.head_dim; d++) {
              dot += do_row[d] * v_data[v_offset + d];
              v_grad[v_offset + d] += p_data[p_offset + s] * do_row[d];
            }
            p_grad[p_offset + s] += dot;
          }
        }
      }
    }
  };
}

struct SelfAttentionDims {
  size_t batch;
  size_t heads;
  size_t seq;
  size_t embed;
  size_t head_dim;
};

SelfAttentionDims MakeSelfAttentionDims(const Tensor &q, size_t heads) {
  const std::vector<size_t> &shape = q.shape();
  return {.batch = shape[0],
          .heads = heads,
          .seq = shape[1],
          .embed = shape[2],
          .head_dim = shape[2] / heads};
}

void CausalScores(const SelfAttentionDims &d, const scalar_t *q,
                  const scalar_t *k, size_t b, size_t h, size_t t,
                  scalar_t scale, std::vector<scalar_t> &scores) {
  const scalar_t *q_row = q + (((b * d.seq) + t) * d.embed) + (h * d.head_dim);
  for (size_t s = 0; s <= t; s++) {
    const scalar_t *k_row =
        k + (((b * d.seq) + s) * d.embed) + (h * d.head_dim);
    scores[s] = Dot(q_row, k_row, d.head_dim) * scale;
  }
}

void CausalSelfAttention(const OpArgs &args) {
  const SelfAttentionDims d =
      MakeSelfAttentionDims(*args.lhs, static_cast<size_t>(args.dim));
  const scalar_t *q = args.lhs->data().data();
  const scalar_t *k = args.rhs->data().data();
  const scalar_t *v = args.extra->data().data();
  scalar_t *out = args.out->data().data();
  scalar_t *lse = args.aux->data().data();

  std::vector<scalar_t> scores(d.seq);
  for (size_t b = 0; b < d.batch; b++) {
    for (size_t h = 0; h < d.heads; h++) {
      for (size_t t = 0; t < d.seq; t++) {
        CausalScores(d, q, k, b, h, t, args.scalar, scores);
        scalar_t row_max = kNegInf;
        for (size_t s = 0; s <= t; s++) {
          row_max = std::max(row_max, scores[s]);
        }
        scalar_t total = 0.0f;
        for (size_t s = 0; s <= t; s++) {
          scores[s] = std::exp(scores[s] - row_max);
          total += scores[s];
        }
        scalar_t *out_row =
            out + (((b * d.seq) + t) * d.embed) + (h * d.head_dim);
        for (size_t i = 0; i < d.head_dim; i++) {
          out_row[i] = 0.0f;
        }
        for (size_t s = 0; s <= t; s++) {
          const scalar_t p = scores[s] / total;
          const scalar_t *v_row =
              v + (((b * d.seq) + s) * d.embed) + (h * d.head_dim);
          for (size_t i = 0; i < d.head_dim; i++) {
            out_row[i] += p * v_row[i];
          }
        }
        lse[(((b * d.heads) + h) * d.seq) + t] = row_max + std::log(total);
      }
    }
  }
}

std::function<void()> CausalSelfAttentionBackward(const GradArgs &args) {
  return [q = args.lhs, k = args.rhs, v = args.extra, out = args.out,
          aux = args.aux, scale = args.scalar]() {
    const SelfAttentionDims d = MakeSelfAttentionDims(*q, aux->shape()[1]);
    const scalar_t *q_data = q->data().data();
    const scalar_t *k_data = k->data().data();
    const scalar_t *v_data = v->data().data();
    const scalar_t *o_data = out->data().data();
    const scalar_t *do_data = out->grad().data();
    const scalar_t *lse = aux->data().data();
    scalar_t *dq = q->grad().data();
    scalar_t *dk = k->grad().data();
    scalar_t *dv = v->grad().data();

    std::vector<scalar_t> scores(d.seq);
    for (size_t b = 0; b < d.batch; b++) {
      for (size_t h = 0; h < d.heads; h++) {
        for (size_t t = 0; t < d.seq; t++) {
          CausalScores(d, q_data, k_data, b, h, t, scale, scores);
          const size_t row = (((b * d.seq) + t) * d.embed) + (h * d.head_dim);
          const scalar_t row_lse = lse[(((b * d.heads) + h) * d.seq) + t];
          const scalar_t delta = Dot(do_data + row, o_data + row, d.head_dim);
          for (size_t s = 0; s <= t; s++) {
            const size_t key = (((b * d.seq) + s) * d.embed) + (h * d.head_dim);
            const scalar_t p = std::exp(scores[s] - row_lse);
            const scalar_t dp = Dot(do_data + row, v_data + key, d.head_dim);
            const scalar_t ds = p * (dp - delta) * scale;
            for (size_t i = 0; i < d.head_dim; i++) {
              dq[row + i] += ds * k_data[key + i];
              dk[key + i] += ds * q_data[row + i];
              dv[key + i] += p * do_data[row + i];
            }
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
  registry.Register(OpId::kCausalAttentionProbs, Device::CPU,
                    CausalAttentionProbs);
  registry.RegisterBackward(OpId::kCausalAttentionProbs, Device::CPU,
                            CausalAttentionProbsBackward);
  registry.Register(OpId::kAttentionCombine, Device::CPU, AttentionCombine);
  registry.RegisterBackward(OpId::kAttentionCombine, Device::CPU,
                            AttentionCombineBackward);
  registry.Register(OpId::kCausalSelfAttention, Device::CPU,
                    CausalSelfAttention);
  registry.RegisterBackward(OpId::kCausalSelfAttention, Device::CPU,
                            CausalSelfAttentionBackward);
}

}  // namespace micrograd::ops::cpu
