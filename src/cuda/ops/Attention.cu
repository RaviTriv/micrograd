#include "micrograd/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>

#include "micrograd/Tensor.h"
#include "micrograd/cuda/ops/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::cuda::ops {
namespace {

constexpr unsigned int kTile = 32;
constexpr size_t kMaxHeadDim = 128;
constexpr scalar_t kNegInf = -1e30f;
constexpr int kReduceBlockSize = 256;
constexpr size_t kMaxReduceBlocks = 65535;

unsigned int TileCount(size_t n) {
  return static_cast<unsigned int>((n + kTile - 1) / kTile);
}

int ReduceGridSize(size_t n) {
  size_t blocks = (n + kReduceBlockSize - 1) / kReduceBlockSize;
  return static_cast<int>(std::min(blocks, kMaxReduceBlocks));
}

__global__ void FlashAttentionForwardKernel(const scalar_t *q,
                                            const scalar_t *k,
                                            const scalar_t *v, scalar_t *out,
                                            size_t seq_len, size_t head_dim,
                                            scalar_t scale) {
  extern __shared__ scalar_t shared[];
  scalar_t *k_tile = shared;
  scalar_t *v_tile = shared + (kTile * head_dim);

  const size_t n = blockIdx.y;
  const size_t row = (blockIdx.x * kTile) + threadIdx.x;
  const bool active = row < seq_len;

  const scalar_t *q_base = q + (n * seq_len * head_dim);
  const scalar_t *k_base = k + (n * seq_len * head_dim);
  const scalar_t *v_base = v + (n * seq_len * head_dim);
  scalar_t *out_base = out + (n * seq_len * head_dim);

  scalar_t q_reg[kMaxHeadDim];
  scalar_t acc[kMaxHeadDim];
  for (size_t d = 0; d < head_dim; d++) {
    q_reg[d] = active ? q_base[(row * head_dim) + d] : 0.0f;
    acc[d] = 0.0f;
  }
  scalar_t m = kNegInf;
  scalar_t l = 0.0f;

  const size_t tile_end = (blockIdx.x * kTile) + kTile;
  const size_t block_last_row = tile_end - 1 < seq_len ? tile_end - 1 : seq_len - 1;
  const size_t num_tiles = (block_last_row / kTile) + 1;

  for (size_t t = 0; t < num_tiles; t++) {
    for (size_t idx = threadIdx.x; idx < kTile * head_dim; idx += blockDim.x) {
      size_t local_row = idx / head_dim;
      size_t d = idx % head_dim;
      size_t col = (t * kTile) + local_row;
      k_tile[idx] = col < seq_len ? k_base[(col * head_dim) + d] : 0.0f;
      v_tile[idx] = col < seq_len ? v_base[(col * head_dim) + d] : 0.0f;
    }
    __syncthreads();

    if (active) {
      for (size_t local_col = 0; local_col < kTile; local_col++) {
        size_t col = (t * kTile) + local_col;
        if (col < seq_len && col <= row) {
          scalar_t s = 0.0f;
          for (size_t d = 0; d < head_dim; d++) {
            s += q_reg[d] * k_tile[(local_col * head_dim) + d];
          }
          s *= scale;

          scalar_t m_new = fmaxf(m, s);
          scalar_t correction = expf(m - m_new);
          scalar_t p = expf(s - m_new);
          l = (l * correction) + p;
          for (size_t d = 0; d < head_dim; d++) {
            acc[d] = (acc[d] * correction) +
                    (p * v_tile[(local_col * head_dim) + d]);
          }
          m = m_new;
        }
      }
    }
    __syncthreads();
  }

  if (active) {
    for (size_t d = 0; d < head_dim; d++) {
      out_base[(row * head_dim) + d] = acc[d] / l;
    }
  }
}

__global__ void FlashAttentionStatsKernel(const scalar_t *q, const scalar_t *k,
                                          scalar_t *row_max, scalar_t *row_sum,
                                          size_t seq_len, size_t head_dim,
                                          scalar_t scale) {
  extern __shared__ scalar_t shared[];
  scalar_t *k_tile = shared;

  const size_t n = blockIdx.y;
  const size_t row = (blockIdx.x * kTile) + threadIdx.x;
  const bool active = row < seq_len;

  const scalar_t *q_base = q + (n * seq_len * head_dim);
  const scalar_t *k_base = k + (n * seq_len * head_dim);

  scalar_t q_reg[kMaxHeadDim];
  for (size_t d = 0; d < head_dim; d++) {
    q_reg[d] = active ? q_base[(row * head_dim) + d] : 0.0f;
  }
  scalar_t m = kNegInf;
  scalar_t l = 0.0f;

  const size_t tile_end = (blockIdx.x * kTile) + kTile;
  const size_t block_last_row = tile_end - 1 < seq_len ? tile_end - 1 : seq_len - 1;
  const size_t num_tiles = (block_last_row / kTile) + 1;

  for (size_t t = 0; t < num_tiles; t++) {
    for (size_t idx = threadIdx.x; idx < kTile * head_dim; idx += blockDim.x) {
      size_t local_row = idx / head_dim;
      size_t d = idx % head_dim;
      size_t col = (t * kTile) + local_row;
      k_tile[idx] = col < seq_len ? k_base[(col * head_dim) + d] : 0.0f;
    }
    __syncthreads();

    if (active) {
      for (size_t local_col = 0; local_col < kTile; local_col++) {
        size_t col = (t * kTile) + local_col;
        if (col < seq_len && col <= row) {
          scalar_t s = 0.0f;
          for (size_t d = 0; d < head_dim; d++) {
            s += q_reg[d] * k_tile[(local_col * head_dim) + d];
          }
          s *= scale;
          scalar_t m_new = fmaxf(m, s);
          l = (l * expf(m - m_new)) + expf(s - m_new);
          m = m_new;
        }
      }
    }
    __syncthreads();
  }

  if (active) {
    row_max[(n * seq_len) + row] = m;
    row_sum[(n * seq_len) + row] = l;
  }
}

__global__ void FlashAttentionDeltaKernel(const scalar_t *out,
                                          const scalar_t *out_grad,
                                          scalar_t *delta, size_t total_rows,
                                          size_t head_dim) {
  for (size_t row = (blockIdx.x * blockDim.x) + threadIdx.x; row < total_rows;
       row += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const scalar_t *o_row = out + (row * head_dim);
    const scalar_t *do_row = out_grad + (row * head_dim);
    scalar_t sum = 0.0f;
    for (size_t d = 0; d < head_dim; d++) {
      sum += o_row[d] * do_row[d];
    }
    delta[row] = sum;
  }
}

__global__ void FlashAttentionBackwardQKernel(
    const scalar_t *q, const scalar_t *k, const scalar_t *v,
    const scalar_t *out_grad, const scalar_t *row_max, const scalar_t *row_sum,
    const scalar_t *delta, scalar_t *q_grad, size_t seq_len, size_t head_dim,
    scalar_t scale) {
  extern __shared__ scalar_t shared[];
  scalar_t *k_tile = shared;
  scalar_t *v_tile = shared + (kTile * head_dim);

  const size_t n = blockIdx.y;
  const size_t row = (blockIdx.x * kTile) + threadIdx.x;
  const bool active = row < seq_len;

  const scalar_t *q_base = q + (n * seq_len * head_dim);
  const scalar_t *k_base = k + (n * seq_len * head_dim);
  const scalar_t *v_base = v + (n * seq_len * head_dim);
  const scalar_t *do_base = out_grad + (n * seq_len * head_dim);
  scalar_t *dq_base = q_grad + (n * seq_len * head_dim);

  scalar_t q_reg[kMaxHeadDim];
  scalar_t do_reg[kMaxHeadDim];
  scalar_t dq_acc[kMaxHeadDim];
  for (size_t d = 0; d < head_dim; d++) {
    q_reg[d] = active ? q_base[(row * head_dim) + d] : 0.0f;
    do_reg[d] = active ? do_base[(row * head_dim) + d] : 0.0f;
    dq_acc[d] = 0.0f;
  }
  scalar_t m = active ? row_max[(n * seq_len) + row] : 0.0f;
  scalar_t z = active ? row_sum[(n * seq_len) + row] : 1.0f;
  scalar_t delta_i = active ? delta[(n * seq_len) + row] : 0.0f;

  const size_t tile_end = (blockIdx.x * kTile) + kTile;
  const size_t block_last_row = tile_end - 1 < seq_len ? tile_end - 1 : seq_len - 1;
  const size_t num_tiles = (block_last_row / kTile) + 1;

  for (size_t t = 0; t < num_tiles; t++) {
    for (size_t idx = threadIdx.x; idx < kTile * head_dim; idx += blockDim.x) {
      size_t local_col = idx / head_dim;
      size_t d = idx % head_dim;
      size_t col = (t * kTile) + local_col;
      k_tile[idx] = col < seq_len ? k_base[(col * head_dim) + d] : 0.0f;
      v_tile[idx] = col < seq_len ? v_base[(col * head_dim) + d] : 0.0f;
    }
    __syncthreads();

    if (active) {
      for (size_t local_col = 0; local_col < kTile; local_col++) {
        size_t col = (t * kTile) + local_col;
        if (col < seq_len && col <= row) {
          scalar_t s = 0.0f;
          for (size_t d = 0; d < head_dim; d++) {
            s += q_reg[d] * k_tile[(local_col * head_dim) + d];
          }
          s *= scale;
          scalar_t p = expf(s - m) / z;

          scalar_t dp = 0.0f;
          for (size_t d = 0; d < head_dim; d++) {
            dp += do_reg[d] * v_tile[(local_col * head_dim) + d];
          }
          scalar_t dl = p * (dp - delta_i);
          for (size_t d = 0; d < head_dim; d++) {
            dq_acc[d] += scale * dl * k_tile[(local_col * head_dim) + d];
          }
        }
      }
    }
    __syncthreads();
  }

  if (active) {
    for (size_t d = 0; d < head_dim; d++) {
      dq_base[(row * head_dim) + d] += dq_acc[d];
    }
  }
}

__global__ void FlashAttentionBackwardKVKernel(
    const scalar_t *q, const scalar_t *k, const scalar_t *v,
    const scalar_t *out_grad, const scalar_t *row_max, const scalar_t *row_sum,
    const scalar_t *delta, scalar_t *k_grad, scalar_t *v_grad, size_t seq_len,
    size_t head_dim, scalar_t scale) {
  extern __shared__ scalar_t shared[];
  scalar_t *q_tile = shared;
  scalar_t *do_tile = shared + (kTile * head_dim);
  scalar_t *stat_tile = shared + (2 * kTile * head_dim);

  const size_t n = blockIdx.y;
  const size_t t = blockIdx.x;
  const size_t col = (t * kTile) + threadIdx.x;
  const bool active = col < seq_len;

  const scalar_t *q_base = q + (n * seq_len * head_dim);
  const scalar_t *k_base = k + (n * seq_len * head_dim);
  const scalar_t *v_base = v + (n * seq_len * head_dim);
  const scalar_t *do_base = out_grad + (n * seq_len * head_dim);
  const scalar_t *m_base = row_max + (n * seq_len);
  const scalar_t *z_base = row_sum + (n * seq_len);
  const scalar_t *d_base = delta + (n * seq_len);
  scalar_t *dk_base = k_grad + (n * seq_len * head_dim);
  scalar_t *dv_base = v_grad + (n * seq_len * head_dim);

  scalar_t k_reg[kMaxHeadDim];
  scalar_t v_reg[kMaxHeadDim];
  scalar_t dk_acc[kMaxHeadDim];
  scalar_t dv_acc[kMaxHeadDim];
  for (size_t d = 0; d < head_dim; d++) {
    k_reg[d] = active ? k_base[(col * head_dim) + d] : 0.0f;
    v_reg[d] = active ? v_base[(col * head_dim) + d] : 0.0f;
    dk_acc[d] = 0.0f;
    dv_acc[d] = 0.0f;
  }

  const size_t num_tiles = (seq_len + kTile - 1) / kTile;
  for (size_t r = t; r < num_tiles; r++) {
    for (size_t idx = threadIdx.x; idx < kTile * head_dim; idx += blockDim.x) {
      size_t local_row = idx / head_dim;
      size_t d = idx % head_dim;
      size_t row = (r * kTile) + local_row;
      q_tile[idx] = row < seq_len ? q_base[(row * head_dim) + d] : 0.0f;
      do_tile[idx] = row < seq_len ? do_base[(row * head_dim) + d] : 0.0f;
    }
    for (size_t idx = threadIdx.x; idx < kTile; idx += blockDim.x) {
      size_t row = (r * kTile) + idx;
      bool row_active = row < seq_len;
      stat_tile[idx] = row_active ? m_base[row] : 0.0f;
      stat_tile[kTile + idx] = row_active ? z_base[row] : 1.0f;
      stat_tile[(2 * kTile) + idx] = row_active ? d_base[row] : 0.0f;
    }
    __syncthreads();

    if (active) {
      for (size_t local_row = 0; local_row < kTile; local_row++) {
        size_t row = (r * kTile) + local_row;
        if (row < seq_len && row >= col) {
          scalar_t s = 0.0f;
          for (size_t d = 0; d < head_dim; d++) {
            s += q_tile[(local_row * head_dim) + d] * k_reg[d];
          }
          s *= scale;
          scalar_t m = stat_tile[local_row];
          scalar_t z = stat_tile[kTile + local_row];
          scalar_t delta_i = stat_tile[(2 * kTile) + local_row];
          scalar_t p = expf(s - m) / z;

          scalar_t dp = 0.0f;
          for (size_t d = 0; d < head_dim; d++) {
            dp += do_tile[(local_row * head_dim) + d] * v_reg[d];
          }
          scalar_t dl = p * (dp - delta_i);
          for (size_t d = 0; d < head_dim; d++) {
            dk_acc[d] += scale * dl * q_tile[(local_row * head_dim) + d];
            dv_acc[d] += p * do_tile[(local_row * head_dim) + d];
          }
        }
      }
    }
    __syncthreads();
  }

  if (active) {
    for (size_t d = 0; d < head_dim; d++) {
      dk_base[(col * head_dim) + d] += dk_acc[d];
      dv_base[(col * head_dim) + d] += dv_acc[d];
    }
  }
}

const scalar_t *DataPtr(const Tensor *t) {
  return static_cast<const scalar_t *>(t->data_storage().device_pointer());
}

scalar_t *DataPtr(Tensor *t) {
  return static_cast<scalar_t *>(t->data_storage().device_pointer());
}

scalar_t *GradPtr(Tensor *t) {
  return static_cast<scalar_t *>(t->grad_storage().device_pointer());
}

void FlashAttention(const OpArgs &args) {
  args.out->to(Backend::CUDA);

  const auto &shape = args.lhs->shape();
  size_t head_dim = shape.back();
  size_t seq_len = shape[shape.size() - 2];
  size_t outer = args.lhs->size() / (seq_len * head_dim);
  if (head_dim > kMaxHeadDim) {
    throw std::runtime_error(
        "flash_attention: head_dim exceeds the CUDA kernel limit");
  }

  const dim3 block(kTile);
  const dim3 grid(TileCount(seq_len), static_cast<unsigned int>(outer));
  const size_t shared_bytes = 2 * kTile * head_dim * sizeof(scalar_t);

  FlashAttentionForwardKernel<<<grid, block, shared_bytes,
                                CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.extra),
      DataPtr(args.out), seq_len, head_dim, args.scalar);
}

std::function<void()> FlashAttentionBackward(const GradArgs &args) {
  return [q = args.lhs, k = args.rhs, v = args.extra, out = args.out,
          scale = args.scalar]() {
    const auto &shape = q->shape();
    size_t head_dim = shape.back();
    size_t seq_len = shape[shape.size() - 2];
    size_t outer = q->size() / (seq_len * head_dim);

    CudaContext &ctx = CudaContext::instance();
    cudaStream_t stream = ctx.stream();

    const size_t stat_bytes = outer * seq_len * sizeof(scalar_t);
    auto *row_max = static_cast<scalar_t *>(ctx.allocate(stat_bytes));
    auto *row_sum = static_cast<scalar_t *>(ctx.allocate(stat_bytes));
    auto *delta = static_cast<scalar_t *>(ctx.allocate(stat_bytes));

    const dim3 block(kTile);
    const dim3 grid(TileCount(seq_len), static_cast<unsigned int>(outer));

    const size_t stats_shared = kTile * head_dim * sizeof(scalar_t);
    FlashAttentionStatsKernel<<<grid, block, stats_shared, stream>>>(
        DataPtr(q.get()), DataPtr(k.get()), row_max, row_sum, seq_len,
        head_dim, scale);

    const size_t total_rows = outer * seq_len;
    FlashAttentionDeltaKernel<<<ReduceGridSize(total_rows), kReduceBlockSize,
                                0, stream>>>(DataPtr(out), GradPtr(out), delta,
                                            total_rows, head_dim);

    const size_t qkv_shared = 2 * kTile * head_dim * sizeof(scalar_t);
    FlashAttentionBackwardQKernel<<<grid, block, qkv_shared, stream>>>(
        DataPtr(q.get()), DataPtr(k.get()), DataPtr(v.get()), GradPtr(out),
        row_max, row_sum, delta, GradPtr(q.get()), seq_len, head_dim, scale);

    const size_t kv_shared =
        ((2 * kTile * head_dim) + (3 * kTile)) * sizeof(scalar_t);
    FlashAttentionBackwardKVKernel<<<grid, block, kv_shared, stream>>>(
        DataPtr(q.get()), DataPtr(k.get()), DataPtr(v.get()), GradPtr(out),
        row_max, row_sum, delta, GradPtr(k.get()), GradPtr(v.get()), seq_len,
        head_dim, scale);

    ctx.deallocate(row_max, stat_bytes);
    ctx.deallocate(row_sum, stat_bytes);
    ctx.deallocate(delta, stat_bytes);
  };
}

}  // namespace

void RegisterAttentionOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kFlashAttention, Device::CUDA, FlashAttention);
  registry.RegisterBackward(OpId::kFlashAttention, Device::CUDA,
                            FlashAttentionBackward);
}

}  // namespace micrograd::cuda::ops

#endif
