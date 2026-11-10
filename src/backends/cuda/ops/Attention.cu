#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <mma.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cuda/ops/Gemm.h"
#include "micrograd/backends/cuda/ops/Grad.h"
#include "micrograd/backends/cuda/ops/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::cuda::ops {
namespace {

namespace wmma = nvcuda::wmma;

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
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
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
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();

    const size_t total_rows = outer * seq_len;
    FlashAttentionDeltaKernel<<<ReduceGridSize(total_rows), kReduceBlockSize,
                                0, stream>>>(DataPtr(out), GradPtr(out), delta,
                                            total_rows, head_dim);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();

    const size_t qkv_shared = 2 * kTile * head_dim * sizeof(scalar_t);
    FlashAttentionBackwardQKernel<<<grid, block, qkv_shared, stream>>>(
        DataPtr(q.get()), DataPtr(k.get()), DataPtr(v.get()), GradPtr(out),
        row_max, row_sum, delta, GradPtr(q.get()), seq_len, head_dim, scale);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();

    const size_t kv_shared =
        ((2 * kTile * head_dim) + (3 * kTile)) * sizeof(scalar_t);
    FlashAttentionBackwardKVKernel<<<grid, block, kv_shared, stream>>>(
        DataPtr(q.get()), DataPtr(k.get()), DataPtr(v.get()), GradPtr(out),
        row_max, row_sum, delta, GradPtr(k.get()), GradPtr(v.get()), seq_len,
        head_dim, scale);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();

    ctx.deallocate(row_max, stat_bytes);
    ctx.deallocate(row_sum, stat_bytes);
    ctx.deallocate(delta, stat_bytes);
  };
}

constexpr unsigned int kWarpSize = 32;
constexpr unsigned int kSoftmaxRowsPerBlock = 8;

__device__ scalar_t WarpMax(scalar_t value) {
  for (unsigned int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, offset));
  }
  return value;
}

__device__ scalar_t WarpSum(scalar_t value) {
  for (unsigned int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffU, value, offset);
  }
  return value;
}

__global__ void CausalSoftmaxKernel(scalar_t *scores, scalar_t *lse,
                                    size_t rows, size_t seq) {
  const size_t row = (blockIdx.x * kSoftmaxRowsPerBlock) + threadIdx.y;
  if (row >= rows) {
    return;
  }
  const size_t valid = (row % seq) + 1;
  scalar_t *x = scores + (row * seq);

  scalar_t row_max = kNegInf;
  for (size_t j = threadIdx.x; j < valid; j += kWarpSize) {
    row_max = fmaxf(row_max, x[j]);
  }
  row_max = WarpMax(row_max);
  scalar_t total = 0.0f;
  for (size_t j = threadIdx.x; j < valid; j += kWarpSize) {
    total += __expf(x[j] - row_max);
  }
  total = WarpSum(total);
  const scalar_t inv = 1.0f / total;
  for (size_t j = threadIdx.x; j < seq; j += kWarpSize) {
    x[j] = j < valid ? __expf(x[j] - row_max) * inv : 0.0f;
  }
  if (lse != nullptr && threadIdx.x == 0) {
    lse[row] = row_max + logf(total);
  }
}

__global__ void CausalSoftmaxBackwardKernel(const scalar_t *probs,
                                            scalar_t *grad, size_t rows,
                                            size_t seq, scalar_t scale) {
  const size_t row = (blockIdx.x * kSoftmaxRowsPerBlock) + threadIdx.y;
  if (row >= rows) {
    return;
  }
  const size_t valid = (row % seq) + 1;
  const scalar_t *p = probs + (row * seq);
  scalar_t *g = grad + (row * seq);

  scalar_t dot = 0.0f;
  for (size_t j = threadIdx.x; j < valid; j += kWarpSize) {
    dot += p[j] * g[j];
  }
  dot = WarpSum(dot);
  for (size_t j = threadIdx.x; j < seq; j += kWarpSize) {
    g[j] = j < valid ? scale * p[j] * (g[j] - dot) : 0.0f;
  }
}

unsigned int SoftmaxGrid(size_t rows) {
  return static_cast<unsigned int>((rows + kSoftmaxRowsPerBlock - 1) /
                                   kSoftmaxRowsPerBlock);
}

struct ProbsDims {
  size_t batch;
  size_t heads;
  size_t seq;
  size_t embed;
  size_t head_dim;
};

ProbsDims MakeProbsDims(const Tensor &q, size_t heads) {
  const auto &shape = q.shape();
  return {.batch = shape[0],
          .heads = heads,
          .seq = shape[1],
          .embed = shape[2],
          .head_dim = shape[2] / heads};
}

GemmOperand HeadOperand(const scalar_t *ptr, const ProbsDims &d, bool trans) {
  return {.ptr = ptr,
          .trans = trans,
          .ld = d.embed,
          .inner_stride = d.head_dim,
          .outer_stride = d.seq * d.embed};
}

GemmOutput HeadOutput(scalar_t *ptr, const ProbsDims &d) {
  return {.ptr = ptr,
          .ld = d.embed,
          .inner_stride = d.head_dim,
          .outer_stride = d.seq * d.embed};
}

GemmOperand ScoreOperand(const scalar_t *ptr, const ProbsDims &d, bool trans) {
  return {.ptr = ptr,
          .trans = trans,
          .ld = d.seq,
          .inner_stride = d.seq * d.seq,
          .outer_stride = d.heads * d.seq * d.seq};
}

GemmOutput ScoreOutput(scalar_t *ptr, const ProbsDims &d) {
  return {.ptr = ptr,
          .ld = d.seq,
          .inner_stride = d.seq * d.seq,
          .outer_stride = d.heads * d.seq * d.seq};
}

void CausalAttentionProbs(const OpArgs &args) {
  const ProbsDims d =
      MakeProbsDims(*args.lhs, static_cast<size_t>(args.dim));
  const GemmShape shape{.outer = d.batch,
                        .inner = d.heads,
                        .m = d.seq,
                        .n = d.seq,
                        .k = d.head_dim};
  StridedGemm(shape, args.scalar, HeadOperand(DataPtr(args.lhs), d, false),
              HeadOperand(DataPtr(args.rhs), d, true), 0.0f,
              ScoreOutput(DataPtr(args.out), d));

  const size_t rows = d.batch * d.heads * d.seq;
  const dim3 block(kWarpSize, kSoftmaxRowsPerBlock);
  CausalSoftmaxKernel<<<SoftmaxGrid(rows), block, 0,
                        CudaContext::instance().stream()>>>(
      DataPtr(args.out), nullptr, rows, d.seq);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

std::function<void()> CausalAttentionProbsBackward(const GradArgs &args) {
  return [q = args.lhs, k = args.rhs, out = args.out, scale = args.scalar]() {
    const ProbsDims d = MakeProbsDims(*q, out->shape()[1]);
    const size_t rows = d.batch * d.heads * d.seq;
    scalar_t *grad_scores = GradPtr(out);
    const dim3 block(kWarpSize, kSoftmaxRowsPerBlock);
    CausalSoftmaxBackwardKernel<<<SoftmaxGrid(rows), block, 0,
                                  CudaContext::instance().stream()>>>(
        DataPtr(static_cast<const Tensor *>(out)), grad_scores, rows, d.seq,
        scale);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();

    const GemmShape shape{.outer = d.batch,
                          .inner = d.heads,
                          .m = d.seq,
                          .n = d.head_dim,
                          .k = d.seq};
    if (q->requires_grad()) {
      const GradTarget dq = GradForWrite(q.get());
      StridedGemm(shape, 1.0f, ScoreOperand(grad_scores, d, false),
                  HeadOperand(DataPtr(static_cast<const Tensor *>(k.get())), d,
                              false),
                  dq.accumulate ? 1.0f : 0.0f, HeadOutput(dq.ptr, d));
    }
    if (k->requires_grad()) {
      const GradTarget dk = GradForWrite(k.get());
      StridedGemm(shape, 1.0f, ScoreOperand(grad_scores, d, true),
                  HeadOperand(DataPtr(static_cast<const Tensor *>(q.get())), d,
                              false),
                  dk.accumulate ? 1.0f : 0.0f, HeadOutput(dk.ptr, d));
    }
  };
}

struct CombineDims {
  size_t batch;
  size_t heads;
  size_t seq;
  size_t keys;
  size_t embed;
  size_t head_dim;
};

CombineDims MakeCombineDims(const Tensor &probs, const Tensor &value) {
  const auto &shape = probs.shape();
  const size_t embed = value.shape()[2];
  return {.batch = shape[0],
          .heads = shape[1],
          .seq = shape[2],
          .keys = shape[3],
          .embed = embed,
          .head_dim = embed / shape[1]};
}

GemmOperand WeightOperand(const scalar_t *ptr, const CombineDims &d,
                          bool trans) {
  return {.ptr = ptr,
          .trans = trans,
          .ld = d.keys,
          .inner_stride = d.seq * d.keys,
          .outer_stride = d.heads * d.seq * d.keys};
}

GemmOperand RowsOperand(const scalar_t *ptr, const CombineDims &d,
                        size_t rows, bool trans) {
  return {.ptr = ptr,
          .trans = trans,
          .ld = d.embed,
          .inner_stride = d.head_dim,
          .outer_stride = rows * d.embed};
}

GemmOutput RowsOutput(scalar_t *ptr, const CombineDims &d, size_t rows) {
  return {.ptr = ptr,
          .ld = d.embed,
          .inner_stride = d.head_dim,
          .outer_stride = rows * d.embed};
}

void AttentionCombine(const OpArgs &args) {
  const CombineDims d = MakeCombineDims(*args.lhs, *args.rhs);
  const GemmShape shape{.outer = d.batch,
                        .inner = d.heads,
                        .m = d.seq,
                        .n = d.head_dim,
                        .k = d.keys};
  StridedGemm(shape, 1.0f, WeightOperand(DataPtr(args.lhs), d, false),
              RowsOperand(DataPtr(args.rhs), d, d.keys, false), 0.0f,
              RowsOutput(DataPtr(args.out), d, d.seq));
}

std::function<void()> AttentionCombineBackward(const GradArgs &args) {
  return [probs = args.lhs, value = args.rhs, out = args.out]() {
    const CombineDims d = MakeCombineDims(*probs, *value);
    const scalar_t *grad_out = GradPtr(out);
    if (probs->requires_grad()) {
      const GradTarget dp = GradForWrite(probs.get());
      const GemmShape shape{.outer = d.batch,
                            .inner = d.heads,
                            .m = d.seq,
                            .n = d.keys,
                            .k = d.head_dim};
      const GemmOutput target{.ptr = dp.ptr,
                              .ld = d.keys,
                              .inner_stride = d.seq * d.keys,
                              .outer_stride = d.heads * d.seq * d.keys};
      StridedGemm(shape, 1.0f, RowsOperand(grad_out, d, d.seq, false),
                  RowsOperand(DataPtr(static_cast<const Tensor *>(value.get())),
                              d, d.keys, true),
                  dp.accumulate ? 1.0f : 0.0f, target);
    }
    if (value->requires_grad()) {
      const GradTarget dv = GradForWrite(value.get());
      const GemmShape shape{.outer = d.batch,
                            .inner = d.heads,
                            .m = d.keys,
                            .n = d.head_dim,
                            .k = d.seq};
      StridedGemm(
          shape, 1.0f,
          WeightOperand(DataPtr(static_cast<const Tensor *>(probs.get())), d,
                        true),
          RowsOperand(grad_out, d, d.seq, false), dv.accumulate ? 1.0f : 0.0f,
          RowsOutput(dv.ptr, d, d.keys));
    }
  };
}

constexpr int kFlashHeadDim = 64;
constexpr int kFlashLd = kFlashHeadDim + 4;
constexpr int kFlashTile = 64;
constexpr int kFlashWarps = 4;
constexpr int kFlashThreads = kFlashWarps * static_cast<int>(kWarpSize);
constexpr int kFlashWarpRows = 16;
constexpr int kFlashQueryTile = 32;
constexpr int kFlashQueryLd = kFlashQueryTile + 4;
constexpr scalar_t kFlashNegInf = -1e30f;

struct FlashDims {
  int batch;
  int heads;
  int seq;
  int embed;
  scalar_t scale;
};

using TF32A = wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                             wmma::row_major>;
using TF32BRow = wmma::fragment<wmma::matrix_b, 16, 16, 8,
                                wmma::precision::tf32, wmma::row_major>;
using TF32BCol = wmma::fragment<wmma::matrix_b, 16, 16, 8,
                                wmma::precision::tf32, wmma::col_major>;
using TF32Acc = wmma::fragment<wmma::accumulator, 16, 16, 8, float>;

template <typename Fragment>
__device__ void ToTf32(Fragment &frag) {
  for (int i = 0; i < frag.num_elements; i++) {
    frag.x[i] = wmma::__float_to_tf32(frag.x[i]);
  }
}

template <typename Fragment>
__device__ void LoadTf32(Fragment &frag, const float *ptr, int ld) {
  wmma::load_matrix_sync(frag, ptr, ld);
  ToTf32(frag);
}

__device__ void LoadRows(float *dst, int dst_ld, const float *src, int row0,
                         int rows, int seq, int embed, float scale, int tid,
                         int threads) {
  constexpr int kVectors = kFlashHeadDim / 4;
  for (int idx = tid; idx < rows * kVectors; idx += threads) {
    const int r = idx / kVectors;
    const int c = (idx % kVectors) * 4;
    float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (row0 + r < seq) {
      value = *reinterpret_cast<const float4 *>(
          src + (static_cast<size_t>(row0 + r) * embed) + c);
      value.x *= scale;
      value.y *= scale;
      value.z *= scale;
      value.w *= scale;
    }
    *reinterpret_cast<float4 *>(dst + (r * dst_ld) + c) = value;
  }
}

__device__ void StoreRows(float *dst, const float *src, int src_ld, int row0,
                          int rows, int seq, int embed, float scale,
                          bool accumulate, int tid, int threads) {
  constexpr int kVectors = kFlashHeadDim / 4;
  for (int idx = tid; idx < rows * kVectors; idx += threads) {
    const int r = idx / kVectors;
    const int c = (idx % kVectors) * 4;
    if (row0 + r >= seq) {
      continue;
    }
    float4 value = *reinterpret_cast<const float4 *>(src + (r * src_ld) + c);
    value.x *= scale;
    value.y *= scale;
    value.z *= scale;
    value.w *= scale;
    auto *target = reinterpret_cast<float4 *>(
        dst + (static_cast<size_t>(row0 + r) * embed) + c);
    if (accumulate) {
      const float4 old = *target;
      value.x += old.x;
      value.y += old.y;
      value.z += old.z;
      value.w += old.w;
    }
    *target = value;
  }
}

__global__ void __launch_bounds__(kFlashThreads)
    FlashForwardKernel(const float *q, const float *k, const float *v,
                       float *out, float *lse, FlashDims d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ __align__(128) float smem[];
  const int warp = static_cast<int>(threadIdx.x / kWarpSize);
  const int lane = static_cast<int>(threadIdx.x % kWarpSize);
  float *ks = smem;
  float *vs = ks + (kFlashTile * kFlashLd);
  float *warp_s = vs + (kFlashTile * kFlashLd) +
                  (warp * 2 * kFlashWarpRows * kFlashLd);
  float *warp_o = warp_s + (kFlashWarpRows * kFlashLd);

  const int q_tile = static_cast<int>(gridDim.x - 1 - blockIdx.x);
  const int bh = static_cast<int>(blockIdx.y);
  const int b = bh / d.heads;
  const int h = bh % d.heads;
  const size_t base = (static_cast<size_t>(b) * d.seq * d.embed) +
                      (static_cast<size_t>(h) * kFlashHeadDim);
  const int q0 = q_tile * kFlashTile;
  const int warp_row0 = q0 + (warp * kFlashWarpRows);

  LoadRows(warp_o, kFlashLd, q + base, warp_row0, kFlashWarpRows, d.seq,
           d.embed, d.scale, lane, kWarpSize);
  __syncwarp();
  TF32A qf[kFlashHeadDim / 8];
  for (int kk = 0; kk < kFlashHeadDim / 8; kk++) {
    LoadTf32(qf[kk], warp_o + (kk * 8), kFlashLd);
  }
  __syncwarp();
  for (int idx = lane; idx < kFlashWarpRows * kFlashLd; idx += kWarpSize) {
    warp_o[idx] = 0.0f;
  }

  const int my_row = lane >> 1;
  const int half = lane & 1;
  const int t = warp_row0 + my_row;
  float m = kFlashNegInf;
  float l = 0.0f;

  for (int kv_tile = 0; kv_tile <= q_tile; kv_tile++) {
    const int k0 = kv_tile * kFlashTile;
    __syncthreads();
    LoadRows(ks, kFlashLd, k + base, k0, kFlashTile, d.seq, d.embed, 1.0f,
             static_cast<int>(threadIdx.x), kFlashThreads);
    LoadRows(vs, kFlashLd, v + base, k0, kFlashTile, d.seq, d.embed, 1.0f,
             static_cast<int>(threadIdx.x), kFlashThreads);
    __syncthreads();

    for (int j = 0; j < kFlashTile / 16; j++) {
      TF32Acc acc;
      wmma::fill_fragment(acc, 0.0f);
      for (int kk = 0; kk < kFlashHeadDim / 8; kk++) {
        TF32BCol kf;
        LoadTf32(kf, ks + (j * 16 * kFlashLd) + (kk * 8), kFlashLd);
        wmma::mma_sync(acc, qf[kk], kf, acc);
      }
      wmma::store_matrix_sync(warp_s + (j * 16), acc, kFlashLd,
                              wmma::mem_row_major);
    }
    __syncwarp();

    float *srow = warp_s + (my_row * kFlashLd) + (half * 32);
    const int col0 = k0 + (half * 32);
    float tile_max = kFlashNegInf;
    for (int c = 0; c < 32; c++) {
      const int s = col0 + c;
      if (s <= t && s < d.seq) {
        tile_max = fmaxf(tile_max, srow[c]);
      }
    }
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffU, tile_max, 1));
    const float m_new = fmaxf(m, tile_max);
    const float alpha = __expf(m - m_new);
    float sum = 0.0f;
    for (int c = 0; c < 32; c++) {
      const int s = col0 + c;
      const float p = (s <= t && s < d.seq) ? __expf(srow[c] - m_new) : 0.0f;
      srow[c] = p;
      sum += p;
    }
    sum += __shfl_xor_sync(0xffffffffU, sum, 1);
    l = (l * alpha) + sum;
    m = m_new;
    __syncwarp();

    TF32Acc pv[kFlashHeadDim / 16];
    for (int j = 0; j < kFlashHeadDim / 16; j++) {
      wmma::fill_fragment(pv[j], 0.0f);
    }
    for (int kk = 0; kk < kFlashTile / 8; kk++) {
      TF32A pf;
      LoadTf32(pf, warp_s + (kk * 8), kFlashLd);
      for (int j = 0; j < kFlashHeadDim / 16; j++) {
        TF32BRow vf;
        LoadTf32(vf, vs + (kk * 8 * kFlashLd) + (j * 16), kFlashLd);
        wmma::mma_sync(pv[j], pf, vf, pv[j]);
      }
    }
    __syncwarp();
    for (int j = 0; j < kFlashHeadDim / 16; j++) {
      wmma::store_matrix_sync(warp_s + (j * 16), pv[j], kFlashLd,
                              wmma::mem_row_major);
    }
    __syncwarp();
    float *orow = warp_o + (my_row * kFlashLd) + (half * 32);
    for (int c = 0; c < 32; c++) {
      orow[c] = (orow[c] * alpha) + srow[c];
    }
    __syncwarp();
  }

  if (t < d.seq) {
    const float inv = 1.0f / l;
    const float *orow = warp_o + (my_row * kFlashLd) + (half * 32);
    float *dst = out + base + (static_cast<size_t>(t) * d.embed) + (half * 32);
    for (int c = 0; c < 32; c++) {
      dst[c] = orow[c] * inv;
    }
    if (half == 0) {
      lse[(static_cast<size_t>(bh) * d.seq) + t] = m + logf(l);
    }
  }
#endif
}

__global__ void FlashDeltaKernel(const float *out, const float *grad_out,
                                 float *delta, FlashDims d) {
  const size_t row = (static_cast<size_t>(blockIdx.x) * kSoftmaxRowsPerBlock) +
                     threadIdx.y;
  const size_t rows = static_cast<size_t>(d.batch) * d.heads * d.seq;
  if (row >= rows) {
    return;
  }
  const size_t t = row % d.seq;
  const size_t bh = row / d.seq;
  const size_t b = bh / d.heads;
  const size_t h = bh % d.heads;
  const size_t offset = (((b * d.seq) + t) * d.embed) + (h * kFlashHeadDim);
  float dot = 0.0f;
  for (size_t i = threadIdx.x; i < kFlashHeadDim; i += kWarpSize) {
    dot += out[offset + i] * grad_out[offset + i];
  }
  dot = WarpSum(dot);
  if (threadIdx.x == 0) {
    delta[row] = dot;
  }
}

__global__ void __launch_bounds__(kFlashThreads)
    FlashBackwardKVKernel(const float *q, const float *k, const float *v,
                          const float *grad_out, const float *lse,
                          const float *delta, float *grad_k, float *grad_v,
                          bool accumulate_k, bool accumulate_v, FlashDims d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ __align__(128) float smem[];
  const int warp = static_cast<int>(threadIdx.x / kWarpSize);
  const int lane = static_cast<int>(threadIdx.x % kWarpSize);
  const int tid = static_cast<int>(threadIdx.x);
  float *ks = smem;
  float *vs = ks + (kFlashTile * kFlashLd);
  float *qs = vs + (kFlashTile * kFlashLd);
  float *dos = qs + (kFlashQueryTile * kFlashLd);
  float *lse_s = dos + (kFlashQueryTile * kFlashLd);
  float *delta_s = lse_s + kFlashQueryTile;
  float *buf_p = delta_s + kFlashQueryTile +
                 (warp * 2 * kFlashWarpRows * kFlashQueryLd);
  float *buf_ds = buf_p + (kFlashWarpRows * kFlashQueryLd);

  const int kv_tile = static_cast<int>(blockIdx.x);
  const int bh = static_cast<int>(blockIdx.y);
  const int b = bh / d.heads;
  const int h = bh % d.heads;
  const size_t base = (static_cast<size_t>(b) * d.seq * d.embed) +
                      (static_cast<size_t>(h) * kFlashHeadDim);
  const size_t row_base = static_cast<size_t>(bh) * d.seq;
  const int k0 = kv_tile * kFlashTile;

  LoadRows(ks, kFlashLd, k + base, k0, kFlashTile, d.seq, d.embed, 1.0f, tid,
           kFlashThreads);
  LoadRows(vs, kFlashLd, v + base, k0, kFlashTile, d.seq, d.embed, 1.0f, tid,
           kFlashThreads);
  const float *k_warp = ks + (warp * kFlashWarpRows * kFlashLd);
  const float *v_warp = vs + (warp * kFlashWarpRows * kFlashLd);

  TF32Acc dk[kFlashHeadDim / 16];
  TF32Acc dv[kFlashHeadDim / 16];
  for (int j = 0; j < kFlashHeadDim / 16; j++) {
    wmma::fill_fragment(dk[j], 0.0f);
    wmma::fill_fragment(dv[j], 0.0f);
  }

  const int key_row = lane >> 1;
  const int half = lane & 1;
  const int s = k0 + (warp * kFlashWarpRows) + key_row;
  const int q_tiles = (d.seq + kFlashQueryTile - 1) / kFlashQueryTile;

  for (int q_tile = k0 / kFlashQueryTile; q_tile < q_tiles; q_tile++) {
    const int q0 = q_tile * kFlashQueryTile;
    __syncthreads();
    LoadRows(qs, kFlashLd, q + base, q0, kFlashQueryTile, d.seq, d.embed,
             d.scale, tid, kFlashThreads);
    LoadRows(dos, kFlashLd, grad_out + base, q0, kFlashQueryTile, d.seq,
             d.embed, 1.0f, tid, kFlashThreads);
    if (tid < kFlashQueryTile) {
      const bool in_range = q0 + tid < d.seq;
      lse_s[tid] = in_range ? lse[row_base + q0 + tid] : 0.0f;
      delta_s[tid] = in_range ? delta[row_base + q0 + tid] : 0.0f;
    }
    __syncthreads();

    for (int j = 0; j < kFlashQueryTile / 16; j++) {
      TF32Acc st;
      TF32Acc dpt;
      wmma::fill_fragment(st, 0.0f);
      wmma::fill_fragment(dpt, 0.0f);
      for (int kk = 0; kk < kFlashHeadDim / 8; kk++) {
        TF32A kf;
        TF32A vf;
        TF32BCol qf;
        TF32BCol of;
        LoadTf32(kf, k_warp + (kk * 8), kFlashLd);
        LoadTf32(qf, qs + (j * 16 * kFlashLd) + (kk * 8), kFlashLd);
        wmma::mma_sync(st, kf, qf, st);
        LoadTf32(vf, v_warp + (kk * 8), kFlashLd);
        LoadTf32(of, dos + (j * 16 * kFlashLd) + (kk * 8), kFlashLd);
        wmma::mma_sync(dpt, vf, of, dpt);
      }
      wmma::store_matrix_sync(buf_p + (j * 16), st, kFlashQueryLd,
                              wmma::mem_row_major);
      wmma::store_matrix_sync(buf_ds + (j * 16), dpt, kFlashQueryLd,
                              wmma::mem_row_major);
    }
    __syncwarp();

    float *prow = buf_p + (key_row * kFlashQueryLd) + (half * 16);
    float *dsrow = buf_ds + (key_row * kFlashQueryLd) + (half * 16);
    for (int c = 0; c < 16; c++) {
      const int qi = (half * 16) + c;
      const int t = q0 + qi;
      const bool valid = t >= s && t < d.seq && s < d.seq;
      const float p = valid ? __expf(prow[c] - lse_s[qi]) : 0.0f;
      dsrow[c] = p * (dsrow[c] - delta_s[qi]);
      prow[c] = p;
    }
    __syncwarp();

    for (int kk = 0; kk < kFlashQueryTile / 8; kk++) {
      TF32A pf;
      TF32A dsf;
      LoadTf32(pf, buf_p + (kk * 8), kFlashQueryLd);
      LoadTf32(dsf, buf_ds + (kk * 8), kFlashQueryLd);
      for (int j = 0; j < kFlashHeadDim / 16; j++) {
        TF32BRow of;
        TF32BRow qf;
        LoadTf32(of, dos + (kk * 8 * kFlashLd) + (j * 16), kFlashLd);
        wmma::mma_sync(dv[j], pf, of, dv[j]);
        LoadTf32(qf, qs + (kk * 8 * kFlashLd) + (j * 16), kFlashLd);
        wmma::mma_sync(dk[j], dsf, qf, dk[j]);
      }
    }
  }

  __syncthreads();
  float *dk_out = ks + (warp * kFlashWarpRows * kFlashLd);
  float *dv_out = vs + (warp * kFlashWarpRows * kFlashLd);
  for (int j = 0; j < kFlashHeadDim / 16; j++) {
    wmma::store_matrix_sync(dk_out + (j * 16), dk[j], kFlashLd,
                            wmma::mem_row_major);
    wmma::store_matrix_sync(dv_out + (j * 16), dv[j], kFlashLd,
                            wmma::mem_row_major);
  }
  __syncthreads();
  StoreRows(grad_k + base, ks, kFlashLd, k0, kFlashTile, d.seq, d.embed, 1.0f,
            accumulate_k, tid, kFlashThreads);
  StoreRows(grad_v + base, vs, kFlashLd, k0, kFlashTile, d.seq, d.embed, 1.0f,
            accumulate_v, tid, kFlashThreads);
#endif
}

__global__ void __launch_bounds__(kFlashThreads)
    FlashBackwardQKernel(const float *q, const float *k, const float *v,
                         const float *grad_out, const float *lse,
                         const float *delta, float *grad_q, bool accumulate_q,
                         FlashDims d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ __align__(128) float smem[];
  const int warp = static_cast<int>(threadIdx.x / kWarpSize);
  const int lane = static_cast<int>(threadIdx.x % kWarpSize);
  const int tid = static_cast<int>(threadIdx.x);
  float *ks = smem;
  float *vs = ks + (kFlashTile * kFlashLd);
  float *buf_s = vs + (kFlashTile * kFlashLd) +
                 (warp * 2 * kFlashWarpRows * kFlashLd);
  float *buf_dp = buf_s + (kFlashWarpRows * kFlashLd);

  const int q_tile = static_cast<int>(gridDim.x - 1 - blockIdx.x);
  const int bh = static_cast<int>(blockIdx.y);
  const int b = bh / d.heads;
  const int h = bh % d.heads;
  const size_t base = (static_cast<size_t>(b) * d.seq * d.embed) +
                      (static_cast<size_t>(h) * kFlashHeadDim);
  const size_t row_base = static_cast<size_t>(bh) * d.seq;
  const int q0 = q_tile * kFlashTile;
  const int warp_row0 = q0 + (warp * kFlashWarpRows);

  LoadRows(buf_s, kFlashLd, q + base, warp_row0, kFlashWarpRows, d.seq,
           d.embed, d.scale, lane, kWarpSize);
  LoadRows(buf_dp, kFlashLd, grad_out + base, warp_row0, kFlashWarpRows,
           d.seq, d.embed, 1.0f, lane, kWarpSize);
  __syncwarp();
  TF32A qf[kFlashHeadDim / 8];
  TF32A dof[kFlashHeadDim / 8];
  for (int kk = 0; kk < kFlashHeadDim / 8; kk++) {
    LoadTf32(qf[kk], buf_s + (kk * 8), kFlashLd);
    LoadTf32(dof[kk], buf_dp + (kk * 8), kFlashLd);
  }

  const int my_row = lane >> 1;
  const int half = lane & 1;
  const int t = warp_row0 + my_row;
  const float row_lse = t < d.seq ? lse[row_base + t] : 0.0f;
  const float row_delta = t < d.seq ? delta[row_base + t] : 0.0f;

  TF32Acc dq[kFlashHeadDim / 16];
  for (int j = 0; j < kFlashHeadDim / 16; j++) {
    wmma::fill_fragment(dq[j], 0.0f);
  }

  for (int kv_tile = 0; kv_tile <= q_tile; kv_tile++) {
    const int k0 = kv_tile * kFlashTile;
    __syncthreads();
    LoadRows(ks, kFlashLd, k + base, k0, kFlashTile, d.seq, d.embed, 1.0f, tid,
             kFlashThreads);
    LoadRows(vs, kFlashLd, v + base, k0, kFlashTile, d.seq, d.embed, 1.0f, tid,
             kFlashThreads);
    __syncthreads();

    for (int j = 0; j < kFlashTile / 16; j++) {
      TF32Acc sacc;
      TF32Acc dpacc;
      wmma::fill_fragment(sacc, 0.0f);
      wmma::fill_fragment(dpacc, 0.0f);
      for (int kk = 0; kk < kFlashHeadDim / 8; kk++) {
        TF32BCol kf;
        TF32BCol vf;
        LoadTf32(kf, ks + (j * 16 * kFlashLd) + (kk * 8), kFlashLd);
        wmma::mma_sync(sacc, qf[kk], kf, sacc);
        LoadTf32(vf, vs + (j * 16 * kFlashLd) + (kk * 8), kFlashLd);
        wmma::mma_sync(dpacc, dof[kk], vf, dpacc);
      }
      __syncwarp();
      wmma::store_matrix_sync(buf_s + (j * 16), sacc, kFlashLd,
                              wmma::mem_row_major);
      wmma::store_matrix_sync(buf_dp + (j * 16), dpacc, kFlashLd,
                              wmma::mem_row_major);
    }
    __syncwarp();

    float *srow = buf_s + (my_row * kFlashLd) + (half * 32);
    const float *dprow = buf_dp + (my_row * kFlashLd) + (half * 32);
    for (int c = 0; c < 32; c++) {
      const int s = k0 + (half * 32) + c;
      const bool valid = s <= t && s < d.seq && t < d.seq;
      const float p = valid ? __expf(srow[c] - row_lse) : 0.0f;
      srow[c] = p * (dprow[c] - row_delta);
    }
    __syncwarp();

    for (int kk = 0; kk < kFlashTile / 8; kk++) {
      TF32A dsf;
      LoadTf32(dsf, buf_s + (kk * 8), kFlashLd);
      for (int j = 0; j < kFlashHeadDim / 16; j++) {
        TF32BRow kf;
        LoadTf32(kf, ks + (kk * 8 * kFlashLd) + (j * 16), kFlashLd);
        wmma::mma_sync(dq[j], dsf, kf, dq[j]);
      }
    }
    __syncwarp();
  }

  for (int j = 0; j < kFlashHeadDim / 16; j++) {
    wmma::store_matrix_sync(buf_s + (j * 16), dq[j], kFlashLd,
                            wmma::mem_row_major);
  }
  __syncwarp();
  StoreRows(grad_q + base, buf_s, kFlashLd, warp_row0, kFlashWarpRows, d.seq,
            d.embed, d.scale, accumulate_q, lane, kWarpSize);
#endif
}

constexpr size_t kFlashForwardSmem =
    ((2 * kFlashTile * kFlashLd) + (kFlashWarps * 2 * kFlashWarpRows * kFlashLd)) *
    sizeof(float);
constexpr size_t kFlashBackwardKVSmem =
    ((2 * kFlashTile * kFlashLd) + (2 * kFlashQueryTile * kFlashLd) +
     (2 * kFlashQueryTile) +
     (kFlashWarps * 2 * kFlashWarpRows * kFlashQueryLd)) *
    sizeof(float);
constexpr size_t kFlashBackwardQSmem = kFlashForwardSmem;

bool UseFlashKernels(const FlashDims &d) {
  static const int major = [] {
    int device = 0;
    int value = 0;
    MICROGRAD_CUDA_CHECK(cudaGetDevice(&device));
    MICROGRAD_CUDA_CHECK(cudaDeviceGetAttribute(
        &value, cudaDevAttrComputeCapabilityMajor, device));
    return value;
  }();
  return major >= 8 && CudaContext::instance().allowTf32() &&
         d.embed == d.heads * kFlashHeadDim;
}

void EnableFlashSmem() {
  static const bool done = [] {
    MICROGRAD_CUDA_CHECK(cudaFuncSetAttribute(
        FlashForwardKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kFlashForwardSmem)));
    MICROGRAD_CUDA_CHECK(cudaFuncSetAttribute(
        FlashBackwardKVKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kFlashBackwardKVSmem)));
    MICROGRAD_CUDA_CHECK(cudaFuncSetAttribute(
        FlashBackwardQKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kFlashBackwardQSmem)));
    return true;
  }();
  (void)done;
}

FlashDims MakeFlashDims(const Tensor &q, size_t heads, scalar_t scale) {
  const auto &shape = q.shape();
  return {.batch = static_cast<int>(shape[0]),
          .heads = static_cast<int>(heads),
          .seq = static_cast<int>(shape[1]),
          .embed = static_cast<int>(shape[2]),
          .scale = scale};
}

ProbsDims ToProbsDims(const FlashDims &d) {
  return {.batch = static_cast<size_t>(d.batch),
          .heads = static_cast<size_t>(d.heads),
          .seq = static_cast<size_t>(d.seq),
          .embed = static_cast<size_t>(d.embed),
          .head_dim = static_cast<size_t>(d.embed / d.heads)};
}

class ScratchBuffer {
 public:
  explicit ScratchBuffer(size_t count)
      : bytes_(count * sizeof(scalar_t)),
        ptr_(static_cast<scalar_t *>(
            CudaContext::instance().allocate(bytes_))) {}
  ~ScratchBuffer() { CudaContext::instance().deallocate(ptr_, bytes_); }
  ScratchBuffer(const ScratchBuffer &) = delete;
  ScratchBuffer &operator=(const ScratchBuffer &) = delete;
  ScratchBuffer(ScratchBuffer &&) = delete;
  ScratchBuffer &operator=(ScratchBuffer &&) = delete;
  scalar_t *get() const { return ptr_; }

 private:
  size_t bytes_;
  scalar_t *ptr_;
};

void MaterializedProbs(const ProbsDims &d, const scalar_t *q, const scalar_t *k,
                       scalar_t scale, scalar_t *probs, scalar_t *lse) {
  const GemmShape shape{
      .outer = d.batch, .inner = d.heads, .m = d.seq, .n = d.seq, .k = d.head_dim};
  StridedGemm(shape, scale, HeadOperand(q, d, false), HeadOperand(k, d, true),
              0.0f, ScoreOutput(probs, d));
  const size_t rows = d.batch * d.heads * d.seq;
  const dim3 block(kWarpSize, kSoftmaxRowsPerBlock);
  CausalSoftmaxKernel<<<SoftmaxGrid(rows), block, 0,
                        CudaContext::instance().stream()>>>(probs, lse, rows,
                                                            d.seq);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void CausalSelfAttention(const OpArgs &args) {
  const FlashDims fd =
      MakeFlashDims(*args.lhs, static_cast<size_t>(args.dim), args.scalar);
  cudaStream_t stream = CudaContext::instance().stream();
  if (UseFlashKernels(fd)) {
    EnableFlashSmem();
    const dim3 grid(
        static_cast<unsigned int>((fd.seq + kFlashTile - 1) / kFlashTile),
        static_cast<unsigned int>(fd.batch * fd.heads));
    FlashForwardKernel<<<grid, kFlashThreads, kFlashForwardSmem, stream>>>(
        DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.extra),
        DataPtr(args.out), DataPtr(args.aux), fd);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  const ProbsDims d = ToProbsDims(fd);
  const ScratchBuffer probs(d.batch * d.heads * d.seq * d.seq);
  MaterializedProbs(d, DataPtr(args.lhs), DataPtr(args.rhs), fd.scale,
                    probs.get(), DataPtr(args.aux));
  const GemmShape shape{
      .outer = d.batch, .inner = d.heads, .m = d.seq, .n = d.head_dim, .k = d.seq};
  StridedGemm(shape, 1.0f, ScoreOperand(probs.get(), d, false),
              HeadOperand(DataPtr(args.extra), d, false), 0.0f,
              HeadOutput(DataPtr(args.out), d));
}

void FlashBackward(const FlashDims &fd, const Tensor &q, const Tensor &k,
                   const Tensor &v, const Tensor &out, const scalar_t *grad_out,
                   const Tensor &lse, Tensor *q_grad_owner, Tensor *k_grad_owner,
                   Tensor *v_grad_owner) {
  cudaStream_t stream = CudaContext::instance().stream();
  EnableFlashSmem();
  const size_t rows = static_cast<size_t>(fd.batch) * fd.heads * fd.seq;
  const ScratchBuffer delta(rows);
  const dim3 delta_block(kWarpSize, kSoftmaxRowsPerBlock);
  FlashDeltaKernel<<<SoftmaxGrid(rows), delta_block, 0, stream>>>(
      DataPtr(&out), grad_out, delta.get(), fd);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();

  const unsigned int tiles =
      static_cast<unsigned int>((fd.seq + kFlashTile - 1) / kFlashTile);
  const dim3 grid(tiles, static_cast<unsigned int>(fd.batch * fd.heads));
  const GradTarget dv = GradForWrite(v_grad_owner);
  const GradTarget dk = GradForWrite(k_grad_owner);
  FlashBackwardKVKernel<<<grid, kFlashThreads, kFlashBackwardKVSmem, stream>>>(
      DataPtr(&q), DataPtr(&k), DataPtr(&v), grad_out, DataPtr(&lse),
      delta.get(), dk.ptr, dv.ptr, dk.accumulate, dv.accumulate, fd);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  const GradTarget dq = GradForWrite(q_grad_owner);
  FlashBackwardQKernel<<<grid, kFlashThreads, kFlashBackwardQSmem, stream>>>(
      DataPtr(&q), DataPtr(&k), DataPtr(&v), grad_out, DataPtr(&lse),
      delta.get(), dq.ptr, dq.accumulate, fd);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void MaterializedBackward(const FlashDims &fd, const Tensor &q,
                          const Tensor &k, const Tensor &v,
                          const scalar_t *grad_out, Tensor *q_grad_owner,
                          Tensor *k_grad_owner, Tensor *v_grad_owner) {
  const ProbsDims d = ToProbsDims(fd);
  const size_t score_count = d.batch * d.heads * d.seq * d.seq;
  const ScratchBuffer probs(score_count);
  const ScratchBuffer grad_scores(score_count);
  MaterializedProbs(d, DataPtr(&q), DataPtr(&k), fd.scale, probs.get(),
                    nullptr);

  const GemmShape scores_shape{
      .outer = d.batch, .inner = d.heads, .m = d.seq, .n = d.seq, .k = d.head_dim};
  StridedGemm(scores_shape, 1.0f, HeadOperand(grad_out, d, false),
              HeadOperand(DataPtr(&v), d, true), 0.0f,
              ScoreOutput(grad_scores.get(), d));

  const GemmShape rows_shape{
      .outer = d.batch, .inner = d.heads, .m = d.seq, .n = d.head_dim, .k = d.seq};
  const GradTarget dv = GradForWrite(v_grad_owner);
  StridedGemm(rows_shape, 1.0f, ScoreOperand(probs.get(), d, true),
              HeadOperand(grad_out, d, false), dv.accumulate ? 1.0f : 0.0f,
              HeadOutput(dv.ptr, d));

  const size_t rows = d.batch * d.heads * d.seq;
  const dim3 block(kWarpSize, kSoftmaxRowsPerBlock);
  CausalSoftmaxBackwardKernel<<<SoftmaxGrid(rows), block, 0,
                                CudaContext::instance().stream()>>>(
      probs.get(), grad_scores.get(), rows, d.seq, fd.scale);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();

  const GradTarget dq = GradForWrite(q_grad_owner);
  StridedGemm(rows_shape, 1.0f, ScoreOperand(grad_scores.get(), d, false),
              HeadOperand(DataPtr(&k), d, false), dq.accumulate ? 1.0f : 0.0f,
              HeadOutput(dq.ptr, d));
  const GradTarget dk = GradForWrite(k_grad_owner);
  StridedGemm(rows_shape, 1.0f, ScoreOperand(grad_scores.get(), d, true),
              HeadOperand(DataPtr(&q), d, false), dk.accumulate ? 1.0f : 0.0f,
              HeadOutput(dk.ptr, d));
}

std::function<void()> CausalSelfAttentionBackward(const GradArgs &args) {
  return [q = args.lhs, k = args.rhs, v = args.extra, out = args.out,
          lse = args.aux, scale = args.scalar]() {
    const FlashDims fd = MakeFlashDims(*q, lse->shape()[1], scale);
    const scalar_t *grad_out = GradPtr(out);
    if (UseFlashKernels(fd) && q.get() != k.get() && q.get() != v.get() &&
        k.get() != v.get()) {
      FlashBackward(fd, *q, *k, *v, *out, grad_out, *lse, q.get(), k.get(),
                    v.get());
      return;
    }
    MaterializedBackward(fd, *q, *k, *v, grad_out, q.get(), k.get(), v.get());
  };
}

}  // namespace

void RegisterAttentionOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kFlashAttention, Device::CUDA, FlashAttention);
  registry.RegisterBackward(OpId::kFlashAttention, Device::CUDA,
                            FlashAttentionBackward);
  registry.Register(OpId::kCausalAttentionProbs, Device::CUDA,
                    CausalAttentionProbs);
  registry.RegisterBackward(OpId::kCausalAttentionProbs, Device::CUDA,
                            CausalAttentionProbsBackward);
  registry.Register(OpId::kAttentionCombine, Device::CUDA, AttentionCombine);
  registry.RegisterBackward(OpId::kAttentionCombine, Device::CUDA,
                            AttentionCombineBackward);
  registry.Register(OpId::kCausalSelfAttention, Device::CUDA,
                    CausalSelfAttention);
  registry.RegisterBackward(OpId::kCausalSelfAttention, Device::CUDA,
                            CausalSelfAttentionBackward);
}

}  // namespace micrograd::cuda::ops

#endif
