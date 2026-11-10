#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cuda/ops/Bf16.h"
#include "micrograd/backends/cuda/ops/Grad.h"
#include "micrograd/backends/cuda/ops/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::cuda::ops {
namespace {

constexpr int kBlockSize = 256;

__global__ void RmsNormKernel(const scalar_t *in, const scalar_t *gain,
                              scalar_t *out, size_t n, scalar_t eps) {
  __shared__ scalar_t shared[kBlockSize];
  size_t row = blockIdx.x;
  size_t tid = threadIdx.x;
  const scalar_t *row_in = in + (row * n);

  scalar_t local_sum_sq = 0.0f;
  for (size_t i = tid; i < n; i += blockDim.x) {
    local_sum_sq += row_in[i] * row_in[i];
  }
  shared[tid] = local_sum_sq;
  __syncthreads();
  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  scalar_t rstd = 1.0f / sqrtf((shared[0] / static_cast<scalar_t>(n)) + eps);
  __syncthreads();

  scalar_t *row_out = out + (row * n);
  for (size_t i = tid; i < n; i += blockDim.x) {
    row_out[i] = row_in[i] * rstd * gain[i];
  }
}

__global__ void RmsNormBackwardKernel(const scalar_t *in, const scalar_t *gain,
                                      const scalar_t *out_grad,
                                      scalar_t *in_grad, scalar_t *gain_grad,
                                      size_t n, scalar_t eps) {
  __shared__ scalar_t shared[kBlockSize];
  size_t row = blockIdx.x;
  size_t tid = threadIdx.x;
  const scalar_t *row_in = in + (row * n);
  const scalar_t *row_out_grad = out_grad + (row * n);

  scalar_t local_sum_sq = 0.0f;
  for (size_t i = tid; i < n; i += blockDim.x) {
    local_sum_sq += row_in[i] * row_in[i];
  }
  shared[tid] = local_sum_sq;
  __syncthreads();
  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  scalar_t rstd = 1.0f / sqrtf((shared[0] / static_cast<scalar_t>(n)) + eps);
  __syncthreads();

  scalar_t local_mean_dxhat_xhat = 0.0f;
  for (size_t i = tid; i < n; i += blockDim.x) {
    scalar_t xhat_i = row_in[i] * rstd;
    scalar_t dxhat_i = row_out_grad[i] * gain[i];
    local_mean_dxhat_xhat += dxhat_i * xhat_i;
    atomicAdd(&gain_grad[i], row_out_grad[i] * xhat_i);
  }
  shared[tid] = local_mean_dxhat_xhat;
  __syncthreads();
  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  scalar_t mean_dxhat_xhat = shared[0] / static_cast<scalar_t>(n);
  __syncthreads();

  scalar_t *row_in_grad = in_grad + (row * n);
  for (size_t i = tid; i < n; i += blockDim.x) {
    scalar_t xhat_i = row_in[i] * rstd;
    scalar_t dxhat_i = row_out_grad[i] * gain[i];
    row_in_grad[i] += rstd * (dxhat_i - (xhat_i * mean_dxhat_xhat));
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

void RmsNorm(const OpArgs &args) {
  size_t n = args.rhs->size();
  size_t outer = args.lhs->size() / n;
  RmsNormKernel<<<static_cast<int>(outer), kBlockSize, 0,
                  CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), n, args.scalar);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

std::function<void()> RmsNormBackward(const GradArgs &args) {
  return
      [out = args.out, lhs = args.lhs, gain = args.rhs, eps = args.scalar]() {
        size_t n = gain->size();
        size_t outer = lhs->size() / n;
        RmsNormBackwardKernel<<<static_cast<int>(outer), kBlockSize, 0,
                                CudaContext::instance().stream()>>>(
            DataPtr(lhs.get()), DataPtr(gain.get()), GradPtr(out),
            GradPtr(lhs.get()), GradPtr(gain.get()), n, eps);
        MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
      };
}

constexpr int kRmsWarpSize = 32;
constexpr int kRmsWarps = 8;
constexpr int kRmsMaxChunks = 4;
constexpr unsigned int kRmsBackwardMaxBlocks = 512;
constexpr size_t kRmsMaxSmem = 48 * 1024;

__device__ float RmsWarpSum(float value) {
  for (int offset = kRmsWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffU, value, offset);
  }
  return value;
}

template <int V>
__device__ void LoadRow(const bf16 *ptr, float (&values)[V]) {
  if constexpr (V == kBf16Vector) {
    const Bf16x8 x = LoadBf16x8(ptr);
    for (int j = 0; j < V; j++) {
      values[j] = x.v[j];
    }
  } else if constexpr (V == 2) {
    const float2 f =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(ptr));
    values[0] = f.x;
    values[1] = f.y;
  } else {
    for (int j = 0; j < V; j++) {
      values[j] = __bfloat162float(ptr[j]);
    }
  }
}

template <int V>
__device__ void StoreRow(bf16 *ptr, const float (&values)[V]) {
  if constexpr (V == kBf16Vector) {
    Bf16x8 x;
    for (int j = 0; j < V; j++) {
      x.v[j] = values[j];
    }
    StoreBf16x8(ptr, x);
  } else if constexpr (V == 2) {
    *reinterpret_cast<__nv_bfloat162 *>(ptr) =
        __floats2bfloat162_rn(values[0], values[1]);
  } else {
    for (int j = 0; j < V; j++) {
      ptr[j] = __float2bfloat16(values[j]);
    }
  }
}

template <int V>
__device__ float RowRstd(const bf16 *x, int n, int lane, float eps) {
  float sum_sq = 0.0f;
  for (int i = lane * V; i < n; i += kRmsWarpSize * V) {
    float v[V];
    LoadRow<V>(x + i, v);
    for (int j = 0; j < V; j++) {
      sum_sq += v[j] * v[j];
    }
  }
  return rsqrtf((RmsWarpSum(sum_sq) / static_cast<float>(n)) + eps);
}

template <int V>
__global__ void RmsNormBf16Kernel(const bf16 *in, const float *gain, bf16 *out,
                                  size_t rows, int n, float eps) {
  const int lane = static_cast<int>(threadIdx.x) % kRmsWarpSize;
  const size_t row = (static_cast<size_t>(blockIdx.x) * kRmsWarps) +
                     (threadIdx.x / kRmsWarpSize);
  if (row >= rows) {
    return;
  }
  const bf16 *x = in + (row * n);
  const float rstd = RowRstd<V>(x, n, lane, eps);
  bf16 *y = out + (row * n);
  for (int i = lane * V; i < n; i += kRmsWarpSize * V) {
    float v[V];
    LoadRow<V>(x + i, v);
    for (int j = 0; j < V; j++) {
      v[j] *= rstd * gain[i + j];
    }
    StoreRow<V>(y + i, v);
  }
}

template <int V, int kChunks>
__global__ void RmsNormBackwardBf16Kernel(const bf16 *in, const float *gain,
                                          const bf16 *out_grad, bf16 *in_grad,
                                          bool accumulate, float *partial,
                                          size_t rows, int n, float eps) {
  extern __shared__ float rms_smem[];
  const int lane = static_cast<int>(threadIdx.x) % kRmsWarpSize;
  const int warp = static_cast<int>(threadIdx.x) / kRmsWarpSize;
  const size_t step = static_cast<size_t>(gridDim.x) * kRmsWarps;
  float dg[kChunks][V] = {};
  float w[kChunks][V];
  for (int c = 0; c < kChunks; c++) {
    const int i = (lane + (c * kRmsWarpSize)) * V;
    for (int j = 0; j < V; j++) {
      w[c][j] = i < n ? gain[i + j] : 0.0f;
    }
  }
  for (size_t row = (static_cast<size_t>(blockIdx.x) * kRmsWarps) + warp;
       row < rows; row += step) {
    const bf16 *x = in + (row * n);
    const bf16 *dy = out_grad + (row * n);
    float xv[kChunks][V];
    float gv[kChunks][V];
    float sum_sq = 0.0f;
    for (int c = 0; c < kChunks; c++) {
      const int i = (lane + (c * kRmsWarpSize)) * V;
      if (i < n) {
        LoadRow<V>(x + i, xv[c]);
        LoadRow<V>(dy + i, gv[c]);
      } else {
        for (int j = 0; j < V; j++) {
          xv[c][j] = 0.0f;
          gv[c][j] = 0.0f;
        }
      }
      for (int j = 0; j < V; j++) {
        sum_sq += xv[c][j] * xv[c][j];
      }
    }
    const float rstd =
        rsqrtf((RmsWarpSum(sum_sq) / static_cast<float>(n)) + eps);
    float dot = 0.0f;
    for (int c = 0; c < kChunks; c++) {
      for (int j = 0; j < V; j++) {
        const float xhat = xv[c][j] * rstd;
        dot += gv[c][j] * w[c][j] * xhat;
        dg[c][j] += gv[c][j] * xhat;
        xv[c][j] = xhat;
      }
    }
    const float mean_dot = RmsWarpSum(dot) / static_cast<float>(n);
    bf16 *dx = in_grad + (row * n);
    for (int c = 0; c < kChunks; c++) {
      const int i = (lane + (c * kRmsWarpSize)) * V;
      if (i >= n) {
        continue;
      }
      float r[V];
      if (accumulate) {
        LoadRow<V>(dx + i, r);
      } else {
        for (int j = 0; j < V; j++) {
          r[j] = 0.0f;
        }
      }
      for (int j = 0; j < V; j++) {
        r[j] += rstd * ((gv[c][j] * w[c][j]) - (xv[c][j] * mean_dot));
      }
      StoreRow<V>(dx + i, r);
    }
  }
  float *warp_dg = rms_smem + (static_cast<size_t>(warp) * n);
  for (int c = 0; c < kChunks; c++) {
    const int i = (lane + (c * kRmsWarpSize)) * V;
    if (i >= n) {
      continue;
    }
    for (int j = 0; j < V; j++) {
      warp_dg[i + j] = dg[c][j];
    }
  }
  __syncthreads();
  for (int i = static_cast<int>(threadIdx.x); i < n;
       i += static_cast<int>(blockDim.x)) {
    float total = 0.0f;
    for (int k = 0; k < kRmsWarps; k++) {
      total += rms_smem[(k * n) + i];
    }
    partial[(static_cast<size_t>(blockIdx.x) * n) + i] = total;
  }
}

__global__ void RmsNormGainGradKernel(const float *partial, unsigned int blocks,
                                      int n, float *gain_grad) {
  __shared__ float shared[kBlockSize];
  const int col = static_cast<int>(blockIdx.x);
  float total = 0.0f;
  for (unsigned int b = threadIdx.x; b < blocks; b += blockDim.x) {
    total += partial[(static_cast<size_t>(b) * n) + col];
  }
  shared[threadIdx.x] = total;
  __syncthreads();
  for (unsigned int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    gain_grad[col] += shared[0];
  }
}

int RmsVectorWidth(size_t n) {
  if (n % kBf16Vector == 0 && n >= kRmsWarpSize * kBf16Vector) {
    return kBf16Vector;
  }
  return n % 2 == 0 ? 2 : 1;
}

template <int V>
void LaunchRmsNormBf16(const OpArgs &args, size_t rows, int n) {
  const auto blocks =
      static_cast<unsigned int>((rows + kRmsWarps - 1) / kRmsWarps);
  RmsNormBf16Kernel<V><<<blocks, kRmsWarps * kRmsWarpSize, 0,
                         CudaContext::instance().stream()>>>(
      Bf16Data(args.lhs), DataPtr(args.rhs), Bf16Data(args.out), rows, n,
      args.scalar);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void RmsNormBf16(const OpArgs &args) {
  const size_t n = args.rhs->size();
  const size_t rows = args.lhs->size() / n;
  if (rows == 0) {
    return;
  }
  const int width = static_cast<int>(n);
  switch (RmsVectorWidth(n)) {
    case kBf16Vector:
      LaunchRmsNormBf16<kBf16Vector>(args, rows, width);
      break;
    case 2:
      LaunchRmsNormBf16<2>(args, rows, width);
      break;
    default:
      LaunchRmsNormBf16<1>(args, rows, width);
      break;
  }
}

template <int V, int kChunks>
void LaunchRmsNormBackwardBf16(Tensor *lhs, Tensor *gain, Tensor *out,
                               float eps) {
  const size_t n = gain->size();
  const size_t rows = lhs->size() / n;
  const unsigned int blocks = std::min<unsigned int>(
      static_cast<unsigned int>((rows + kRmsWarps - 1) / kRmsWarps),
      kRmsBackwardMaxBlocks);
  const size_t smem = kRmsWarps * n * sizeof(float);
  cudaStream_t stream = CudaContext::instance().stream();
  const Bf16GradTarget dx = Bf16GradForWrite(lhs);
  const size_t partial_bytes = blocks * n * sizeof(float);
  auto *partial =
      static_cast<float *>(CudaContext::instance().allocate(partial_bytes));
  RmsNormBackwardBf16Kernel<V, kChunks>
      <<<blocks, kRmsWarps * kRmsWarpSize, smem, stream>>>(
          Bf16Data(lhs), DataPtr(gain), Bf16Grad(out), dx.ptr, dx.accumulate,
          partial, rows, static_cast<int>(n), eps);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  RmsNormGainGradKernel<<<static_cast<unsigned int>(n), kBlockSize, 0,
                          stream>>>(partial, blocks, static_cast<int>(n),
                                    GradPtr(gain));
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  CudaContext::instance().deallocate(partial, partial_bytes);
}

template <int V>
void DispatchRmsNormBackwardBf16(Tensor *lhs, Tensor *gain, Tensor *out,
                                 float eps) {
  const size_t n = gain->size();
  const size_t chunks = (n + (kRmsWarpSize * V) - 1) / (kRmsWarpSize * V);
  switch (chunks) {
    case 1:
      LaunchRmsNormBackwardBf16<V, 1>(lhs, gain, out, eps);
      break;
    case 2:
      LaunchRmsNormBackwardBf16<V, 2>(lhs, gain, out, eps);
      break;
    case 3:
      LaunchRmsNormBackwardBf16<V, 3>(lhs, gain, out, eps);
      break;
    default:
      LaunchRmsNormBackwardBf16<V, kRmsMaxChunks>(lhs, gain, out, eps);
      break;
  }
}

std::function<void()> RmsNormBackwardBf16(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, gain = args.rhs,
          eps = args.scalar]() {
    const size_t n = gain->size();
    if (lhs->size() < n) {
      return;
    }
    const int v = RmsVectorWidth(n);
    const size_t chunks = (n + (kRmsWarpSize * v) - 1) / (kRmsWarpSize * v);
    if (chunks > kRmsMaxChunks || kRmsWarps * n * sizeof(float) > kRmsMaxSmem) {
      throw std::runtime_error("bf16 rms_norm backward: row is too wide");
    }
    switch (v) {
      case kBf16Vector:
        DispatchRmsNormBackwardBf16<kBf16Vector>(lhs.get(), gain.get(), out,
                                                 eps);
        break;
      case 2:
        DispatchRmsNormBackwardBf16<2>(lhs.get(), gain.get(), out, eps);
        break;
      default:
        DispatchRmsNormBackwardBf16<1>(lhs.get(), gain.get(), out, eps);
        break;
    }
  };
}

__device__ inline float LoadScalar(const float *p) { return *p; }
__device__ inline float LoadScalar(const bf16 *p) {
  return __bfloat162float(*p);
}
__device__ inline void StoreScalar(float *p, float v) { *p = v; }
__device__ inline void StoreScalar(bf16 *p, float v) {
  *p = __float2bfloat16(v);
}

constexpr int kHeadMaxChunks = 4;

template <typename T, int kChunks>
__global__ void HeadNormRotaryKernel(const T *in, const float *gain,
                                     const float *cos_table,
                                     const float *sin_table, T *out,
                                     size_t rows, int half, int n_head,
                                     int seq_len, float eps) {
  const int lane = static_cast<int>(threadIdx.x) % kRmsWarpSize;
  const size_t row = (static_cast<size_t>(blockIdx.x) * kRmsWarps) +
                     (threadIdx.x / kRmsWarpSize);
  if (row >= rows) {
    return;
  }
  const size_t width = 2 * static_cast<size_t>(half);
  const T *x = in + (row * width);
  float lo[kChunks];
  float hi[kChunks];
  float sum_sq = 0.0f;
  for (int c = 0; c < kChunks; c++) {
    const int j = lane + (c * kRmsWarpSize);
    lo[c] = j < half ? LoadScalar(x + j) : 0.0f;
    hi[c] = j < half ? LoadScalar(x + j + half) : 0.0f;
    sum_sq += (lo[c] * lo[c]) + (hi[c] * hi[c]);
  }
  if (gain != nullptr) {
    const float rstd =
        rsqrtf((RmsWarpSum(sum_sq) / static_cast<float>(width)) + eps);
    for (int c = 0; c < kChunks; c++) {
      const int j = lane + (c * kRmsWarpSize);
      if (j < half) {
        lo[c] *= rstd * gain[j];
        hi[c] *= rstd * gain[j + half];
      }
    }
  }
  if (cos_table != nullptr) {
    const size_t t =
        (row / static_cast<size_t>(n_head)) % static_cast<size_t>(seq_len);
    for (int c = 0; c < kChunks; c++) {
      const int j = lane + (c * kRmsWarpSize);
      if (j < half) {
        const float cs = cos_table[(t * half) + j];
        const float sn = sin_table[(t * half) + j];
        const float a = lo[c];
        lo[c] = (a * cs) + (hi[c] * sn);
        hi[c] = (hi[c] * cs) - (a * sn);
      }
    }
  }
  T *y = out + (row * width);
  for (int c = 0; c < kChunks; c++) {
    const int j = lane + (c * kRmsWarpSize);
    if (j < half) {
      StoreScalar(y + j, lo[c]);
      StoreScalar(y + j + half, hi[c]);
    }
  }
}

template <typename T, int kChunks>
__global__ void HeadNormRotaryBackwardKernel(
    const T *in, const float *gain, const float *cos_table,
    const float *sin_table, const T *out_grad, T *in_grad, bool accumulate,
    float *partial, size_t rows, int half, int n_head, int seq_len, float eps) {
  extern __shared__ float head_smem[];
  const int lane = static_cast<int>(threadIdx.x) % kRmsWarpSize;
  const int warp = static_cast<int>(threadIdx.x) / kRmsWarpSize;
  const size_t step = static_cast<size_t>(gridDim.x) * kRmsWarps;
  const size_t width = 2 * static_cast<size_t>(half);
  float g_lo[kChunks];
  float g_hi[kChunks];
  float dg_lo[kChunks] = {};
  float dg_hi[kChunks] = {};
  for (int c = 0; c < kChunks; c++) {
    const int j = lane + (c * kRmsWarpSize);
    g_lo[c] = gain != nullptr && j < half ? gain[j] : 1.0f;
    g_hi[c] = gain != nullptr && j < half ? gain[j + half] : 1.0f;
  }
  for (size_t row = (static_cast<size_t>(blockIdx.x) * kRmsWarps) + warp;
       row < rows; row += step) {
    const T *dy = out_grad + (row * width);
    float lo[kChunks];
    float hi[kChunks];
    for (int c = 0; c < kChunks; c++) {
      const int j = lane + (c * kRmsWarpSize);
      lo[c] = j < half ? LoadScalar(dy + j) : 0.0f;
      hi[c] = j < half ? LoadScalar(dy + j + half) : 0.0f;
    }
    if (cos_table != nullptr) {
      const size_t t =
          (row / static_cast<size_t>(n_head)) % static_cast<size_t>(seq_len);
      for (int c = 0; c < kChunks; c++) {
        const int j = lane + (c * kRmsWarpSize);
        if (j < half) {
          const float cs = cos_table[(t * half) + j];
          const float sn = sin_table[(t * half) + j];
          const float a = lo[c];
          lo[c] = (a * cs) - (hi[c] * sn);
          hi[c] = (hi[c] * cs) + (a * sn);
        }
      }
    }
    if (gain != nullptr) {
      const T *x = in + (row * width);
      float x_lo[kChunks];
      float x_hi[kChunks];
      float sum_sq = 0.0f;
      for (int c = 0; c < kChunks; c++) {
        const int j = lane + (c * kRmsWarpSize);
        x_lo[c] = j < half ? LoadScalar(x + j) : 0.0f;
        x_hi[c] = j < half ? LoadScalar(x + j + half) : 0.0f;
        sum_sq += (x_lo[c] * x_lo[c]) + (x_hi[c] * x_hi[c]);
      }
      const float rstd =
          rsqrtf((RmsWarpSum(sum_sq) / static_cast<float>(width)) + eps);
      float dot = 0.0f;
      for (int c = 0; c < kChunks; c++) {
        x_lo[c] *= rstd;
        x_hi[c] *= rstd;
        dot += (lo[c] * g_lo[c] * x_lo[c]) + (hi[c] * g_hi[c] * x_hi[c]);
        dg_lo[c] += lo[c] * x_lo[c];
        dg_hi[c] += hi[c] * x_hi[c];
      }
      const float mean_dot = RmsWarpSum(dot) / static_cast<float>(width);
      for (int c = 0; c < kChunks; c++) {
        lo[c] = rstd * ((lo[c] * g_lo[c]) - (x_lo[c] * mean_dot));
        hi[c] = rstd * ((hi[c] * g_hi[c]) - (x_hi[c] * mean_dot));
      }
    }
    T *dx = in_grad + (row * width);
    for (int c = 0; c < kChunks; c++) {
      const int j = lane + (c * kRmsWarpSize);
      if (j < half) {
        const float old_lo = accumulate ? LoadScalar(dx + j) : 0.0f;
        const float old_hi = accumulate ? LoadScalar(dx + j + half) : 0.0f;
        StoreScalar(dx + j, old_lo + lo[c]);
        StoreScalar(dx + j + half, old_hi + hi[c]);
      }
    }
  }
  if (gain == nullptr) {
    return;
  }
  float *warp_dg = head_smem + (static_cast<size_t>(warp) * width);
  for (int c = 0; c < kChunks; c++) {
    const int j = lane + (c * kRmsWarpSize);
    if (j < half) {
      warp_dg[j] = dg_lo[c];
      warp_dg[j + half] = dg_hi[c];
    }
  }
  __syncthreads();
  for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(width);
       i += static_cast<int>(blockDim.x)) {
    float total = 0.0f;
    for (int k = 0; k < kRmsWarps; k++) {
      total += head_smem[(k * width) + i];
    }
    partial[(static_cast<size_t>(blockIdx.x) * width) + i] = total;
  }
}

struct HeadRows {
  size_t rows = 0;
  int half = 0;
  int n_head = 1;
  int seq_len = 1;
  int chunks = 1;
};

HeadRows HeadRowsFor(const Tensor &input, int64_t n_head) {
  const std::vector<size_t> &shape = input.shape();
  const size_t head_dim = shape.back() / static_cast<size_t>(n_head);
  HeadRows layout;
  layout.rows = input.size() / head_dim;
  layout.half = static_cast<int>(head_dim / 2);
  layout.n_head = static_cast<int>(n_head);
  layout.seq_len = static_cast<int>(shape[shape.size() - 2]);
  layout.chunks = (layout.half + kRmsWarpSize - 1) / kRmsWarpSize;
  if (head_dim % 2 != 0 || layout.chunks > kHeadMaxChunks) {
    throw std::runtime_error(
        "head_norm_rotary on CUDA needs an even head_dim of at most 256");
  }
  return layout;
}

const float *OptionalData(const Tensor *t) {
  return t == nullptr ? nullptr : DataPtr(t);
}

template <typename T>
void LaunchHeadNormRotary(const T *in, const float *gain,
                          const float *cos_table, const float *sin_table,
                          T *out, const HeadRows &layout, float eps) {
  const auto blocks =
      static_cast<unsigned int>((layout.rows + kRmsWarps - 1) / kRmsWarps);
  cudaStream_t stream = CudaContext::instance().stream();
  const auto launch = [&](auto kernel) {
    kernel<<<blocks, kRmsWarps * kRmsWarpSize, 0, stream>>>(
        in, gain, cos_table, sin_table, out, layout.rows, layout.half,
        layout.n_head, layout.seq_len, eps);
  };
  switch (layout.chunks) {
    case 1:
      launch(HeadNormRotaryKernel<T, 1>);
      break;
    case 2:
      launch(HeadNormRotaryKernel<T, 2>);
      break;
    case 3:
      launch(HeadNormRotaryKernel<T, 3>);
      break;
    default:
      launch(HeadNormRotaryKernel<T, kHeadMaxChunks>);
      break;
  }
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void LaunchHeadNormRotaryBackward(const T *in, Tensor *gain,
                                  const float *cos_table,
                                  const float *sin_table, const T *out_grad,
                                  T *in_grad, bool accumulate,
                                  const HeadRows &layout, float eps) {
  if (layout.rows == 0) {
    return;
  }
  auto &ctx = CudaContext::instance();
  cudaStream_t stream = ctx.stream();
  const auto row_blocks =
      static_cast<unsigned int>((layout.rows + kRmsWarps - 1) / kRmsWarps);
  const unsigned int blocks = gain != nullptr
                                  ? std::min(row_blocks, kRmsBackwardMaxBlocks)
                                  : row_blocks;
  const size_t width = 2 * static_cast<size_t>(layout.half);
  const size_t smem = gain != nullptr ? kRmsWarps * width * sizeof(float) : 0;
  const size_t partial_bytes =
      gain != nullptr ? blocks * width * sizeof(float) : 0;
  float *partial = gain != nullptr
                       ? static_cast<float *>(ctx.allocate(partial_bytes))
                       : nullptr;
  const float *gain_data = gain != nullptr ? DataPtr(gain) : nullptr;
  const auto launch = [&](auto kernel) {
    kernel<<<blocks, kRmsWarps * kRmsWarpSize, smem, stream>>>(
        in, gain_data, cos_table, sin_table, out_grad, in_grad, accumulate,
        partial, layout.rows, layout.half, layout.n_head, layout.seq_len, eps);
  };
  switch (layout.chunks) {
    case 1:
      launch(HeadNormRotaryBackwardKernel<T, 1>);
      break;
    case 2:
      launch(HeadNormRotaryBackwardKernel<T, 2>);
      break;
    case 3:
      launch(HeadNormRotaryBackwardKernel<T, 3>);
      break;
    default:
      launch(HeadNormRotaryBackwardKernel<T, kHeadMaxChunks>);
      break;
  }
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  if (gain != nullptr) {
    RmsNormGainGradKernel<<<static_cast<unsigned int>(width), kBlockSize, 0,
                            stream>>>(partial, blocks, static_cast<int>(width),
                                      GradPtr(gain));
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
    ctx.deallocate(partial, partial_bytes);
  }
}

void HeadNormRotary(const OpArgs &args) {
  LaunchHeadNormRotary(DataPtr(args.lhs), OptionalData(args.rhs),
                       OptionalData(args.extra), OptionalData(args.aux),
                       DataPtr(args.out), HeadRowsFor(*args.lhs, args.dim),
                       args.scalar);
}

void HeadNormRotaryBf16(const OpArgs &args) {
  LaunchHeadNormRotary(Bf16Data(args.lhs), OptionalData(args.rhs),
                       OptionalData(args.extra), OptionalData(args.aux),
                       Bf16Data(args.out), HeadRowsFor(*args.lhs, args.dim),
                       args.scalar);
}

std::function<void()> HeadNormRotaryBackward(const GradArgs &args) {
  return [lhs = args.lhs, gain = args.rhs, cos_table = args.extra,
          sin_table = args.aux, out = args.out, eps = args.scalar,
          n_head = args.dim]() {
    const HeadRows layout = HeadRowsFor(*lhs, n_head);
    const GradTarget dx = GradForWrite(lhs.get());
    LaunchHeadNormRotaryBackward(
        static_cast<const scalar_t *>(DataPtr(lhs.get())), gain.get(),
        OptionalData(cos_table.get()), OptionalData(sin_table.get()),
        static_cast<const scalar_t *>(GradPtr(out)), dx.ptr, dx.accumulate,
        layout, eps);
  };
}

std::function<void()> HeadNormRotaryBackwardBf16(const GradArgs &args) {
  return [lhs = args.lhs, gain = args.rhs, cos_table = args.extra,
          sin_table = args.aux, out = args.out, eps = args.scalar,
          n_head = args.dim]() {
    const HeadRows layout = HeadRowsFor(*lhs, n_head);
    const Bf16GradTarget dx = Bf16GradForWrite(lhs.get());
    LaunchHeadNormRotaryBackward(static_cast<const bf16 *>(Bf16Data(lhs.get())),
                                 gain.get(), OptionalData(cos_table.get()),
                                 OptionalData(sin_table.get()),
                                 static_cast<const bf16 *>(Bf16Grad(out)),
                                 dx.ptr, dx.accumulate, layout, eps);
  };
}

}  // namespace

void RegisterRmsNormOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kRmsNorm, Device::CUDA, RmsNorm);
  registry.RegisterBackward(OpId::kRmsNorm, Device::CUDA, RmsNormBackward);
  registry.Register(OpId::kRmsNorm, Device::CUDA, RmsNormBf16,
                    DType::kBFloat16);
  registry.RegisterBackward(OpId::kRmsNorm, Device::CUDA, RmsNormBackwardBf16,
                            DType::kBFloat16);
  registry.Register(OpId::kHeadNormRotary, Device::CUDA, HeadNormRotary);
  registry.RegisterBackward(OpId::kHeadNormRotary, Device::CUDA,
                            HeadNormRotaryBackward);
  registry.Register(OpId::kHeadNormRotary, Device::CUDA, HeadNormRotaryBf16,
                    DType::kBFloat16);
  registry.RegisterBackward(OpId::kHeadNormRotary, Device::CUDA,
                            HeadNormRotaryBackwardBf16, DType::kBFloat16);
}

}  // namespace micrograd::cuda::ops

#endif
