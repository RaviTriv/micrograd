#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cuda/ops/Bf16.h"
#include "micrograd/backends/cuda/ops/Grad.h"
#include "micrograd/backends/cuda/ops/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::cuda::ops {
namespace {

constexpr int kBlockSize = 256;

__global__ void LayerNormKernel(const scalar_t *in, const scalar_t *gain,
                                const scalar_t *bias, scalar_t *out, size_t n,
                                scalar_t eps) {
  __shared__ scalar_t shared[kBlockSize];
  size_t row = blockIdx.x;
  size_t tid = threadIdx.x;
  const scalar_t *row_in = in + (row * n);

  scalar_t local_sum = 0.0f;
  for (size_t i = tid; i < n; i += blockDim.x) {
    local_sum += row_in[i];
  }
  shared[tid] = local_sum;
  __syncthreads();
  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  scalar_t mean = shared[0] / static_cast<scalar_t>(n);
  __syncthreads();

  scalar_t local_var = 0.0f;
  for (size_t i = tid; i < n; i += blockDim.x) {
    scalar_t centered = row_in[i] - mean;
    local_var += centered * centered;
  }
  shared[tid] = local_var;
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
    scalar_t normalized = (row_in[i] - mean) * rstd;
    row_out[i] = (normalized * gain[i]) + bias[i];
  }
}

__global__ void LayerNormBackwardKernel(const scalar_t *in,
                                        const scalar_t *gain,
                                        const scalar_t *out_grad,
                                        scalar_t *in_grad, scalar_t *gain_grad,
                                        scalar_t *bias_grad, bool accumulate,
                                        size_t n, scalar_t eps) {
  __shared__ scalar_t shared[kBlockSize];
  size_t row = blockIdx.x;
  size_t tid = threadIdx.x;
  const scalar_t *row_in = in + (row * n);
  const scalar_t *row_out_grad = out_grad + (row * n);

  scalar_t local_sum = 0.0f;
  for (size_t i = tid; i < n; i += blockDim.x) {
    local_sum += row_in[i];
  }
  shared[tid] = local_sum;
  __syncthreads();
  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  scalar_t mean = shared[0] / static_cast<scalar_t>(n);
  __syncthreads();

  scalar_t local_var = 0.0f;
  for (size_t i = tid; i < n; i += blockDim.x) {
    scalar_t centered = row_in[i] - mean;
    local_var += centered * centered;
  }
  shared[tid] = local_var;
  __syncthreads();
  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  scalar_t rstd = 1.0f / sqrtf((shared[0] / static_cast<scalar_t>(n)) + eps);
  __syncthreads();

  scalar_t local_mean_dxhat = 0.0f;
  scalar_t local_mean_dxhat_xhat = 0.0f;
  for (size_t i = tid; i < n; i += blockDim.x) {
    scalar_t xhat_i = (row_in[i] - mean) * rstd;
    scalar_t dxhat_i = row_out_grad[i] * gain[i];
    local_mean_dxhat += dxhat_i;
    local_mean_dxhat_xhat += dxhat_i * xhat_i;
    atomicAdd(&gain_grad[i], row_out_grad[i] * xhat_i);
    atomicAdd(&bias_grad[i], row_out_grad[i]);
  }
  shared[tid] = local_mean_dxhat;
  __syncthreads();
  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  scalar_t mean_dxhat = shared[0] / static_cast<scalar_t>(n);
  __syncthreads();

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
    scalar_t xhat_i = (row_in[i] - mean) * rstd;
    scalar_t dxhat_i = row_out_grad[i] * gain[i];
    const scalar_t g =
        rstd * (dxhat_i - mean_dxhat - (xhat_i * mean_dxhat_xhat));
    row_in_grad[i] = accumulate ? row_in_grad[i] + g : g;
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

void LayerNorm(const OpArgs &args) {
  size_t n = args.rhs->size();
  size_t outer = args.lhs->size() / n;
  LayerNormKernel<<<static_cast<int>(outer), kBlockSize, 0,
                    CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.extra),
      DataPtr(args.out), n, args.scalar);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

std::function<void()> LayerNormBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, gain = args.rhs, bias = args.extra,
          eps = args.scalar]() {
    size_t n = gain->size();
    size_t outer = lhs->size() / n;
    const GradTarget in_grad = GradForWrite(lhs.get());
    LayerNormBackwardKernel<<<static_cast<int>(outer), kBlockSize, 0,
                              CudaContext::instance().stream()>>>(
        DataPtr(lhs.get()), DataPtr(gain.get()), GradPtr(out), in_grad.ptr,
        GradPtr(gain.get()), GradPtr(bias.get()), in_grad.accumulate, n, eps);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

constexpr int kLnWarpSize = 32;
constexpr int kLnWarps = 8;
constexpr int kLnBackwardMaxBlocks = 128;
constexpr size_t kLnMaxSmem = 48 * 1024;

template <int V>
__device__ void LoadVec(const bf16 *ptr, float (&values)[V]) {
  if constexpr (V == kBf16Vector) {
    const Bf16x8 x = LoadBf16x8(ptr);
    for (int j = 0; j < V; j++) {
      values[j] = x.v[j];
    }
  } else {
    for (int j = 0; j < V; j++) {
      values[j] = __bfloat162float(ptr[j]);
    }
  }
}

template <int V>
__device__ void StoreVec(bf16 *ptr, const float (&values)[V]) {
  if constexpr (V == kBf16Vector) {
    Bf16x8 x;
    for (int j = 0; j < V; j++) {
      x.v[j] = values[j];
    }
    StoreBf16x8(ptr, x);
  } else {
    for (int j = 0; j < V; j++) {
      ptr[j] = __float2bfloat16(values[j]);
    }
  }
}

__device__ float LnWarpSum(float value) {
  for (int offset = kLnWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffU, value, offset);
  }
  return value;
}

template <int V>
__device__ float2 RowStats(const bf16 *x, int n, int lane, float eps) {
  float sum = 0.0f;
  for (int i = lane * V; i < n; i += kLnWarpSize * V) {
    float v[V];
    LoadVec<V>(x + i, v);
    for (int j = 0; j < V; j++) {
      sum += v[j];
    }
  }
  const float mean = LnWarpSum(sum) / static_cast<float>(n);
  float var = 0.0f;
  for (int i = lane * V; i < n; i += kLnWarpSize * V) {
    float v[V];
    LoadVec<V>(x + i, v);
    for (int j = 0; j < V; j++) {
      const float c = v[j] - mean;
      var += c * c;
    }
  }
  const float rstd = rsqrtf((LnWarpSum(var) / static_cast<float>(n)) + eps);
  return make_float2(mean, rstd);
}

template <int V>
__global__ void LayerNormBf16Kernel(const bf16 *in, const float *gain,
                                    const float *bias, bf16 *out, size_t rows,
                                    int n, float eps) {
  const int lane = static_cast<int>(threadIdx.x) % kLnWarpSize;
  const size_t row = (static_cast<size_t>(blockIdx.x) * kLnWarps) +
                     (threadIdx.x / kLnWarpSize);
  if (row >= rows) {
    return;
  }
  const bf16 *x = in + (row * n);
  const float2 stats = RowStats<V>(x, n, lane, eps);
  bf16 *y = out + (row * n);
  for (int i = lane * V; i < n; i += kLnWarpSize * V) {
    float v[V];
    LoadVec<V>(x + i, v);
    for (int j = 0; j < V; j++) {
      v[j] = ((v[j] - stats.x) * stats.y * gain[i + j]) + bias[i + j];
    }
    StoreVec<V>(y + i, v);
  }
}

template <int V>
__global__ void LayerNormBackwardBf16Kernel(
    const bf16 *in, const float *gain, const bf16 *out_grad, bf16 *in_grad,
    bool accumulate, float *gain_grad, float *bias_grad, size_t rows, int n,
    float eps) {
  extern __shared__ float ln_smem[];
  float *dgain = ln_smem;
  float *dbias = ln_smem + n;
  for (int i = static_cast<int>(threadIdx.x); i < 2 * n;
       i += static_cast<int>(blockDim.x)) {
    ln_smem[i] = 0.0f;
  }
  __syncthreads();

  const int lane = static_cast<int>(threadIdx.x) % kLnWarpSize;
  const size_t warp = threadIdx.x / kLnWarpSize;
  const size_t step = static_cast<size_t>(gridDim.x) * kLnWarps;
  for (size_t row = (static_cast<size_t>(blockIdx.x) * kLnWarps) + warp;
       row < rows; row += step) {
    const bf16 *x = in + (row * n);
    const bf16 *dy = out_grad + (row * n);
    const float2 stats = RowStats<V>(x, n, lane, eps);
    float s1 = 0.0f;
    float s2 = 0.0f;
    for (int i = lane * V; i < n; i += kLnWarpSize * V) {
      float v[V];
      float g[V];
      LoadVec<V>(x + i, v);
      LoadVec<V>(dy + i, g);
      for (int j = 0; j < V; j++) {
        const float xhat = (v[j] - stats.x) * stats.y;
        const float dxhat = g[j] * gain[i + j];
        s1 += dxhat;
        s2 += dxhat * xhat;
        atomicAdd(&dgain[i + j], g[j] * xhat);
        atomicAdd(&dbias[i + j], g[j]);
      }
    }
    const float mean_dxhat = LnWarpSum(s1) / static_cast<float>(n);
    const float mean_dxhat_xhat = LnWarpSum(s2) / static_cast<float>(n);
    bf16 *dx = in_grad + (row * n);
    for (int i = lane * V; i < n; i += kLnWarpSize * V) {
      float v[V];
      float g[V];
      float old[V];
      LoadVec<V>(x + i, v);
      LoadVec<V>(dy + i, g);
      if (accumulate) {
        LoadVec<V>(dx + i, old);
      }
      for (int j = 0; j < V; j++) {
        const float xhat = (v[j] - stats.x) * stats.y;
        const float dxhat = g[j] * gain[i + j];
        const float r =
            stats.y * (dxhat - mean_dxhat - (xhat * mean_dxhat_xhat));
        v[j] = accumulate ? old[j] + r : r;
      }
      StoreVec<V>(dx + i, v);
    }
  }
  __syncthreads();
  for (int i = static_cast<int>(threadIdx.x); i < n;
       i += static_cast<int>(blockDim.x)) {
    atomicAdd(&gain_grad[i], dgain[i]);
    atomicAdd(&bias_grad[i], dbias[i]);
  }
}

constexpr int kLnMaxChunks = 4;

template <int kChunks>
__global__ void LayerNormBackwardBf16RegKernel(
    const bf16 *in, const float *gain, const bf16 *out_grad, bf16 *in_grad,
    bool accumulate, float *partial, size_t rows, int n, float eps) {
  constexpr int V = kBf16Vector;
  const int lane = static_cast<int>(threadIdx.x) % kLnWarpSize;
  const size_t warp = threadIdx.x / kLnWarpSize;
  const size_t step = static_cast<size_t>(gridDim.x) * kLnWarps;
  float dg[kChunks][V] = {};
  float db[kChunks][V] = {};
  float w[kChunks][V];
  for (int c = 0; c < kChunks; c++) {
    const int i = (lane + (c * kLnWarpSize)) * V;
    for (int j = 0; j < V; j++) {
      w[c][j] = i < n ? gain[i + j] : 0.0f;
    }
  }
  for (size_t row = (static_cast<size_t>(blockIdx.x) * kLnWarps) + warp;
       row < rows; row += step) {
    const bf16 *x = in + (row * n);
    const bf16 *dy = out_grad + (row * n);
    float xv[kChunks][V];
    float gv[kChunks][V];
    float sum = 0.0f;
    for (int c = 0; c < kChunks; c++) {
      const int i = (lane + (c * kLnWarpSize)) * V;
      if (i < n) {
        LoadVec<V>(x + i, xv[c]);
        LoadVec<V>(dy + i, gv[c]);
      } else {
        for (int j = 0; j < V; j++) {
          xv[c][j] = 0.0f;
          gv[c][j] = 0.0f;
        }
      }
      for (int j = 0; j < V; j++) {
        sum += xv[c][j];
      }
    }
    const float mean = LnWarpSum(sum) / static_cast<float>(n);
    float var = 0.0f;
    for (int c = 0; c < kChunks; c++) {
      const int i = (lane + (c * kLnWarpSize)) * V;
      for (int j = 0; j < V && i < n; j++) {
        const float centered = xv[c][j] - mean;
        var += centered * centered;
      }
    }
    const float rstd = rsqrtf((LnWarpSum(var) / static_cast<float>(n)) + eps);
    float s1 = 0.0f;
    float s2 = 0.0f;
    for (int c = 0; c < kChunks; c++) {
      for (int j = 0; j < V; j++) {
        const float xhat = (xv[c][j] - mean) * rstd;
        const float dxhat = gv[c][j] * w[c][j];
        s1 += dxhat;
        s2 += dxhat * xhat;
        dg[c][j] += gv[c][j] * xhat;
        db[c][j] += gv[c][j];
        xv[c][j] = xhat;
      }
    }
    const float mean_dxhat = LnWarpSum(s1) / static_cast<float>(n);
    const float mean_dxhat_xhat = LnWarpSum(s2) / static_cast<float>(n);
    bf16 *dx = in_grad + (row * n);
    for (int c = 0; c < kChunks; c++) {
      const int i = (lane + (c * kLnWarpSize)) * V;
      if (i >= n) {
        continue;
      }
      float old[V];
      if (accumulate) {
        LoadVec<V>(dx + i, old);
      }
      float r[V];
      for (int j = 0; j < V; j++) {
        const float value =
            rstd * ((gv[c][j] * w[c][j]) - mean_dxhat -
                    (xv[c][j] * mean_dxhat_xhat));
        r[j] = accumulate ? old[j] + value : value;
      }
      StoreVec<V>(dx + i, r);
    }
  }
  float *warp_partial =
      partial +
      (((static_cast<size_t>(blockIdx.x) * kLnWarps) + warp) * 2 * n);
  for (int c = 0; c < kChunks; c++) {
    const int i = (lane + (c * kLnWarpSize)) * V;
    if (i >= n) {
      continue;
    }
    for (int j = 0; j < V; j++) {
      warp_partial[i + j] = dg[c][j];
      warp_partial[n + i + j] = db[c][j];
    }
  }
}

__global__ void LayerNormParamGradKernel(const float *partial, int blocks,
                                         int n, float *gain_grad,
                                         float *bias_grad) {
  const int i = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (i >= 2 * n) {
    return;
  }
  float total = 0.0f;
  for (int b = 0; b < blocks; b++) {
    total += partial[(static_cast<size_t>(b) * 2 * n) + i];
  }
  float *target = i < n ? gain_grad + i : bias_grad + (i - n);
  *target += total;
}

unsigned int LnForwardBlocks(size_t rows) {
  return static_cast<unsigned int>((rows + kLnWarps - 1) / kLnWarps);
}

void LayerNormBf16(const OpArgs &args) {
  const size_t n = args.rhs->size();
  const size_t rows = args.lhs->size() / n;
  if (rows == 0) {
    return;
  }
  const unsigned int blocks = LnForwardBlocks(rows);
  cudaStream_t stream = CudaContext::instance().stream();
  const int width = static_cast<int>(n);
  if (n % kBf16Vector == 0) {
    LayerNormBf16Kernel<kBf16Vector>
        <<<blocks, kLnWarps * kLnWarpSize, 0, stream>>>(
            Bf16Data(args.lhs), DataPtr(args.rhs), DataPtr(args.extra),
            Bf16Data(args.out), rows, width, args.scalar);
  } else {
    LayerNormBf16Kernel<1><<<blocks, kLnWarps * kLnWarpSize, 0, stream>>>(
        Bf16Data(args.lhs), DataPtr(args.rhs), DataPtr(args.extra),
        Bf16Data(args.out), rows, width, args.scalar);
  }
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

std::function<void()> LayerNormBackwardBf16(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, gain = args.rhs, bias = args.extra,
          eps = args.scalar]() {
    const size_t n = gain->size();
    const size_t rows = lhs->size() / n;
    const size_t smem = 2 * n * sizeof(float);
    if (smem > kLnMaxSmem) {
      throw std::runtime_error("bf16 layer_norm backward: row is too wide");
    }
    const Bf16GradTarget dx = Bf16GradForWrite(lhs.get());
    const unsigned int blocks = std::min<unsigned int>(
        LnForwardBlocks(rows), kLnBackwardMaxBlocks);
    cudaStream_t stream = CudaContext::instance().stream();
    const int width = static_cast<int>(n);
    const size_t chunks = (n + (kLnWarpSize * kBf16Vector) - 1) /
                          (kLnWarpSize * kBf16Vector);
    if (n % kBf16Vector == 0 && chunks <= kLnMaxChunks) {
      const size_t partial_bytes = blocks * kLnWarps * smem;
      auto *partial = static_cast<float *>(
          CudaContext::instance().allocate(partial_bytes));
      const auto launch = [&](auto kernel) {
        kernel<<<blocks, kLnWarps * kLnWarpSize, 0, stream>>>(
            Bf16Data(lhs.get()), DataPtr(gain.get()), Bf16Grad(out), dx.ptr,
            dx.accumulate, partial, rows, width, eps);
      };
      switch (chunks) {
        case 1:
          launch(LayerNormBackwardBf16RegKernel<1>);
          break;
        case 2:
          launch(LayerNormBackwardBf16RegKernel<2>);
          break;
        case 3:
          launch(LayerNormBackwardBf16RegKernel<3>);
          break;
        default:
          launch(LayerNormBackwardBf16RegKernel<4>);
          break;
      }
      MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
      const int params = 2 * width;
      LayerNormParamGradKernel<<<(params + 255) / 256, 256, 0, stream>>>(
          partial, static_cast<int>(blocks * kLnWarps), width,
          GradPtr(gain.get()),
          GradPtr(bias.get()));
      CudaContext::instance().deallocate(partial, partial_bytes);
    } else if (n % kBf16Vector == 0) {
      LayerNormBackwardBf16Kernel<kBf16Vector>
          <<<blocks, kLnWarps * kLnWarpSize, smem, stream>>>(
              Bf16Data(lhs.get()), DataPtr(gain.get()), Bf16Grad(out), dx.ptr,
              dx.accumulate, GradPtr(gain.get()), GradPtr(bias.get()), rows,
              width, eps);
    } else {
      LayerNormBackwardBf16Kernel<1>
          <<<blocks, kLnWarps * kLnWarpSize, smem, stream>>>(
              Bf16Data(lhs.get()), DataPtr(gain.get()), Bf16Grad(out), dx.ptr,
              dx.accumulate, GradPtr(gain.get()), GradPtr(bias.get()), rows,
              width, eps);
    }
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

}  // namespace

void RegisterLayerNormOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kLayerNorm, Device::CUDA, LayerNormBf16,
                    DType::kBFloat16);
  registry.RegisterBackward(OpId::kLayerNorm, Device::CUDA,
                            LayerNormBackwardBf16, DType::kBFloat16);
  registry.Register(OpId::kLayerNorm, Device::CUDA, LayerNorm);
  registry.RegisterBackward(OpId::kLayerNorm, Device::CUDA, LayerNormBackward);
}

}  // namespace micrograd::cuda::ops

#endif
