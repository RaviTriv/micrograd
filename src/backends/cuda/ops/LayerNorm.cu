#include "micrograd/backends/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <cstddef>
#include <functional>

#include "micrograd/Tensor.h"
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
                                        scalar_t *bias_grad, size_t n,
                                        scalar_t eps) {
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
    row_in_grad[i] +=
        rstd * (dxhat_i - mean_dxhat - (xhat_i * mean_dxhat_xhat));
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
  args.out->to(Backend::CUDA);
  size_t n = args.rhs->size();
  size_t outer = args.lhs->size() / n;
  LayerNormKernel<<<static_cast<int>(outer), kBlockSize, 0,
                    CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.extra),
      DataPtr(args.out), n, args.scalar);
}

std::function<void()> LayerNormBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, gain = args.rhs, bias = args.extra,
          eps = args.scalar]() {
    size_t n = gain->size();
    size_t outer = lhs->size() / n;
    LayerNormBackwardKernel<<<static_cast<int>(outer), kBlockSize, 0,
                              CudaContext::instance().stream()>>>(
        DataPtr(lhs.get()), DataPtr(gain.get()), GradPtr(out),
        GradPtr(lhs.get()), GradPtr(gain.get()), GradPtr(bias.get()), n, eps);
  };
}

}  // namespace

void RegisterLayerNormOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kLayerNorm, Device::CUDA, LayerNorm);
  registry.RegisterBackward(OpId::kLayerNorm, Device::CUDA, LayerNormBackward);
}

}  // namespace micrograd::cuda::ops

#endif
