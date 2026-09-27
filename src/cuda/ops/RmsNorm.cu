#include "micrograd/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <cstddef>
#include <functional>

#include "micrograd/Tensor.h"
#include "micrograd/cuda/ops/Ops.h"
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
  args.out->to(Backend::CUDA);
  size_t n = args.rhs->size();
  size_t outer = args.lhs->size() / n;
  RmsNormKernel<<<static_cast<int>(outer), kBlockSize, 0,
                  CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), n, args.scalar);
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
      };
}

}  // namespace

void RegisterRmsNormOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kRmsNorm, Device::CUDA, RmsNorm);
  registry.RegisterBackward(OpId::kRmsNorm, Device::CUDA, RmsNormBackward);
}

}  // namespace micrograd::cuda::ops

#endif
