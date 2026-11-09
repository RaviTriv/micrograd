#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <algorithm>
#include <cstddef>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cuda/ops/Ops.h"

namespace micrograd::cuda::ops {
namespace {

constexpr int kBlockSize = 256;
constexpr size_t kMaxBlocks = 65535;

int GridSize(size_t n) {
  size_t blocks = (n + kBlockSize - 1) / kBlockSize;
  return static_cast<int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

__global__ void GradNormSquaredKernel(const scalar_t *grad, scalar_t *out,
                                      size_t n) {
  __shared__ scalar_t shared[kBlockSize];
  size_t tid = threadIdx.x;

  scalar_t local = 0.0f;
  for (size_t i = blockIdx.x * blockDim.x + tid; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    scalar_t g = grad[i];
    local += g * g;
  }
  shared[tid] = local;
  __syncthreads();
  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(out, shared[0]);
  }
}

__global__ void ScaleGradKernel(scalar_t *grad, scalar_t scale, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    grad[i] *= scale;
  }
}

__global__ void AdamWStepKernel(scalar_t *data, const scalar_t *grad,
                                scalar_t *m, scalar_t *v, size_t n, scalar_t lr,
                                scalar_t beta1, scalar_t beta2, scalar_t eps,
                                scalar_t weight_decay, bool decay,
                                scalar_t bias_correction1,
                                scalar_t bias_correction2) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    if (decay) {
      data[i] -= lr * weight_decay * data[i];
    }
    scalar_t g = grad[i];
    m[i] = (beta1 * m[i]) + ((1.0f - beta1) * g);
    v[i] = (beta2 * v[i]) + ((1.0f - beta2) * g * g);
    scalar_t m_hat = m[i] / bias_correction1;
    scalar_t v_hat = v[i] / bias_correction2;
    data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
  }
}

__global__ void FusedAdamWStepKernel(const AdamWTensor *tensors, scalar_t lr,
                                     scalar_t beta1, scalar_t beta2,
                                     scalar_t eps, scalar_t weight_decay,
                                     scalar_t bias_correction1,
                                     scalar_t bias_correction2) {
  const AdamWTensor &t = tensors[blockIdx.y];
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < t.n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    if (t.decay) {
      t.data[i] -= lr * weight_decay * t.data[i];
    }
    scalar_t g = t.grad[i];
    t.m[i] = (beta1 * t.m[i]) + ((1.0f - beta1) * g);
    t.v[i] = (beta2 * t.v[i]) + ((1.0f - beta2) * g * g);
    scalar_t m_hat = t.m[i] / bias_correction1;
    scalar_t v_hat = t.v[i] / bias_correction2;
    t.data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
  }
}

scalar_t *DataPtr(Tensor *t) {
  return static_cast<scalar_t *>(t->data_storage().device_pointer());
}

scalar_t *GradPtr(Tensor *t) {
  return static_cast<scalar_t *>(t->grad_storage().device_pointer());
}

const scalar_t *GradPtr(const Tensor *t) {
  return static_cast<const scalar_t *>(t->grad_storage().device_pointer());
}

bool SameEntry(const AdamWTensor &a, const AdamWTensor &b) {
  return a.data == b.data && a.grad == b.grad && a.m == b.m && a.v == b.v &&
         a.n == b.n && a.decay == b.decay;
}

const AdamWTensor *UploadAdamWTable(const std::vector<AdamWTensor> &tensors) {
  static std::vector<AdamWTensor> cached;
  static AdamWTensor *device_table = nullptr;
  static size_t device_bytes = 0;

  if (device_table != nullptr && cached.size() == tensors.size() &&
      std::equal(cached.begin(), cached.end(), tensors.begin(), SameEntry)) {
    return device_table;
  }

  auto &ctx = CudaContext::instance();
  if (device_table != nullptr) {
    ctx.deallocate(device_table, device_bytes);
  }
  device_bytes = tensors.size() * sizeof(AdamWTensor);
  device_table = static_cast<AdamWTensor *>(ctx.allocate(device_bytes));
  cached = tensors;
  MICROGRAD_CUDA_CHECK(cudaMemcpyAsync(device_table, cached.data(),
                                       device_bytes, cudaMemcpyHostToDevice,
                                       ctx.stream()));
  return device_table;
}

}  // namespace

scalar_t GradNormSquared(const Tensor &param) {
  size_t n = param.size();
  auto &ctx = CudaContext::instance();
  auto *scratch = static_cast<scalar_t *>(ctx.allocate(sizeof(scalar_t)));
  MICROGRAD_CUDA_CHECK(
      cudaMemsetAsync(scratch, 0, sizeof(scalar_t), ctx.stream()));
  GradNormSquaredKernel<<<GridSize(n), kBlockSize, 0, ctx.stream()>>>(
      GradPtr(&param), scratch, n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  scalar_t result = 0.0f;
  MICROGRAD_CUDA_CHECK(cudaMemcpyAsync(&result, scratch, sizeof(scalar_t),
                                       cudaMemcpyDeviceToHost, ctx.stream()));
  ctx.synchronize();
  ctx.deallocate(scratch, sizeof(scalar_t));
  return result;
}

scalar_t GradNormSquaredSum(const std::vector<const Tensor *> &params) {
  if (params.empty()) {
    return 0.0f;
  }
  auto &ctx = CudaContext::instance();
  auto *scratch = static_cast<scalar_t *>(ctx.allocate(sizeof(scalar_t)));
  MICROGRAD_CUDA_CHECK(
      cudaMemsetAsync(scratch, 0, sizeof(scalar_t), ctx.stream()));
  for (const Tensor *param : params) {
    size_t n = param->size();
    GradNormSquaredKernel<<<GridSize(n), kBlockSize, 0, ctx.stream()>>>(
        GradPtr(param), scratch, n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  }
  scalar_t result = 0.0f;
  MICROGRAD_CUDA_CHECK(cudaMemcpyAsync(&result, scratch, sizeof(scalar_t),
                                       cudaMemcpyDeviceToHost, ctx.stream()));
  ctx.synchronize();
  ctx.deallocate(scratch, sizeof(scalar_t));
  return result;
}

void ScaleGrad(Tensor &param, scalar_t scale) {
  size_t n = param.size();
  ScaleGradKernel<<<GridSize(n), kBlockSize, 0,
                    CudaContext::instance().stream()>>>(GradPtr(&param), scale,
                                                        n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void AdamWStep(Tensor &param, Tensor &m, Tensor &v, scalar_t lr, scalar_t beta1,
               scalar_t beta2, scalar_t eps, scalar_t weight_decay, bool decay,
               scalar_t bias_correction1, scalar_t bias_correction2) {
  size_t n = param.size();
  AdamWStepKernel<<<GridSize(n), kBlockSize, 0,
                    CudaContext::instance().stream()>>>(
      DataPtr(&param), GradPtr(&param), DataPtr(&m), DataPtr(&v), n, lr, beta1,
      beta2, eps, weight_decay, decay, bias_correction1, bias_correction2);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void FusedAdamWStep(const std::vector<AdamWTensor> &tensors, scalar_t lr,
                    scalar_t beta1, scalar_t beta2, scalar_t eps,
                    scalar_t weight_decay, scalar_t bias_correction1,
                    scalar_t bias_correction2) {
  if (tensors.empty()) {
    return;
  }
  size_t max_n = 0;
  for (const auto &t : tensors) {
    max_n = std::max(max_n, t.n);
  }

  auto &ctx = CudaContext::instance();
  const AdamWTensor *device_tensors = UploadAdamWTable(tensors);

  dim3 grid(static_cast<unsigned>(GridSize(max_n)),
            static_cast<unsigned>(tensors.size()));
  FusedAdamWStepKernel<<<grid, kBlockSize, 0, ctx.stream()>>>(
      device_tensors, lr, beta1, beta2, eps, weight_decay, bias_correction1,
      bias_correction2);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace micrograd::cuda::ops

#endif
