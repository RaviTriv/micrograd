#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "micrograd/Storage.h"
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
                                     scalar_t bias_correction2,
                                     const scalar_t *clip_total,
                                     scalar_t max_norm) {
  const AdamWTensor &t = tensors[blockIdx.y];
  const bool clip = clip_total != nullptr;
  const scalar_t scale =
      clip ? fminf(1.0f, max_norm / (sqrtf(*clip_total) + 1e-6f)) : 1.0f;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < t.n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    if (t.decay) {
      t.data[i] -= lr * weight_decay * t.data[i];
    }
    scalar_t g = clip ? t.grad[i] * scale : t.grad[i];
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

template <typename Entry>
struct DeviceTable {
  std::vector<Entry> cached;
  Entry *device = nullptr;
  size_t bytes = 0;
};

template <typename Entry, typename Equal>
const Entry *UploadTable(DeviceTable<Entry> &table,
                         const std::vector<Entry> &entries, Equal equal) {
  if (table.device != nullptr && table.cached.size() == entries.size() &&
      std::equal(table.cached.begin(), table.cached.end(), entries.begin(),
                 equal)) {
    return table.device;
  }

  auto &ctx = CudaContext::instance();
  if (table.device != nullptr) {
    ctx.deallocate(table.device, table.bytes);
  }
  table.bytes = entries.size() * sizeof(Entry);
  table.device = static_cast<Entry *>(ctx.allocate(table.bytes));
  table.cached = entries;
  MICROGRAD_CUDA_CHECK(cudaMemcpyAsync(table.device, table.cached.data(),
                                       table.bytes, cudaMemcpyHostToDevice,
                                       ctx.stream()));
  return table.device;
}

bool SameEntry(const AdamWTensor &a, const AdamWTensor &b) {
  return a.data == b.data && a.grad == b.grad && a.m == b.m && a.v == b.v &&
         a.n == b.n && a.decay == b.decay;
}

const AdamWTensor *UploadAdamWTable(const std::vector<AdamWTensor> &tensors) {
  static DeviceTable<AdamWTensor> table;
  return UploadTable(table, tensors, SameEntry);
}

struct ClipTensor {
  scalar_t *grad;
  size_t n;
};

constexpr size_t kClipMaxBlocksPerTensor = 128;
constexpr size_t kAdamWMaxBlocksPerTensor = 1024;

__global__ void MultiGradNormSquaredKernel(const ClipTensor *tensors,
                                           scalar_t *total) {
  __shared__ scalar_t shared[kBlockSize];
  const ClipTensor t = tensors[blockIdx.y];
  size_t tid = threadIdx.x;

  scalar_t local = 0.0f;
  for (size_t i = blockIdx.x * blockDim.x + tid; i < t.n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    scalar_t g = t.grad[i];
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
    atomicAdd(total, shared[0]);
  }
}

__global__ void MultiClipScaleKernel(const ClipTensor *tensors,
                                     const scalar_t *total, scalar_t max_norm) {
  const ClipTensor t = tensors[blockIdx.y];
  const scalar_t scale = fminf(1.0f, max_norm / (sqrtf(*total) + 1e-6f));
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < t.n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    t.grad[i] *= scale;
  }
}

bool SameClipEntry(const ClipTensor &a, const ClipTensor &b) {
  return a.grad == b.grad && a.n == b.n;
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

const scalar_t *GradNormSquaredTotal(const std::vector<Tensor *> &params) {
  std::vector<ClipTensor> tensors;
  tensors.reserve(params.size());
  size_t max_n = 0;
  for (Tensor *param : params) {
    tensors.push_back({.grad = GradPtr(param), .n = param->size()});
    max_n = std::max(max_n, param->size());
  }

  static DeviceTable<ClipTensor> table;
  const ClipTensor *device_tensors = UploadTable(table, tensors, SameClipEntry);

  auto &ctx = CudaContext::instance();
  static auto *total =
      static_cast<scalar_t *>(ctx.allocate(sizeof(scalar_t)));
  MICROGRAD_CUDA_CHECK(
      cudaMemsetAsync(total, 0, sizeof(scalar_t), ctx.stream()));
  if (tensors.empty()) {
    return total;
  }
  const size_t blocks = std::clamp<size_t>(
      (max_n + kBlockSize - 1) / kBlockSize, 1, kClipMaxBlocksPerTensor);
  const dim3 grid(static_cast<unsigned>(blocks),
                  static_cast<unsigned>(tensors.size()));
  MultiGradNormSquaredKernel<<<grid, kBlockSize, 0, ctx.stream()>>>(
      device_tensors, total);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  return total;
}

void ClipGradNorm(const std::vector<Tensor *> &params, scalar_t max_norm) {
  if (params.empty()) {
    return;
  }
  std::vector<ClipTensor> tensors;
  tensors.reserve(params.size());
  size_t max_n = 0;
  for (Tensor *param : params) {
    tensors.push_back({.grad = GradPtr(param), .n = param->size()});
    max_n = std::max(max_n, param->size());
  }

  static DeviceTable<ClipTensor> table;
  const ClipTensor *device_tensors = UploadTable(table, tensors, SameClipEntry);

  auto &ctx = CudaContext::instance();
  auto *total = static_cast<scalar_t *>(ctx.allocate(sizeof(scalar_t)));
  MICROGRAD_CUDA_CHECK(
      cudaMemsetAsync(total, 0, sizeof(scalar_t), ctx.stream()));

  const size_t blocks = std::clamp<size_t>(
      (max_n + kBlockSize - 1) / kBlockSize, 1, kClipMaxBlocksPerTensor);
  const dim3 grid(static_cast<unsigned>(blocks),
                  static_cast<unsigned>(tensors.size()));
  MultiGradNormSquaredKernel<<<grid, kBlockSize, 0, ctx.stream()>>>(
      device_tensors, total);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  MultiClipScaleKernel<<<grid, kBlockSize, 0, ctx.stream()>>>(device_tensors,
                                                              total, max_norm);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  ctx.deallocate(total, sizeof(scalar_t));
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
                    scalar_t bias_correction2, const scalar_t *clip_total,
                    scalar_t max_norm) {
  if (tensors.empty()) {
    return;
  }
  size_t max_n = 0;
  for (const auto &t : tensors) {
    max_n = std::max(max_n, t.n);
  }

  auto &ctx = CudaContext::instance();
  const AdamWTensor *device_tensors = UploadAdamWTable(tensors);

  const size_t blocks = std::clamp<size_t>(
      (max_n + kBlockSize - 1) / kBlockSize, 1, kAdamWMaxBlocksPerTensor);
  dim3 grid(static_cast<unsigned>(blocks),
            static_cast<unsigned>(tensors.size()));
  FusedAdamWStepKernel<<<grid, kBlockSize, 0, ctx.stream()>>>(
      device_tensors, lr, beta1, beta2, eps, weight_decay, bias_correction1,
      bias_correction2, clip_total, max_norm);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

__global__ void MuonMomentumKernel(scalar_t *buf, const scalar_t *grad,
                                   scalar_t *update, size_t n, scalar_t beta,
                                   bool nesterov) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const scalar_t g = grad[i];
    const scalar_t m = (beta * buf[i]) + ((1.0f - beta) * g);
    buf[i] = m;
    update[i] = nesterov ? (beta * m) + ((1.0f - beta) * g) : m;
  }
}

__global__ void ScaleByInverseNormKernel(scalar_t *x, size_t n,
                                         const scalar_t *sum_sq) {
  const scalar_t inv = 1.0f / (sqrtf(*sum_sq) + 1e-7f);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    x[i] *= inv;
  }
}

__global__ void MuonApplyKernel(scalar_t *data, const scalar_t *update,
                                size_t n, scalar_t decay_factor,
                                scalar_t step) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    data[i] = (data[i] * decay_factor) - (step * update[i]);
  }
}

}  // namespace

void MuonStep(Tensor &param, Tensor &momentum, scalar_t lr, scalar_t beta,
              scalar_t weight_decay, bool nesterov, size_t ns_steps,
              scalar_t update_scale) {
#ifdef MICROGRAD_CUBLAS_ENABLED
  constexpr scalar_t kA = 3.4445f;
  constexpr scalar_t kB = -4.7750f;
  constexpr scalar_t kC = 2.0315f;

  auto &ctx = CudaContext::instance();
  cudaStream_t stream = ctx.stream();
  cublasHandle_t handle = ctx.cublasHandle();
  const size_t rows = param.shape()[0];
  const size_t cols = param.shape()[1];
  const size_t n = rows * cols;
  const size_t small = std::min(rows, cols);
  const bool wide = rows <= cols;

  Storage x_storage(n * sizeof(scalar_t), Device::CUDA);
  Storage y_storage(n * sizeof(scalar_t), Device::CUDA);
  Storage gram_storage(small * small * sizeof(scalar_t), Device::CUDA);
  Storage poly_storage(small * small * sizeof(scalar_t), Device::CUDA);
  Storage norm_storage(sizeof(scalar_t), Device::CUDA);
  auto *x = static_cast<scalar_t *>(x_storage.device_pointer());
  auto *y = static_cast<scalar_t *>(y_storage.device_pointer());
  auto *gram = static_cast<scalar_t *>(gram_storage.device_pointer());
  auto *poly = static_cast<scalar_t *>(poly_storage.device_pointer());
  auto *norm = static_cast<scalar_t *>(norm_storage.device_pointer());

  MuonMomentumKernel<<<GridSize(n), kBlockSize, 0, stream>>>(
      DataPtr(&momentum), GradPtr(&param), x, n, beta, nesterov);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  MICROGRAD_CUDA_CHECK(cudaMemsetAsync(norm, 0, sizeof(scalar_t), stream));
  GradNormSquaredKernel<<<GridSize(n), kBlockSize, 0, stream>>>(x, norm, n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  ScaleByInverseNormKernel<<<GridSize(n), kBlockSize, 0, stream>>>(x, n,
                                                                   norm);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();

  const int ld = static_cast<int>(cols);
  const int s = static_cast<int>(small);
  const int big = static_cast<int>(wide ? cols : rows);
  const scalar_t one = 1.0f;
  const scalar_t zero = 0.0f;
  const scalar_t c = kC;
  const scalar_t b = kB;
  const scalar_t a = kA;
  for (size_t step = 0; step < ns_steps; step++) {
    if (wide) {
      MICROGRAD_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, s,
                                         s, big, &one, x, ld, x, ld, &zero,
                                         gram, s));
    } else {
      MICROGRAD_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, s,
                                         s, big, &one, x, ld, x, ld, &zero,
                                         gram, s));
    }
    MICROGRAD_CUDA_CHECK(cudaMemcpyAsync(poly, gram,
                                         small * small * sizeof(scalar_t),
                                         cudaMemcpyDeviceToDevice, stream));
    MICROGRAD_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, s, s,
                                       s, &c, gram, s, gram, s, &b, poly, s));
    MICROGRAD_CUDA_CHECK(cudaMemcpyAsync(y, x, n * sizeof(scalar_t),
                                         cudaMemcpyDeviceToDevice, stream));
    if (wide) {
      MICROGRAD_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ld,
                                         s, s, &one, x, ld, poly, s, &a, y,
                                         ld));
    } else {
      MICROGRAD_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, s,
                                         big, s, &one, poly, s, x, ld, &a, y,
                                         ld));
    }
    std::swap(x, y);
  }

  MuonApplyKernel<<<GridSize(n), kBlockSize, 0, stream>>>(
      DataPtr(&param), x, n, 1.0f - (lr * weight_decay), lr * update_scale);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
#else
  (void)param;
  (void)momentum;
  (void)lr;
  (void)beta;
  (void)weight_decay;
  (void)nesterov;
  (void)ns_steps;
  (void)update_scale;
  throw std::runtime_error("CUDA Muon needs a build with -DENABLE_CUBLAS=ON");
#endif
}

}  // namespace micrograd::cuda::ops

#endif
