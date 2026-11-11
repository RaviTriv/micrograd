#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "micrograd/Profiler.h"
#include "micrograd/Storage.h"
#include "micrograd/Tensor.h"
#include "micrograd/backends/cuda/ops/Bf16.h"
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

scalar_t ReadGradNorm(const scalar_t *total) {
  auto &ctx = CudaContext::instance();
  scalar_t sum_sq = 0.0f;
  MICROGRAD_CUDA_CHECK(cudaMemcpyAsync(&sum_sq, total, sizeof(scalar_t),
                                       cudaMemcpyDeviceToHost, ctx.stream()));
  ctx.synchronize();
  return std::sqrt(sum_sq);
}

scalar_t ClipGradNorm(const std::vector<Tensor *> &params, scalar_t max_norm) {
  if (params.empty()) {
    return 0.0f;
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
  if (max_norm > 0.0f) {
    MultiClipScaleKernel<<<grid, kBlockSize, 0, ctx.stream()>>>(
        device_tensors, total, max_norm);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const scalar_t norm = ReadGradNorm(total);
  ctx.deallocate(total, sizeof(scalar_t));
  return norm;
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

namespace {

#ifdef MICROGRAD_CUBLAS_ENABLED
constexpr size_t kMuonMaxBlocksPerTensor = 512;

__global__ void MuonMomentumBatchedKernel(const MuonTensor *tensors,
                                          scalar_t *update, scalar_t *norms,
                                          size_t n, scalar_t beta,
                                          bool nesterov,
                                          const scalar_t *clip_total,
                                          scalar_t max_norm) {
  __shared__ scalar_t shared[kBlockSize];
  const MuonTensor t = tensors[blockIdx.y];
  const scalar_t scale =
      clip_total != nullptr
          ? fminf(1.0f, max_norm / (sqrtf(*clip_total) + 1e-6f))
          : 1.0f;
  scalar_t *u = update + (static_cast<size_t>(blockIdx.y) * n);
  scalar_t local = 0.0f;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const scalar_t g = t.grad[i] * scale;
    const scalar_t m = (beta * t.momentum[i]) + ((1.0f - beta) * g);
    t.momentum[i] = m;
    const scalar_t v = nesterov ? (beta * m) + ((1.0f - beta) * g) : m;
    u[i] = v;
    local += v * v;
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (unsigned int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(&norms[blockIdx.y], shared[0]);
  }
}

__global__ void MuonNormalizeBf16Kernel(const scalar_t *update,
                                        const scalar_t *norms, bf16 *x,
                                        size_t n) {
  const size_t offset = static_cast<size_t>(blockIdx.y) * n;
  const scalar_t inv = 1.0f / (sqrtf(norms[blockIdx.y]) + 1e-7f);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    x[offset + i] = __float2bfloat16(update[offset + i] * inv);
  }
}

__global__ void MuonApplyBatchedKernel(const MuonTensor *tensors,
                                       const bf16 *update, size_t n,
                                       scalar_t decay_factor, scalar_t step) {
  const MuonTensor t = tensors[blockIdx.y];
  const bf16 *u = update + (static_cast<size_t>(blockIdx.y) * n);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    t.data[i] = (t.data[i] * decay_factor) - (step * __bfloat162float(u[i]));
  }
}

struct LtGemmKey {
  bool trans_a;
  bool trans_b;
  int m;
  int n;
  int k;
  int lda;
  int ldb;
  int ldc;
  long long stride_a;
  long long stride_b;
  long long stride_c;
  int batch;
  bool operator==(const LtGemmKey &) const = default;
};

struct LtGemmPlan {
  LtGemmKey key;
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatmulAlgo_t algo{};
  bool usable = false;
};

constexpr size_t kLtWorkspaceBytes = 32 << 20;

cublasLtMatrixLayout_t LtLayout(bool trans, int rows, int cols, int ld,
                                long long stride, int batch) {
  cublasLtMatrixLayout_t layout = nullptr;
  MICROGRAD_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &layout, CUDA_R_16BF, static_cast<uint64_t>(trans ? cols : rows),
      static_cast<uint64_t>(trans ? rows : cols), ld));
  MICROGRAD_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
  MICROGRAD_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
      sizeof(stride)));
  return layout;
}

const LtGemmPlan &LtPlanFor(const LtGemmKey &key) {
  static std::deque<LtGemmPlan> plans;
  for (const auto &plan : plans) {
    if (plan.key == key) {
      return plan;
    }
  }
  LtGemmPlan plan;
  plan.key = key;
  MICROGRAD_CUBLAS_CHECK(
      cublasLtMatmulDescCreate(&plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t op_a = key.trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t op_b = key.trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  MICROGRAD_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
  MICROGRAD_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
  plan.a =
      LtLayout(key.trans_a, key.m, key.k, key.lda, key.stride_a, key.batch);
  plan.b =
      LtLayout(key.trans_b, key.k, key.n, key.ldb, key.stride_b, key.batch);
  plan.c = LtLayout(false, key.m, key.n, key.ldc, key.stride_c, key.batch);

  cublasLtMatmulPreference_t preference = nullptr;
  MICROGRAD_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  const size_t workspace = kLtWorkspaceBytes;
  MICROGRAD_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
      sizeof(workspace)));
  cublasLtMatmulHeuristicResult_t result{};
  int returned = 0;
  const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
      CudaContext::instance().cublasLtHandle(), plan.desc, plan.a, plan.b,
      plan.c, plan.c, preference, 1, &result, &returned);
  MICROGRAD_CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  plan.usable = status == CUBLAS_STATUS_SUCCESS && returned > 0;
  plan.algo = result.algo;
  plans.push_back(plan);
  return plans.back();
}

void BatchedGemmBf16(cublasHandle_t handle, cublasOperation_t trans_a,
                     cublasOperation_t trans_b, int m, int n, int k,
                     float alpha, const bf16 *a, int lda, long long stride_a,
                     const bf16 *b, int ldb, long long stride_b, float beta,
                     bf16 *c, int ldc, long long stride_c, int batch) {
  MICROGRAD_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, trans_a, trans_b, m, n, k, &alpha, a, CUDA_R_16BF, lda, stride_a,
      b, CUDA_R_16BF, ldb, stride_b, &beta, c, CUDA_R_16BF, ldc, stride_c,
      batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void LtBatchedGemmBf16(cublasHandle_t handle, cublasOperation_t trans_a,
                       cublasOperation_t trans_b, int m, int n, int k,
                       float alpha, const bf16 *a, int lda, long long stride_a,
                       const bf16 *b, int ldb, long long stride_b, float beta,
                       const bf16 *c, bf16 *d, int ldc, long long stride_c,
                       int batch) {
  const LtGemmPlan &plan = LtPlanFor({.trans_a = trans_a == CUBLAS_OP_T,
                                      .trans_b = trans_b == CUBLAS_OP_T,
                                      .m = m,
                                      .n = n,
                                      .k = k,
                                      .lda = lda,
                                      .ldb = ldb,
                                      .ldc = ldc,
                                      .stride_a = stride_a,
                                      .stride_b = stride_b,
                                      .stride_c = stride_c,
                                      .batch = batch});
  auto &ctx = CudaContext::instance();
  cudaStream_t stream = ctx.stream();
  if (!plan.usable) {
    if (c != d) {
      MICROGRAD_CUDA_CHECK(cudaMemcpyAsync(
          d, c, static_cast<size_t>(stride_c) * batch * sizeof(bf16),
          cudaMemcpyDeviceToDevice, stream));
    }
    BatchedGemmBf16(handle, trans_a, trans_b, m, n, k, alpha, a, lda, stride_a,
                    b, ldb, stride_b, beta, d, ldc, stride_c, batch);
    return;
  }
  static void *workspace = ctx.allocate(kLtWorkspaceBytes);
  MICROGRAD_CUBLAS_CHECK(cublasLtMatmul(
      ctx.cublasLtHandle(), plan.desc, &alpha, a, plan.a, b, plan.b, &beta, c,
      plan.c, d, plan.c, &plan.algo, workspace, kLtWorkspaceBytes, stream));
}

bool SameMuonEntry(const MuonTensor &a, const MuonTensor &b) {
  return a.data == b.data && a.grad == b.grad && a.momentum == b.momentum;
}

const MuonTensor *UploadMuonTable(const std::vector<MuonTensor> &tensors) {
  static std::vector<DeviceTable<MuonTensor>> tables;
  for (auto &table : tables) {
    if (!table.cached.empty() &&
        table.cached.front().data == tensors.front().data) {
      return UploadTable(table, tensors, SameMuonEntry);
    }
  }
  tables.emplace_back();
  return UploadTable(tables.back(), tensors, SameMuonEntry);
}
#endif

}  // namespace

void MuonStepBatchedBf16(const std::vector<MuonTensor> &tensors, size_t rows,
                         size_t cols, scalar_t lr, scalar_t beta,
                         scalar_t weight_decay, bool nesterov, size_t ns_steps,
                         scalar_t update_scale, const scalar_t *clip_total,
                         scalar_t max_norm) {
#ifdef MICROGRAD_CUBLAS_ENABLED
  if (tensors.empty()) {
    return;
  }
  constexpr float kA = 3.4445f;
  constexpr float kB = -4.7750f;
  constexpr float kC = 2.0315f;

  auto &ctx = CudaContext::instance();
  cudaStream_t stream = ctx.stream();
  cublasHandle_t handle = ctx.cublasHandle();
  const size_t batch = tensors.size();
  const size_t n = rows * cols;
  const size_t small = std::min(rows, cols);
  const bool wide = rows <= cols;
  const MuonTensor *device_tensors = UploadMuonTable(tensors);

  Storage update_storage(batch * n * sizeof(scalar_t), Device::CUDA);
  Storage norm_storage(batch * sizeof(scalar_t), Device::CUDA);
  Storage x_storage(batch * n * sizeof(bf16), Device::CUDA);
  Storage y_storage(batch * n * sizeof(bf16), Device::CUDA);
  Storage gram_storage(batch * small * small * sizeof(bf16), Device::CUDA);
  Storage poly_storage(batch * small * small * sizeof(bf16), Device::CUDA);
  auto *update = static_cast<scalar_t *>(update_storage.device_pointer());
  auto *norms = static_cast<scalar_t *>(norm_storage.device_pointer());
  auto *x = static_cast<bf16 *>(x_storage.device_pointer());
  auto *y = static_cast<bf16 *>(y_storage.device_pointer());
  auto *gram = static_cast<bf16 *>(gram_storage.device_pointer());
  auto *poly = static_cast<bf16 *>(poly_storage.device_pointer());

  const dim3 grid(
      static_cast<unsigned int>(std::clamp<size_t>(
          (n + kBlockSize - 1) / kBlockSize, 1, kMuonMaxBlocksPerTensor)),
      static_cast<unsigned int>(batch));
  std::optional<profiler::Scope> scope;
  scope.emplace("muon_prep", "step", Device::CUDA);
  MICROGRAD_CUDA_CHECK(
      cudaMemsetAsync(norms, 0, batch * sizeof(scalar_t), stream));
  MuonMomentumBatchedKernel<<<grid, kBlockSize, 0, stream>>>(
      device_tensors, update, norms, n, beta, nesterov, clip_total, max_norm);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  MuonNormalizeBf16Kernel<<<grid, kBlockSize, 0, stream>>>(update, norms, x, n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();

  const int ld = static_cast<int>(cols);
  const int s = static_cast<int>(small);
  const int big = static_cast<int>(wide ? cols : rows);
  const auto stride_x = static_cast<long long>(n);
  const auto stride_g = static_cast<long long>(small * small);
  const int count = static_cast<int>(batch);
  scope.reset();
  scope.emplace("muon_newton_schulz", "step", Device::CUDA);
  for (size_t step = 0; step < ns_steps; step++) {
    if (wide) {
      BatchedGemmBf16(handle, CUBLAS_OP_T, CUBLAS_OP_N, s, s, big, 1.0f, x, ld,
                      stride_x, x, ld, stride_x, 0.0f, gram, s, stride_g,
                      count);
    } else {
      BatchedGemmBf16(handle, CUBLAS_OP_N, CUBLAS_OP_T, s, s, big, 1.0f, x, ld,
                      stride_x, x, ld, stride_x, 0.0f, gram, s, stride_g,
                      count);
    }
    LtBatchedGemmBf16(handle, CUBLAS_OP_N, CUBLAS_OP_N, s, s, s, kC, gram, s,
                      stride_g, gram, s, stride_g, kB, gram, poly, s, stride_g,
                      count);
    if (wide) {
      LtBatchedGemmBf16(handle, CUBLAS_OP_N, CUBLAS_OP_N, ld, s, s, 1.0f, x, ld,
                        stride_x, poly, s, stride_g, kA, x, y, ld, stride_x,
                        count);
    } else {
      LtBatchedGemmBf16(handle, CUBLAS_OP_N, CUBLAS_OP_N, s, big, s, 1.0f, poly,
                        s, stride_g, x, ld, stride_x, kA, x, y, ld, stride_x,
                        count);
    }
    std::swap(x, y);
  }

  scope.reset();
  scope.emplace("muon_apply", "step", Device::CUDA);
  MuonApplyBatchedKernel<<<grid, kBlockSize, 0, stream>>>(
      device_tensors, x, n, 1.0f - (lr * weight_decay), lr * update_scale);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
#else
  (void)tensors;
  (void)rows;
  (void)cols;
  (void)lr;
  (void)beta;
  (void)weight_decay;
  (void)nesterov;
  (void)ns_steps;
  (void)update_scale;
  (void)clip_total;
  (void)max_norm;
  throw std::runtime_error("CUDA Muon needs a build with -DENABLE_CUBLAS=ON");
#endif
}

}  // namespace micrograd::cuda::ops

#endif
