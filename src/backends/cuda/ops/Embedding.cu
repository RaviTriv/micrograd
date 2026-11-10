#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <algorithm>
#include <cstddef>
#include <functional>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cuda/ops/Grad.h"
#include "micrograd/backends/cuda/ops/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::cuda::ops {
namespace {

constexpr int kBlockSize = 256;
constexpr size_t kMaxBlocks = 65535;

int GridSize(size_t n) {
  size_t blocks = (n + kBlockSize - 1) / kBlockSize;
  return static_cast<int>(std::min(blocks, kMaxBlocks));
}

__global__ void EmbeddingLookupKernel(const scalar_t *weight,
                                      const scalar_t *indices, scalar_t *out,
                                      size_t dim, size_t count) {
  for (size_t p = blockIdx.x; p < count; p += gridDim.x) {
    auto index = static_cast<size_t>(llroundf(indices[p]));
    for (size_t d = threadIdx.x; d < dim; d += blockDim.x) {
      out[(p * dim) + d] = weight[(index * dim) + d];
    }
  }
}

__global__ void EmbeddingLookupBackwardKernel(const scalar_t *indices,
                                              const scalar_t *out_grad,
                                              scalar_t *weight_grad, size_t dim,
                                              size_t count) {
  for (size_t p = blockIdx.x; p < count; p += gridDim.x) {
    auto index = static_cast<size_t>(llroundf(indices[p]));
    for (size_t d = threadIdx.x; d < dim; d += blockDim.x) {
      atomicAdd(&weight_grad[(index * dim) + d], out_grad[(p * dim) + d]);
    }
  }
}

__global__ void GatherPerRowKernel(const scalar_t *values,
                                   const scalar_t *indices, scalar_t *out,
                                   size_t cols, size_t rows) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < rows;
       i += gridDim.x * blockDim.x) {
    auto index = static_cast<size_t>(llroundf(indices[i]));
    out[i] = values[(i * cols) + index];
  }
}

__global__ void GatherPerRowBackwardKernel(const scalar_t *indices,
                                           const scalar_t *out_grad,
                                           scalar_t *grad, size_t cols,
                                           size_t rows) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < rows;
       i += gridDim.x * blockDim.x) {
    auto index = static_cast<size_t>(llroundf(indices[i]));
    grad[(i * cols) + index] += out_grad[i];
  }
}

struct MaxSum {
  scalar_t max;
  scalar_t sum;
};

__device__ MaxSum CombineMaxSum(MaxSum a, MaxSum b) {
  if (a.sum == 0.0f) {
    return b;
  }
  if (b.sum == 0.0f) {
    return a;
  }
  const scalar_t m = fmaxf(a.max, b.max);
  return {m, (a.sum * expf(a.max - m)) + (b.sum * expf(b.max - m))};
}

__device__ MaxSum RowMaxSum(const scalar_t *row, size_t cols) {
  __shared__ MaxSum shared[kBlockSize];
  MaxSum local = {0.0f, 0.0f};
  for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
    local = CombineMaxSum(local, {row[j], 1.0f});
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          CombineMaxSum(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const MaxSum result = shared[0];
  __syncthreads();
  return result;
}

__global__ void CrossEntropyKernel(const scalar_t *logits,
                                   const scalar_t *targets, scalar_t *out,
                                   size_t rows, size_t cols) {
  for (size_t i = blockIdx.x; i < rows; i += gridDim.x) {
    const scalar_t *row = logits + (i * cols);
    const MaxSum ms = RowMaxSum(row, cols);
    if (threadIdx.x == 0) {
      auto target = static_cast<size_t>(llroundf(targets[i]));
      out[i] = ms.max + logf(ms.sum) - row[target];
    }
  }
}

__global__ void CrossEntropyBackwardKernel(const scalar_t *logits,
                                           const scalar_t *targets,
                                           const scalar_t *losses,
                                           const scalar_t *out_grad,
                                           scalar_t *grad, bool accumulate,
                                           size_t rows, size_t cols) {
  for (size_t i = blockIdx.x; i < rows; i += gridDim.x) {
    const scalar_t *row = logits + (i * cols);
    scalar_t *grad_row = grad + (i * cols);
    const auto target = static_cast<size_t>(llroundf(targets[i]));
    const scalar_t log_sum_exp = losses[i] + row[target];
    const scalar_t scale = out_grad[i];
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
      scalar_t g = expf(row[j] - log_sum_exp) * scale;
      if (j == target) {
        g -= out_grad[i];
      }
      grad_row[j] = accumulate ? grad_row[j] + g : g;
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

void EmbeddingLookup(const OpArgs &args) {
  size_t dim = args.lhs->shape()[1];
  size_t count = args.rhs->size();
  EmbeddingLookupKernel<<<GridSize(count), kBlockSize, 0,
                          CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), dim, count);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

std::function<void()> EmbeddingLookupBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, indices = args.rhs]() {
    size_t dim = lhs->shape()[1];
    size_t count = indices->size();
    EmbeddingLookupBackwardKernel<<<GridSize(count), kBlockSize, 0,
                                    CudaContext::instance().stream()>>>(
        DataPtr(indices.get()), GradPtr(out), GradPtr(lhs.get()), dim, count);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

void GatherPerRow(const OpArgs &args) {
  size_t rows = args.lhs->shape()[0];
  size_t cols = args.lhs->shape()[1];
  GatherPerRowKernel<<<GridSize(rows), kBlockSize, 0,
                       CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), cols, rows);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

std::function<void()> GatherPerRowBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, indices = args.rhs]() {
    size_t rows = lhs->shape()[0];
    size_t cols = lhs->shape()[1];
    GatherPerRowBackwardKernel<<<GridSize(rows), kBlockSize, 0,
                                 CudaContext::instance().stream()>>>(
        DataPtr(indices.get()), GradPtr(out), GradPtr(lhs.get()), cols, rows);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

int RowGrid(size_t rows) {
  return static_cast<int>(std::min(rows, kMaxBlocks));
}

void CrossEntropy(const OpArgs &args) {
  size_t cols = args.lhs->shape().back();
  size_t rows = args.lhs->size() / cols;
  CrossEntropyKernel<<<RowGrid(rows), kBlockSize, 0,
                       CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), rows, cols);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

std::function<void()> CrossEntropyBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, targets = args.rhs]() {
    size_t cols = lhs->shape().back();
    size_t rows = lhs->size() / cols;
    const scalar_t *out_grad = GradPtr(out);
    const GradTarget grad = GradForWrite(lhs.get());
    CrossEntropyBackwardKernel<<<RowGrid(rows), kBlockSize, 0,
                                 CudaContext::instance().stream()>>>(
        DataPtr(static_cast<const Tensor *>(lhs.get())),
        DataPtr(static_cast<const Tensor *>(targets.get())),
        DataPtr(static_cast<const Tensor *>(out)), out_grad, grad.ptr,
        grad.accumulate, rows, cols);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

}  // namespace

void RegisterEmbeddingOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kEmbeddingLookup, Device::CUDA, EmbeddingLookup);
  registry.RegisterBackward(OpId::kEmbeddingLookup, Device::CUDA,
                            EmbeddingLookupBackward);
  registry.Register(OpId::kGatherPerRow, Device::CUDA, GatherPerRow);
  registry.RegisterBackward(OpId::kGatherPerRow, Device::CUDA,
                            GatherPerRowBackward);
  registry.Register(OpId::kCrossEntropy, Device::CUDA, CrossEntropy);
  registry.RegisterBackward(OpId::kCrossEntropy, Device::CUDA,
                            CrossEntropyBackward);
}

}  // namespace micrograd::cuda::ops

#endif
