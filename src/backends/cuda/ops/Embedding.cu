#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cuda/ops/Bf16.h"
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

__device__ inline float ToFloat(float value) { return value; }
__device__ inline float ToFloat(bf16 value) { return __bfloat162float(value); }

template <typename T>
__device__ inline T FromFloat(float value);
template <>
__device__ inline float FromFloat<float>(float value) {
  return value;
}
template <>
__device__ inline bf16 FromFloat<bf16>(float value) {
  return __float2bfloat16(value);
}

template <typename T>
__device__ MaxSum RowMaxSum(const T *row, size_t cols) {
  __shared__ MaxSum shared[kBlockSize];
  MaxSum local = {0.0f, 0.0f};
  if constexpr (std::is_same_v<T, bf16>) {
    float max = -INFINITY;
    float sum = 0.0f;
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
      const float x = ToFloat(row[j]);
      if (x > max) {
        sum = (sum * __expf(max - x)) + 1.0f;
        max = x;
      } else {
        sum += __expf(x - max);
      }
    }
    local = {sum == 0.0f ? 0.0f : max, sum};
  } else {
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
      local = CombineMaxSum(local, {ToFloat(row[j]), 1.0f});
    }
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

template <typename T>
__global__ void CrossEntropyKernel(const T *logits, const scalar_t *targets,
                                   scalar_t *out, size_t rows, size_t width,
                                   size_t cols) {
  for (size_t i = blockIdx.x; i < rows; i += gridDim.x) {
    const T *row = logits + (i * width);
    const MaxSum ms = RowMaxSum(row, cols);
    if (threadIdx.x == 0) {
      auto target = static_cast<size_t>(llroundf(targets[i]));
      out[i] = ms.max + logf(ms.sum) - ToFloat(row[target]);
    }
  }
}

template <typename T>
__global__ void CrossEntropyBackwardKernel(const T *logits,
                                           const scalar_t *targets,
                                           const scalar_t *losses,
                                           const scalar_t *out_grad, T *grad,
                                           bool accumulate, size_t rows,
                                           size_t width, size_t cols) {
  for (size_t i = blockIdx.x; i < rows; i += gridDim.x) {
    const T *row = logits + (i * width);
    T *grad_row = grad + (i * width);
    const auto target = static_cast<size_t>(llroundf(targets[i]));
    const scalar_t log_sum_exp = losses[i] + ToFloat(row[target]);
    const scalar_t scale = out_grad[i];
    for (size_t j = threadIdx.x; j < width; j += blockDim.x) {
      scalar_t g = 0.0f;
      if (j < cols) {
        g = expf(ToFloat(row[j]) - log_sum_exp) * scale;
        if (j == target) {
          g -= out_grad[i];
        }
      }
      grad_row[j] = accumulate ? FromFloat<T>(ToFloat(grad_row[j]) + g)
                               : FromFloat<T>(g);
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

size_t ClassCount(int64_t classes, size_t width) {
  return classes == 0 ? width : static_cast<size_t>(classes);
}

void CrossEntropy(const OpArgs &args) {
  size_t width = args.lhs->shape().back();
  size_t rows = args.lhs->size() / width;
  size_t cols = ClassCount(args.dim, width);
  CrossEntropyKernel<<<RowGrid(rows), kBlockSize, 0,
                       CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), rows, width,
      cols);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

std::function<void()> CrossEntropyBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, targets = args.rhs,
          classes = args.dim]() {
    size_t width = lhs->shape().back();
    size_t rows = lhs->size() / width;
    size_t cols = ClassCount(classes, width);
    const scalar_t *out_grad = GradPtr(out);
    const GradTarget grad = GradForWrite(lhs.get());
    CrossEntropyBackwardKernel<<<RowGrid(rows), kBlockSize, 0,
                                 CudaContext::instance().stream()>>>(
        DataPtr(static_cast<const Tensor *>(lhs.get())),
        DataPtr(static_cast<const Tensor *>(targets.get())),
        DataPtr(static_cast<const Tensor *>(out)), out_grad, grad.ptr,
        grad.accumulate, rows, width, cols);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

void CrossEntropyBf16(const OpArgs &args) {
  size_t width = args.lhs->shape().back();
  size_t rows = args.lhs->size() / width;
  size_t cols = ClassCount(args.dim, width);
  CrossEntropyKernel<<<RowGrid(rows), kBlockSize, 0,
                       CudaContext::instance().stream()>>>(
      Bf16Data(args.lhs), DataPtr(args.rhs), DataPtr(args.out), rows, width,
      cols);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

std::function<void()> CrossEntropyBackwardBf16(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, targets = args.rhs,
          classes = args.dim]() {
    size_t width = lhs->shape().back();
    size_t rows = lhs->size() / width;
    size_t cols = ClassCount(classes, width);
    const scalar_t *out_grad = GradPtr(out);
    const Bf16GradTarget grad = Bf16GradForWrite(lhs.get());
    CrossEntropyBackwardKernel<<<RowGrid(rows), kBlockSize, 0,
                                 CudaContext::instance().stream()>>>(
        Bf16Data(static_cast<const Tensor *>(lhs.get())),
        DataPtr(static_cast<const Tensor *>(targets.get())),
        DataPtr(static_cast<const Tensor *>(out)), out_grad, grad.ptr,
        grad.accumulate, rows, width, cols);
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
  registry.Register(OpId::kCrossEntropy, Device::CUDA, CrossEntropyBf16,
                    DType::kBFloat16);
  registry.RegisterBackward(OpId::kCrossEntropy, Device::CUDA,
                            CrossEntropyBackwardBf16, DType::kBFloat16);
}

}  // namespace micrograd::cuda::ops

#endif
