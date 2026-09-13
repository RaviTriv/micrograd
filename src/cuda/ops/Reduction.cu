#include "micrograd/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <cstddef>
#include <functional>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/cuda/ops/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::cuda::ops {
namespace {

constexpr int kBlockSize = 256;
constexpr size_t kMaxBlocks = 65535;

int GridSize(size_t n) {
  size_t blocks = (n + kBlockSize - 1) / kBlockSize;
  return static_cast<int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

struct AxisLayout {
  size_t outer = 1;
  size_t reduced = 1;
  size_t inner = 1;
};

AxisLayout LayoutFor(const std::vector<size_t> &shape, size_t axis) {
  AxisLayout layout;
  for (size_t i = 0; i < axis; i++) {
    layout.outer *= shape[i];
  }
  layout.reduced = shape[axis];
  for (size_t i = axis + 1; i < shape.size(); i++) {
    layout.inner *= shape[i];
  }
  return layout;
}

__device__ size_t SliceOffset(const AxisLayout &layout, size_t slice,
                              size_t position) {
  size_t outer = slice / layout.inner;
  size_t inner = slice % layout.inner;
  return (((outer * layout.reduced) + position) * layout.inner) + inner;
}

__global__ void SumReduceKernel(const scalar_t *in, scalar_t *out, size_t n) {
  __shared__ scalar_t shared[kBlockSize];
  size_t tid = threadIdx.x;

  scalar_t local = 0.0f;
  for (size_t i = blockIdx.x * blockDim.x + tid; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    local += in[i];
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

__global__ void SumBackwardKernel(const scalar_t *out_grad, scalar_t *a_grad,
                                  size_t n) {
  scalar_t grad = out_grad[0];
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    a_grad[i] += grad;
  }
}

__global__ void SumDimKernel(const scalar_t *in, scalar_t *out,
                             AxisLayout layout) {
  __shared__ scalar_t shared[kBlockSize];
  size_t slice = blockIdx.x;
  size_t tid = threadIdx.x;

  scalar_t local = 0.0f;
  for (size_t k = tid; k < layout.reduced; k += blockDim.x) {
    local += in[SliceOffset(layout, slice, k)];
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
    out[slice] = shared[0];
  }
}

__global__ void SumDimBackwardKernel(const scalar_t *out_grad, scalar_t *a_grad,
                                     AxisLayout layout, size_t n) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
    size_t inner = idx % layout.inner;
    size_t outer = (idx / layout.inner) / layout.reduced;
    a_grad[idx] += out_grad[(outer * layout.inner) + inner];
  }
}

__global__ void MaxKernel(const scalar_t *in, scalar_t *out,
                          AxisLayout layout) {
  __shared__ scalar_t shared_val[kBlockSize];
  __shared__ size_t shared_idx[kBlockSize];
  size_t slice = blockIdx.x;
  size_t tid = threadIdx.x;

  scalar_t best_val = in[SliceOffset(layout, slice, tid % layout.reduced)];
  size_t best_idx = tid % layout.reduced;
  for (size_t k = tid; k < layout.reduced; k += blockDim.x) {
    scalar_t candidate = in[SliceOffset(layout, slice, k)];
    if (candidate > best_val) {
      best_val = candidate;
      best_idx = k;
    }
  }
  shared_val[tid] = best_val;
  shared_idx[tid] = best_idx;
  __syncthreads();

  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride && shared_val[tid + stride] > shared_val[tid]) {
      shared_val[tid] = shared_val[tid + stride];
      shared_idx[tid] = shared_idx[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[slice] = shared_val[0];
  }
}

__global__ void ArgmaxKernel(const scalar_t *in, scalar_t *out,
                             AxisLayout layout) {
  __shared__ scalar_t shared_val[kBlockSize];
  __shared__ size_t shared_idx[kBlockSize];
  size_t slice = blockIdx.x;
  size_t tid = threadIdx.x;

  scalar_t best_val = in[SliceOffset(layout, slice, tid % layout.reduced)];
  size_t best_idx = tid % layout.reduced;
  for (size_t k = tid; k < layout.reduced; k += blockDim.x) {
    scalar_t candidate = in[SliceOffset(layout, slice, k)];
    if (candidate > best_val) {
      best_val = candidate;
      best_idx = k;
    }
  }
  shared_val[tid] = best_val;
  shared_idx[tid] = best_idx;
  __syncthreads();

  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride && shared_val[tid + stride] > shared_val[tid]) {
      shared_val[tid] = shared_val[tid + stride];
      shared_idx[tid] = shared_idx[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[slice] = static_cast<scalar_t>(shared_idx[0]);
  }
}

__global__ void MaxBackwardKernel(const scalar_t *in, const scalar_t *out_grad,
                                  scalar_t *a_grad, AxisLayout layout) {
  __shared__ scalar_t shared_val[kBlockSize];
  __shared__ size_t shared_idx[kBlockSize];
  size_t slice = blockIdx.x;
  size_t tid = threadIdx.x;

  scalar_t best_val = in[SliceOffset(layout, slice, tid % layout.reduced)];
  size_t best_idx = tid % layout.reduced;
  for (size_t k = tid; k < layout.reduced; k += blockDim.x) {
    scalar_t candidate = in[SliceOffset(layout, slice, k)];
    if (candidate > best_val) {
      best_val = candidate;
      best_idx = k;
    }
  }
  shared_val[tid] = best_val;
  shared_idx[tid] = best_idx;
  __syncthreads();

  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride && shared_val[tid + stride] > shared_val[tid]) {
      shared_val[tid] = shared_val[tid + stride];
      shared_idx[tid] = shared_idx[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    a_grad[SliceOffset(layout, slice, shared_idx[0])] += out_grad[slice];
  }
}

__global__ void SoftmaxKernel(const scalar_t *in, scalar_t *out,
                              AxisLayout layout) {
  __shared__ scalar_t shared[kBlockSize];
  size_t slice = blockIdx.x;
  size_t tid = threadIdx.x;

  scalar_t local_max = in[SliceOffset(layout, slice, tid % layout.reduced)];
  for (size_t k = tid; k < layout.reduced; k += blockDim.x) {
    scalar_t candidate = in[SliceOffset(layout, slice, k)];
    local_max = candidate > local_max ? candidate : local_max;
  }
  shared[tid] = local_max;
  __syncthreads();
  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = shared[tid + stride] > shared[tid] ? shared[tid + stride]
                                                       : shared[tid];
    }
    __syncthreads();
  }
  scalar_t slice_max = shared[0];
  __syncthreads();

  scalar_t local_sum = 0.0f;
  for (size_t k = tid; k < layout.reduced; k += blockDim.x) {
    size_t offset = SliceOffset(layout, slice, k);
    scalar_t value = expf(in[offset] - slice_max);
    out[offset] = value;
    local_sum += value;
  }
  shared[tid] = local_sum;
  __syncthreads();
  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  scalar_t total = shared[0];
  __syncthreads();

  for (size_t k = tid; k < layout.reduced; k += blockDim.x) {
    out[SliceOffset(layout, slice, k)] /= total;
  }
}

__global__ void LogSoftmaxKernel(const scalar_t *in, scalar_t *out,
                                 AxisLayout layout) {
  __shared__ scalar_t shared[kBlockSize];
  size_t slice = blockIdx.x;
  size_t tid = threadIdx.x;

  scalar_t local_max = in[SliceOffset(layout, slice, tid % layout.reduced)];
  for (size_t k = tid; k < layout.reduced; k += blockDim.x) {
    scalar_t candidate = in[SliceOffset(layout, slice, k)];
    local_max = candidate > local_max ? candidate : local_max;
  }
  shared[tid] = local_max;
  __syncthreads();
  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = shared[tid + stride] > shared[tid] ? shared[tid + stride]
                                                       : shared[tid];
    }
    __syncthreads();
  }
  scalar_t slice_max = shared[0];
  __syncthreads();

  scalar_t local_sum = 0.0f;
  for (size_t k = tid; k < layout.reduced; k += blockDim.x) {
    local_sum += expf(in[SliceOffset(layout, slice, k)] - slice_max);
  }
  shared[tid] = local_sum;
  __syncthreads();
  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  scalar_t shift = slice_max + logf(shared[0]);
  __syncthreads();

  for (size_t k = tid; k < layout.reduced; k += blockDim.x) {
    size_t offset = SliceOffset(layout, slice, k);
    out[offset] = in[offset] - shift;
  }
}

__global__ void SoftmaxBackwardKernel(const scalar_t *out_grad,
                                      const scalar_t *out_data,
                                      scalar_t *a_grad, AxisLayout layout) {
  __shared__ scalar_t shared[kBlockSize];
  size_t slice = blockIdx.x;
  size_t tid = threadIdx.x;

  scalar_t local = 0.0f;
  for (size_t k = tid; k < layout.reduced; k += blockDim.x) {
    size_t offset = SliceOffset(layout, slice, k);
    local += out_grad[offset] * out_data[offset];
  }
  shared[tid] = local;
  __syncthreads();
  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  scalar_t weighted = shared[0];
  __syncthreads();

  for (size_t k = tid; k < layout.reduced; k += blockDim.x) {
    size_t offset = SliceOffset(layout, slice, k);
    a_grad[offset] += out_data[offset] * (out_grad[offset] - weighted);
  }
}

__global__ void LogSoftmaxBackwardKernel(const scalar_t *out_grad,
                                         const scalar_t *out_data,
                                         scalar_t *a_grad, AxisLayout layout) {
  __shared__ scalar_t shared[kBlockSize];
  size_t slice = blockIdx.x;
  size_t tid = threadIdx.x;

  scalar_t local = 0.0f;
  for (size_t k = tid; k < layout.reduced; k += blockDim.x) {
    local += out_grad[SliceOffset(layout, slice, k)];
  }
  shared[tid] = local;
  __syncthreads();
  for (size_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  scalar_t total = shared[0];
  __syncthreads();

  for (size_t k = tid; k < layout.reduced; k += blockDim.x) {
    size_t offset = SliceOffset(layout, slice, k);
    a_grad[offset] += out_grad[offset] - (expf(out_data[offset]) * total);
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

void Sum(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  size_t n = args.lhs->size();
  cudaMemsetAsync(DataPtr(args.out), 0, sizeof(scalar_t),
                  CudaContext::instance().stream());
  SumReduceKernel<<<GridSize(n), kBlockSize, 0,
                    CudaContext::instance().stream()>>>(DataPtr(args.lhs),
                                                        DataPtr(args.out), n);
}

void SumDim(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  auto axis = static_cast<size_t>(args.dim);
  AxisLayout layout = LayoutFor(args.lhs->shape(), axis);
  size_t slices = layout.outer * layout.inner;
  SumDimKernel<<<static_cast<int>(slices), kBlockSize, 0,
                 CudaContext::instance().stream()>>>(DataPtr(args.lhs),
                                                     DataPtr(args.out), layout);
}

void Max(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  auto axis = static_cast<size_t>(args.dim);
  AxisLayout layout = LayoutFor(args.lhs->shape(), axis);
  size_t slices = layout.outer * layout.inner;
  MaxKernel<<<static_cast<int>(slices), kBlockSize, 0,
              CudaContext::instance().stream()>>>(DataPtr(args.lhs),
                                                  DataPtr(args.out), layout);
}

void Argmax(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  auto axis = static_cast<size_t>(args.dim);
  AxisLayout layout = LayoutFor(args.lhs->shape(), axis);
  size_t slices = layout.outer * layout.inner;
  ArgmaxKernel<<<static_cast<int>(slices), kBlockSize, 0,
                 CudaContext::instance().stream()>>>(DataPtr(args.lhs),
                                                     DataPtr(args.out), layout);
}

void Softmax(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  auto axis = static_cast<size_t>(args.dim);
  AxisLayout layout = LayoutFor(args.lhs->shape(), axis);
  size_t slices = layout.outer * layout.inner;
  SoftmaxKernel<<<static_cast<int>(slices), kBlockSize, 0,
                  CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), layout);
}

void LogSoftmax(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  auto axis = static_cast<size_t>(args.dim);
  AxisLayout layout = LayoutFor(args.lhs->shape(), axis);
  size_t slices = layout.outer * layout.inner;
  LogSoftmaxKernel<<<static_cast<int>(slices), kBlockSize, 0,
                     CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), layout);
}

std::function<void()> SumBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    SumBackwardKernel<<<GridSize(n), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), n);
  };
}

std::function<void()> SumDimBackward(const GradArgs &args) {
  auto axis = static_cast<size_t>(args.dim);
  return [out = args.out, lhs = args.lhs, axis]() {
    AxisLayout layout = LayoutFor(lhs->shape(), axis);
    size_t n = lhs->size();
    SumDimBackwardKernel<<<GridSize(n), kBlockSize, 0,
                           CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), layout, n);
  };
}

std::function<void()> MaxBackward(const GradArgs &args) {
  auto axis = static_cast<size_t>(args.dim);
  return [out = args.out, lhs = args.lhs, axis]() {
    AxisLayout layout = LayoutFor(lhs->shape(), axis);
    size_t slices = layout.outer * layout.inner;
    MaxBackwardKernel<<<static_cast<int>(slices), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        DataPtr(lhs.get()), GradPtr(out), GradPtr(lhs.get()), layout);
  };
}

std::function<void()> SoftmaxBackward(const GradArgs &args) {
  auto axis = static_cast<size_t>(args.dim);
  return [out = args.out, lhs = args.lhs, axis]() {
    AxisLayout layout = LayoutFor(lhs->shape(), axis);
    size_t slices = layout.outer * layout.inner;
    SoftmaxBackwardKernel<<<static_cast<int>(slices), kBlockSize, 0,
                            CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(out), GradPtr(lhs.get()), layout);
  };
}

std::function<void()> LogSoftmaxBackward(const GradArgs &args) {
  auto axis = static_cast<size_t>(args.dim);
  return [out = args.out, lhs = args.lhs, axis]() {
    AxisLayout layout = LayoutFor(lhs->shape(), axis);
    size_t slices = layout.outer * layout.inner;
    LogSoftmaxBackwardKernel<<<static_cast<int>(slices), kBlockSize, 0,
                               CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(out), GradPtr(lhs.get()), layout);
  };
}

}  // namespace

void RegisterReductionOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kSum, Device::CUDA, Sum);
  registry.Register(OpId::kSumDim, Device::CUDA, SumDim);
  registry.Register(OpId::kMax, Device::CUDA, Max);
  registry.Register(OpId::kArgmax, Device::CUDA, Argmax);
  registry.Register(OpId::kSoftmax, Device::CUDA, Softmax);
  registry.Register(OpId::kLogSoftmax, Device::CUDA, LogSoftmax);
  registry.RegisterBackward(OpId::kSum, Device::CUDA, SumBackward);
  registry.RegisterBackward(OpId::kSumDim, Device::CUDA, SumDimBackward);
  registry.RegisterBackward(OpId::kMax, Device::CUDA, MaxBackward);
  registry.RegisterBackward(OpId::kSoftmax, Device::CUDA, SoftmaxBackward);
  registry.RegisterBackward(OpId::kLogSoftmax, Device::CUDA,
                            LogSoftmaxBackward);
}

}  // namespace micrograd::cuda::ops

#endif
