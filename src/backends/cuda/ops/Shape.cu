#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <algorithm>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cuda/ops/Bf16.h"
#include "micrograd/backends/cuda/ops/Grad.h"
#include "micrograd/backends/cuda/ops/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::cuda::ops {
namespace {

constexpr int kBlockSize = 256;
constexpr size_t kMaxBlocks = 65535;
constexpr size_t kMaxRank = 8;

int GridSize(size_t n) {
  size_t blocks = (n + kBlockSize - 1) / kBlockSize;
  return static_cast<int>(std::min(blocks, kMaxBlocks));
}

struct StridedLayout {
  size_t shape[kMaxRank];
  size_t strides[kMaxRank];
  size_t rank;
};

StridedLayout MakeLayout(const std::vector<size_t> &shape,
                         std::span<const size_t> strides) {
  StridedLayout layout{};
  layout.rank = shape.size();
  for (size_t d = 0; d < layout.rank; d++) {
    layout.shape[d] = shape[d];
    layout.strides[d] = strides[d];
  }
  return layout;
}

__device__ size_t StridedOffset(const StridedLayout &layout, size_t linear) {
  size_t offset = 0;
  for (size_t d = layout.rank; d > 0; d--) {
    size_t dim = layout.shape[d - 1];
    offset += (linear % dim) * layout.strides[d - 1];
    linear /= dim;
  }
  return offset;
}

__global__ void ReshapeBackwardKernel(const scalar_t *out_grad,
                                      scalar_t *lhs_grad, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    lhs_grad[i] += out_grad[i];
  }
}

__global__ void StridedCopyKernel(const scalar_t *source, scalar_t *out,
                                  StridedLayout layout, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = source[StridedOffset(layout, i)];
  }
}

__global__ void StridedCopyBackwardKernel(const scalar_t *out_grad,
                                          scalar_t *source_grad,
                                          StridedLayout layout, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    atomicAdd(&source_grad[StridedOffset(layout, i)], out_grad[i]);
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

std::function<void()> ReshapeBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    if (MoveGrad(out, lhs.get())) {
      return;
    }
    size_t n = out->size();
    ReshapeBackwardKernel<<<GridSize(n), kBlockSize, 0,
                            CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

void StridedCopy(const OpArgs &args) {
  StridedLayout layout = MakeLayout(args.out->shape(), args.strides);
  size_t n = args.out->size();
  StridedCopyKernel<<<GridSize(n), kBlockSize, 0,
                      CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), layout, n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

std::function<void()> StridedCopyBackward(const GradArgs &args) {
  auto strides = std::make_shared<const std::vector<size_t>>(
      args.strides.begin(), args.strides.end());
  return [out = args.out, lhs = args.lhs, strides]() {
    StridedLayout layout = MakeLayout(out->shape(), *strides);
    size_t n = out->size();
    StridedCopyBackwardKernel<<<GridSize(n), kBlockSize, 0,
                                CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), layout, n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

struct ToBf16Op {
  const float *in;
  bf16 *out;
  __device__ void vector(size_t i) const {
    Bf16x8 values;
    for (int j = 0; j < kBf16Vector; j++) {
      values.v[j] = in[i + j];
    }
    StoreBf16x8(out + i, values);
  }
  __device__ void scalar(size_t i) const { out[i] = __float2bfloat16(in[i]); }
};

struct ToFloatOp {
  const bf16 *in;
  float *out;
  bool accumulate;
  __device__ void vector(size_t i) const {
    const Bf16x8 values = LoadBf16x8(in + i);
    for (int j = 0; j < kBf16Vector; j++) {
      out[i + j] = accumulate ? out[i + j] + values.v[j] : values.v[j];
    }
  }
  __device__ void scalar(size_t i) const {
    const float value = __bfloat162float(in[i]);
    out[i] = accumulate ? out[i] + value : value;
  }
};

struct AddToBf16Op {
  const float *in;
  bf16 *out;
  bool accumulate;
  __device__ void vector(size_t i) const {
    Bf16x8 values = accumulate ? LoadBf16x8(out + i) : Bf16x8{};
    for (int j = 0; j < kBf16Vector; j++) {
      values.v[j] = accumulate ? values.v[j] + in[i + j] : in[i + j];
    }
    StoreBf16x8(out + i, values);
  }
  __device__ void scalar(size_t i) const {
    const float old = accumulate ? __bfloat162float(out[i]) : 0.0f;
    out[i] = __float2bfloat16(old + in[i]);
  }
};

struct AddBf16Op {
  const bf16 *in;
  bf16 *out;
  __device__ void vector(size_t i) const {
    Bf16x8 values = LoadBf16x8(out + i);
    const Bf16x8 extra = LoadBf16x8(in + i);
    for (int j = 0; j < kBf16Vector; j++) {
      values.v[j] += extra.v[j];
    }
    StoreBf16x8(out + i, values);
  }
  __device__ void scalar(size_t i) const {
    out[i] = __float2bfloat16(__bfloat162float(out[i]) +
                              __bfloat162float(in[i]));
  }
};

void Cast(const OpArgs &args) {
  const size_t n = args.out->size();
  if (args.out->dtype() == DType::kBFloat16) {
    LaunchBf16Map(n, ToBf16Op{DataPtr(args.lhs), Bf16Data(args.out)});
    return;
  }
  LaunchBf16Map(n, ToFloatOp{Bf16Data(args.lhs), DataPtr(args.out), false});
}

std::function<void()> CastBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    if (!lhs->requires_grad()) {
      return;
    }
    const size_t n = lhs->size();
    if (lhs->dtype() == DType::kFloat32) {
      const GradTarget target = GradForWrite(lhs.get());
      LaunchBf16Map(n, ToFloatOp{Bf16Grad(out), target.ptr, target.accumulate});
      return;
    }
    const Bf16GradTarget target = Bf16GradForWrite(lhs.get());
    LaunchBf16Map(n, AddToBf16Op{GradPtr(out), target.ptr, target.accumulate});
  };
}

std::function<void()> ReshapeBackwardBf16(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    if (MoveGrad(out, lhs.get())) {
      return;
    }
    LaunchBf16Map(out->size(), AddBf16Op{Bf16Grad(out), Bf16Grad(lhs.get())});
  };
}

}  // namespace

void RegisterShapeOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kStridedCopy, Device::CUDA, StridedCopy);
  registry.RegisterBackward(OpId::kReshape, Device::CUDA, ReshapeBackward);
  registry.RegisterBackward(OpId::kReshape, Device::CUDA, ReshapeBackwardBf16,
                            DType::kBFloat16);
  for (DType dtype : {DType::kFloat32, DType::kBFloat16}) {
    registry.Register(OpId::kCast, Device::CUDA, Cast, dtype);
    registry.RegisterBackward(OpId::kCast, Device::CUDA, CastBackward, dtype);
  }
  registry.RegisterBackward(OpId::kStridedCopy, Device::CUDA,
                            StridedCopyBackward);
}

}  // namespace micrograd::cuda::ops

#endif
