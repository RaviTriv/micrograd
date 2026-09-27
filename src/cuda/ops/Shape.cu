#include "micrograd/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <algorithm>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/cuda/ops/Ops.h"
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

void StridedCopy(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  StridedLayout layout = MakeLayout(args.out->shape(), args.strides);
  size_t n = args.out->size();
  StridedCopyKernel<<<GridSize(n), kBlockSize, 0,
                      CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), layout, n);
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
  };
}

}  // namespace

void RegisterShapeOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kStridedCopy, Device::CUDA, StridedCopy);
  registry.RegisterBackward(OpId::kStridedCopy, Device::CUDA,
                            StridedCopyBackward);
}

}  // namespace micrograd::cuda::ops

#endif
