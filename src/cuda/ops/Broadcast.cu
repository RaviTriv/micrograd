#include "micrograd/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <algorithm>
#include <cstddef>
#include <functional>
#include <vector>

#include "micrograd/Broadcast.h"
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

struct BroadcastLayout {
  size_t shape[kMaxRank];
  size_t strides[kMaxRank];
  size_t rank;
};

BroadcastLayout MakeLayout(const std::vector<size_t> &shape,
                           const std::vector<size_t> &strides) {
  BroadcastLayout layout{};
  layout.rank = shape.size();
  for (size_t d = 0; d < layout.rank; d++) {
    layout.shape[d] = shape[d];
    layout.strides[d] = strides[d];
  }
  return layout;
}

__device__ size_t BroadcastOffset(const BroadcastLayout &layout,
                                  size_t linear) {
  size_t offset = 0;
  for (size_t d = layout.rank; d > 0; d--) {
    size_t dim = layout.shape[d - 1];
    offset += (linear % dim) * layout.strides[d - 1];
    linear /= dim;
  }
  return offset;
}

__global__ void BroadcastToKernel(const scalar_t *source, scalar_t *out,
                                  BroadcastLayout layout, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = source[BroadcastOffset(layout, i)];
  }
}

__global__ void BroadcastToBackwardKernel(const scalar_t *out_grad,
                                          scalar_t *source_grad,
                                          BroadcastLayout layout, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    atomicAdd(&source_grad[BroadcastOffset(layout, i)], out_grad[i]);
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

void BroadcastTo(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  const auto &shape = args.out->shape();
  std::vector<size_t> strides = BroadcastStrides(
      args.lhs->shape(), ContiguousStrides(args.lhs->shape()), shape);
  BroadcastLayout layout = MakeLayout(shape, strides);
  size_t n = args.out->size();
  BroadcastToKernel<<<GridSize(n), kBlockSize, 0,
                      CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), layout, n);
}

std::function<void()> BroadcastToBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    std::vector<size_t> strides = BroadcastStrides(
        lhs->shape(), ContiguousStrides(lhs->shape()), out->shape());
    BroadcastLayout layout = MakeLayout(out->shape(), strides);
    size_t n = out->size();
    BroadcastToBackwardKernel<<<GridSize(n), kBlockSize, 0,
                                CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), layout, n);
  };
}

}  // namespace

void RegisterBroadcastOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kBroadcastTo, Device::CUDA, BroadcastTo);
  registry.RegisterBackward(OpId::kBroadcastTo, Device::CUDA,
                            BroadcastToBackward);
}

}  // namespace micrograd::cuda::ops

#endif
