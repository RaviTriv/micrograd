#include "micrograd/backends/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <algorithm>
#include <cstddef>
#include <functional>

#include "micrograd/Tensor.h"
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
  args.out->to(Backend::CUDA);
  size_t dim = args.lhs->shape()[1];
  size_t count = args.rhs->size();
  EmbeddingLookupKernel<<<GridSize(count), kBlockSize, 0,
                          CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), dim, count);
}

std::function<void()> EmbeddingLookupBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, indices = args.rhs]() {
    size_t dim = lhs->shape()[1];
    size_t count = indices->size();
    EmbeddingLookupBackwardKernel<<<GridSize(count), kBlockSize, 0,
                                    CudaContext::instance().stream()>>>(
        DataPtr(indices.get()), GradPtr(out), GradPtr(lhs.get()), dim, count);
  };
}

}  // namespace

void RegisterEmbeddingOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kEmbeddingLookup, Device::CUDA, EmbeddingLookup);
  registry.RegisterBackward(OpId::kEmbeddingLookup, Device::CUDA,
                            EmbeddingLookupBackward);
}

}  // namespace micrograd::cuda::ops

#endif
