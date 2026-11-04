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

template <bool IsBackward>
__global__ void RotateHalfKernel(const scalar_t *input,
                                 const scalar_t *cos_table,
                                 const scalar_t *sin_table, scalar_t *output,
                                 size_t outer, size_t seq_len, size_t half) {
  size_t head_dim = half * 2;
  size_t total = outer * seq_len * half;
  scalar_t sign = IsBackward ? -1.0f : 1.0f;

  for (size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
    size_t j = idx % half;
    size_t ot = idx / half;
    size_t t = ot % seq_len;
    size_t o = ot / seq_len;

    const scalar_t *row = input + (((o * seq_len) + t) * head_dim);
    scalar_t *out_row = output + (((o * seq_len) + t) * head_dim);
    const scalar_t *cos_row = cos_table + (t * half);
    const scalar_t *sin_row = sin_table + (t * half);

    scalar_t x1 = row[j];
    scalar_t x2 = row[j + half];
    scalar_t y1 = (x1 * cos_row[j]) + (sign * x2 * sin_row[j]);
    scalar_t y2 = (x2 * cos_row[j]) - (sign * x1 * sin_row[j]);
    if constexpr (IsBackward) {
      out_row[j] += y1;
      out_row[j + half] += y2;
    } else {
      out_row[j] = y1;
      out_row[j + half] = y2;
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

void RotaryEmbedding(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  size_t head_dim = args.lhs->shape().back();
  size_t seq_len = args.lhs->shape()[args.lhs->shape().size() - 2];
  size_t half = head_dim / 2;
  size_t outer = args.lhs->size() / (seq_len * head_dim);
  size_t total = outer * seq_len * half;

  RotateHalfKernel<false>
      <<<GridSize(total), kBlockSize, 0, CudaContext::instance().stream()>>>(
          DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.extra),
          DataPtr(args.out), outer, seq_len, half);
}

std::function<void()> RotaryEmbeddingBackward(const GradArgs &args) {
  return [lhs = args.lhs, cos_table = args.rhs, sin_table = args.extra,
          out = args.out]() {
    size_t head_dim = lhs->shape().back();
    size_t seq_len = lhs->shape()[lhs->shape().size() - 2];
    size_t half = head_dim / 2;
    size_t outer = lhs->size() / (seq_len * head_dim);
    size_t total = outer * seq_len * half;

    RotateHalfKernel<true>
        <<<GridSize(total), kBlockSize, 0, CudaContext::instance().stream()>>>(
            GradPtr(out), DataPtr(cos_table.get()), DataPtr(sin_table.get()),
            GradPtr(lhs.get()), outer, seq_len, half);
  };
}

}  // namespace

void RegisterRotaryOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kRotaryEmbedding, Device::CUDA, RotaryEmbedding);
  registry.RegisterBackward(OpId::kRotaryEmbedding, Device::CUDA,
                            RotaryEmbeddingBackward);
}

}  // namespace micrograd::cuda::ops

#endif
