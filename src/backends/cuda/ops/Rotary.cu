#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
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

int GridSize(size_t n) {
  size_t blocks = (n + kBlockSize - 1) / kBlockSize;
  return static_cast<int>(std::clamp<size_t>(blocks, 1, kMaxBlocks));
}

struct RotaryLayout {
  size_t rows = 0;
  size_t seq_len = 1;
  size_t inner = 1;
  size_t half = 0;
};

RotaryLayout LayoutFor(const std::vector<size_t> &shape, int64_t seq_dim) {
  RotaryLayout layout;
  const auto seq = static_cast<size_t>(seq_dim);
  size_t outer = 1;
  for (size_t i = 0; i < seq; i++) {
    outer *= shape[i];
  }
  layout.seq_len = shape[seq];
  for (size_t i = seq + 1; i + 1 < shape.size(); i++) {
    layout.inner *= shape[i];
  }
  layout.half = shape.back() / 2;
  layout.rows = outer * layout.seq_len * layout.inner;
  return layout;
}

template <int V>
__device__ inline void LoadChunk(const float *p, float (&v)[V]) {
  for (int i = 0; i < V; i++) {
    v[i] = p[i];
  }
}

template <int V>
__device__ inline void LoadChunk(const bf16 *p, float (&v)[V]) {
  if constexpr (V == kBf16Vector) {
    const Bf16x8 x = LoadBf16x8(p);
    for (int i = 0; i < V; i++) {
      v[i] = x.v[i];
    }
  } else {
    for (int i = 0; i < V; i++) {
      v[i] = __bfloat162float(p[i]);
    }
  }
}

template <int V>
__device__ inline void StoreChunk(float *p, const float (&v)[V]) {
  for (int i = 0; i < V; i++) {
    p[i] = v[i];
  }
}

template <int V>
__device__ inline void StoreChunk(bf16 *p, const float (&v)[V]) {
  if constexpr (V == kBf16Vector) {
    Bf16x8 x;
    for (int i = 0; i < V; i++) {
      x.v[i] = v[i];
    }
    StoreBf16x8(p, x);
  } else {
    for (int i = 0; i < V; i++) {
      p[i] = __float2bfloat16(v[i]);
    }
  }
}

template <typename T, int V, bool IsBackward>
__global__ void RotateHalfKernel(const T *input, const float *cos_table,
                                 const float *sin_table, T *output,
                                 bool accumulate, size_t rows, size_t seq_len,
                                 size_t inner, size_t half) {
  const size_t chunks = half / V;
  const size_t total = rows * chunks;
  const float sign = IsBackward ? -1.0f : 1.0f;
  for (size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const size_t j = (idx % chunks) * V;
    const size_t row = idx / chunks;
    const size_t t = (row / inner) % seq_len;
    const T *in_row = input + (row * half * 2);
    T *out_row = output + (row * half * 2);
    const float *cos_row = cos_table + (t * half) + j;
    const float *sin_row = sin_table + (t * half) + j;
    float x1[V];
    float x2[V];
    LoadChunk<V>(in_row + j, x1);
    LoadChunk<V>(in_row + j + half, x2);
    float y1[V];
    float y2[V];
    if (accumulate) {
      LoadChunk<V>(out_row + j, y1);
      LoadChunk<V>(out_row + j + half, y2);
    } else {
      for (int i = 0; i < V; i++) {
        y1[i] = 0.0f;
        y2[i] = 0.0f;
      }
    }
    for (int i = 0; i < V; i++) {
      const float c = cos_row[i];
      const float s = sign * sin_row[i];
      y1[i] += (x1[i] * c) + (x2[i] * s);
      y2[i] += (x2[i] * c) - (x1[i] * s);
    }
    StoreChunk<V>(out_row + j, y1);
    StoreChunk<V>(out_row + j + half, y2);
  }
}

template <bool IsBackward, typename T>
void LaunchRotate(const T *input, const float *cos_table,
                  const float *sin_table, T *output, bool accumulate,
                  const RotaryLayout &layout) {
  cudaStream_t stream = CudaContext::instance().stream();
  if (layout.rows == 0 || layout.half == 0) {
    return;
  }
  if (layout.half % kBf16Vector == 0) {
    const size_t total = layout.rows * (layout.half / kBf16Vector);
    RotateHalfKernel<T, kBf16Vector, IsBackward>
        <<<GridSize(total), kBlockSize, 0, stream>>>(
            input, cos_table, sin_table, output, accumulate, layout.rows,
            layout.seq_len, layout.inner, layout.half);
  } else {
    const size_t total = layout.rows * layout.half;
    RotateHalfKernel<T, 1, IsBackward>
        <<<GridSize(total), kBlockSize, 0, stream>>>(
            input, cos_table, sin_table, output, accumulate, layout.rows,
            layout.seq_len, layout.inner, layout.half);
  }
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
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
  LaunchRotate<false>(DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.extra),
                      DataPtr(args.out), false,
                      LayoutFor(args.lhs->shape(), args.dim));
}

std::function<void()> RotaryEmbeddingBackward(const GradArgs &args) {
  return [lhs = args.lhs, cos_table = args.rhs, sin_table = args.extra,
          out = args.out, seq_dim = args.dim]() {
    const GradTarget target = GradForWrite(lhs.get());
    LaunchRotate<true>(static_cast<const scalar_t *>(GradPtr(out)),
                       DataPtr(cos_table.get()), DataPtr(sin_table.get()),
                       target.ptr, target.accumulate,
                       LayoutFor(lhs->shape(), seq_dim));
  };
}

void RotaryEmbeddingBf16(const OpArgs &args) {
  LaunchRotate<false>(Bf16Data(args.lhs), DataPtr(args.rhs),
                      DataPtr(args.extra), Bf16Data(args.out), false,
                      LayoutFor(args.lhs->shape(), args.dim));
}

std::function<void()> RotaryEmbeddingBackwardBf16(const GradArgs &args) {
  return [lhs = args.lhs, cos_table = args.rhs, sin_table = args.extra,
          out = args.out, seq_dim = args.dim]() {
    const Bf16GradTarget target = Bf16GradForWrite(lhs.get());
    LaunchRotate<true>(static_cast<const bf16 *>(Bf16Grad(out)),
                       DataPtr(cos_table.get()), DataPtr(sin_table.get()),
                       target.ptr, target.accumulate,
                       LayoutFor(lhs->shape(), seq_dim));
  };
}

}  // namespace

void RegisterRotaryOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kRotaryEmbedding, Device::CUDA, RotaryEmbedding);
  registry.RegisterBackward(OpId::kRotaryEmbedding, Device::CUDA,
                            RotaryEmbeddingBackward);
  registry.Register(OpId::kRotaryEmbedding, Device::CUDA, RotaryEmbeddingBf16,
                    DType::kBFloat16);
  registry.RegisterBackward(OpId::kRotaryEmbedding, Device::CUDA,
                            RotaryEmbeddingBackwardBf16, DType::kBFloat16);
}

}  // namespace micrograd::cuda::ops

#endif
