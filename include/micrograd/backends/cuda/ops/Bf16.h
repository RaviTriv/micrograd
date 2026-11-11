#pragma once

#include <cuda_bf16.h>

#include <algorithm>
#include <cstddef>

#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"
#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"

namespace micrograd::cuda::ops {

using bf16 = __nv_bfloat16;

constexpr int kBf16Vector = 8;
constexpr int kBf16BlockSize = 256;
constexpr size_t kBf16MaxBlocks = 65535;

inline int Bf16GridSize(size_t n) {
  const size_t vectors = (n + kBf16Vector - 1) / kBf16Vector;
  const size_t blocks = (vectors + kBf16BlockSize - 1) / kBf16BlockSize;
  return static_cast<int>(std::clamp<size_t>(blocks, 1, kBf16MaxBlocks));
}

struct Bf16x8 {
  float v[kBf16Vector];
};

__device__ inline Bf16x8 LoadBf16x8(const bf16 *ptr) {
  const uint4 raw = *reinterpret_cast<const uint4 *>(ptr);
  const auto *pairs = reinterpret_cast<const __nv_bfloat162 *>(&raw);
  Bf16x8 out;
  for (int i = 0; i < kBf16Vector / 2; i++) {
    const float2 f = __bfloat1622float2(pairs[i]);
    out.v[2 * i] = f.x;
    out.v[(2 * i) + 1] = f.y;
  }
  return out;
}

__device__ inline void StoreBf16x8(bf16 *ptr, const Bf16x8 &values) {
  uint4 raw;
  auto *pairs = reinterpret_cast<__nv_bfloat162 *>(&raw);
  for (int i = 0; i < kBf16Vector / 2; i++) {
    pairs[i] = __floats2bfloat162_rn(values.v[2 * i], values.v[(2 * i) + 1]);
  }
  *reinterpret_cast<uint4 *>(ptr) = raw;
}

inline const bf16 *Bf16Data(const Tensor *t) {
  return static_cast<const bf16 *>(t->data_storage().device_pointer());
}

inline bf16 *Bf16Data(Tensor *t) {
  return static_cast<bf16 *>(t->data_storage().device_pointer());
}

inline bf16 *Bf16Grad(Tensor *t) {
  return static_cast<bf16 *>(t->grad_storage().device_pointer());
}

struct Bf16GradTarget {
  bf16 *ptr = nullptr;
  bool accumulate = false;
};

inline Bf16GradTarget Bf16GradForWrite(Tensor *t) {
  const bool accumulate = t->has_grad();
  return {static_cast<bf16 *>(t->grad_storage_for_overwrite().device_pointer()),
          accumulate};
}

template <typename Op>
__global__ void Bf16MapKernel(size_t n, Op op) {
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t v = (blockIdx.x * blockDim.x) + threadIdx.x; v * kBf16Vector < n;
       v += stride) {
    const size_t base = v * kBf16Vector;
    if (base + kBf16Vector <= n) {
      op.vector(base);
    } else {
      for (size_t i = base; i < n; i++) {
        op.scalar(i);
      }
    }
  }
}

void Bf16BiasGrad(const bf16 *grad, float *bias_grad, size_t rows, size_t cols);

template <typename Op>
void LaunchBf16Map(size_t n, const Op &op) {
  if (n == 0) {
    return;
  }
  Bf16MapKernel<<<Bf16GridSize(n), kBf16BlockSize, 0,
                  CudaContext::instance().stream()>>>(n, op);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace micrograd::cuda::ops
