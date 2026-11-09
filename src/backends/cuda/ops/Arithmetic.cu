#include "micrograd/backends/cuda/CudaCheck.h"
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

__global__ void AddKernel(const scalar_t *lhs, const scalar_t *rhs,
                          scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = lhs[i] + rhs[i];
  }
}

__global__ void SubKernel(const scalar_t *lhs, const scalar_t *rhs,
                          scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = lhs[i] - rhs[i];
  }
}

__global__ void MulKernel(const scalar_t *lhs, const scalar_t *rhs,
                          scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = lhs[i] * rhs[i];
  }
}

__global__ void DivKernel(const scalar_t *lhs, const scalar_t *rhs,
                          scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = lhs[i] / rhs[i];
  }
}

__global__ void AddScalarKernel(const scalar_t *lhs, scalar_t *out,
                                scalar_t scalar, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = lhs[i] + scalar;
  }
}

__global__ void SubScalarKernel(const scalar_t *lhs, scalar_t *out,
                                scalar_t scalar, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = lhs[i] - scalar;
  }
}

__global__ void MulScalarKernel(const scalar_t *lhs, scalar_t *out,
                                scalar_t scalar, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = lhs[i] * scalar;
  }
}

__global__ void DivScalarKernel(const scalar_t *lhs, scalar_t *out,
                                scalar_t scalar, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = lhs[i] / scalar;
  }
}

__global__ void PowKernel(const scalar_t *lhs, scalar_t *out, scalar_t scalar,
                          size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = powf(lhs[i], scalar);
  }
}

__global__ void AddBackwardKernel(const scalar_t *out_grad, scalar_t *a_grad,
                                  scalar_t *b_grad, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    a_grad[i] += out_grad[i];
    b_grad[i] += out_grad[i];
  }
}

__global__ void SubBackwardKernel(const scalar_t *out_grad, scalar_t *a_grad,
                                  scalar_t *b_grad, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    a_grad[i] += out_grad[i];
    b_grad[i] -= out_grad[i];
  }
}

__global__ void MulBackwardKernel(const scalar_t *out_grad,
                                  const scalar_t *a_data,
                                  const scalar_t *b_data, scalar_t *a_grad,
                                  scalar_t *b_grad, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    a_grad[i] += out_grad[i] * b_data[i];
    b_grad[i] += out_grad[i] * a_data[i];
  }
}

__global__ void DivBackwardKernel(const scalar_t *out_grad,
                                  const scalar_t *a_data,
                                  const scalar_t *b_data, scalar_t *a_grad,
                                  scalar_t *b_grad, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    a_grad[i] += out_grad[i] / b_data[i];
    b_grad[i] -= out_grad[i] * a_data[i] / b_data[i] / b_data[i];
  }
}

__global__ void AccumulateKernel(const scalar_t *out_grad, scalar_t *a_grad,
                                 size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    a_grad[i] += out_grad[i];
  }
}

__global__ void AddBiasKernel(const scalar_t *lhs, const scalar_t *bias,
                              scalar_t *out, size_t n, size_t cols) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = lhs[i] + bias[i % cols];
  }
}

constexpr unsigned int kBiasCols = 32;
constexpr unsigned int kBiasRowThreads = 8;
constexpr size_t kBiasRowsPerBlock = 256;

__global__ void BiasGradKernel(const scalar_t *out_grad, scalar_t *bias_grad,
                               size_t rows, size_t cols) {
  __shared__ scalar_t partial[kBiasRowThreads][kBiasCols];
  const size_t col = (blockIdx.x * kBiasCols) + threadIdx.x;
  const size_t row_begin = blockIdx.y * kBiasRowsPerBlock;
  const size_t row_end = row_begin + kBiasRowsPerBlock < rows
                             ? row_begin + kBiasRowsPerBlock
                             : rows;

  scalar_t sum = 0.0f;
  if (col < cols) {
    for (size_t row = row_begin + threadIdx.y; row < row_end;
         row += kBiasRowThreads) {
      sum += out_grad[(row * cols) + col];
    }
  }
  partial[threadIdx.y][threadIdx.x] = sum;
  __syncthreads();

  if (threadIdx.y == 0 && col < cols) {
    scalar_t total = 0.0f;
    for (unsigned int k = 0; k < kBiasRowThreads; k++) {
      total += partial[k][threadIdx.x];
    }
    atomicAdd(&bias_grad[col], total);
  }
}

__global__ void MulScalarBackwardKernel(const scalar_t *out_grad,
                                        scalar_t *a_grad, scalar_t scalar,
                                        size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    a_grad[i] += out_grad[i] * scalar;
  }
}

__global__ void DivScalarBackwardKernel(const scalar_t *out_grad,
                                        scalar_t *a_grad, scalar_t scalar,
                                        size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    a_grad[i] += out_grad[i] / scalar;
  }
}

__global__ void PowBackwardKernel(const scalar_t *out_grad,
                                  const scalar_t *a_data, scalar_t *a_grad,
                                  scalar_t exponent, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    a_grad[i] += out_grad[i] * exponent * powf(a_data[i], exponent - 1.0f);
  }
}

__global__ void ClampMinKernel(const scalar_t *lhs, scalar_t *out,
                               scalar_t bound, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = lhs[i] < bound ? bound : lhs[i];
  }
}

__global__ void ClampMaxKernel(const scalar_t *lhs, scalar_t *out,
                               scalar_t bound, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = lhs[i] > bound ? bound : lhs[i];
  }
}

__global__ void ClampMinBackwardKernel(const scalar_t *out_grad,
                                       const scalar_t *a_data,
                                       scalar_t *a_grad, scalar_t bound,
                                       size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    if (a_data[i] >= bound) {
      a_grad[i] += out_grad[i];
    }
  }
}

__global__ void ClampMaxBackwardKernel(const scalar_t *out_grad,
                                       const scalar_t *a_data,
                                       scalar_t *a_grad, scalar_t bound,
                                       size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    if (a_data[i] <= bound) {
      a_grad[i] += out_grad[i];
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

void Add(const OpArgs &args) {
  size_t n = args.lhs->size();
  AddKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void Sub(const OpArgs &args) {
  size_t n = args.lhs->size();
  SubKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void Mul(const OpArgs &args) {
  size_t n = args.lhs->size();
  MulKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void Div(const OpArgs &args) {
  size_t n = args.lhs->size();
  DivKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void AddScalar(const OpArgs &args) {
  size_t n = args.lhs->size();
  AddScalarKernel<<<GridSize(n), kBlockSize, 0,
                    CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), args.scalar, n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void SubScalar(const OpArgs &args) {
  size_t n = args.lhs->size();
  SubScalarKernel<<<GridSize(n), kBlockSize, 0,
                    CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), args.scalar, n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void MulScalar(const OpArgs &args) {
  size_t n = args.lhs->size();
  MulScalarKernel<<<GridSize(n), kBlockSize, 0,
                    CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), args.scalar, n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void DivScalar(const OpArgs &args) {
  size_t n = args.lhs->size();
  DivScalarKernel<<<GridSize(n), kBlockSize, 0,
                    CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), args.scalar, n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void Pow(const OpArgs &args) {
  size_t n = args.lhs->size();
  PowKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), args.scalar, n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void ClampMin(const OpArgs &args) {
  size_t n = args.lhs->size();
  ClampMinKernel<<<GridSize(n), kBlockSize, 0,
                   CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), args.scalar, n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void ClampMax(const OpArgs &args) {
  size_t n = args.lhs->size();
  ClampMaxKernel<<<GridSize(n), kBlockSize, 0,
                   CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), args.scalar, n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

std::function<void()> AddBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    size_t n = lhs->size();
    AddBackwardKernel<<<GridSize(n), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), GradPtr(rhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

void AddBias(const OpArgs &args) {
  size_t n = args.lhs->size();
  size_t cols = args.rhs->size();
  AddBiasKernel<<<GridSize(n), kBlockSize, 0,
                  CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), n, cols);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

std::function<void()> AddBiasBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    cudaStream_t stream = CudaContext::instance().stream();
    size_t n = lhs->size();
    size_t cols = rhs->size();
    size_t rows = n / cols;

    AccumulateKernel<<<GridSize(n), kBlockSize, 0, stream>>>(
        GradPtr(out), GradPtr(lhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 block(kBiasCols, kBiasRowThreads);
    const dim3 grid(
        static_cast<unsigned int>((cols + kBiasCols - 1) / kBiasCols),
        static_cast<unsigned int>((rows + kBiasRowsPerBlock - 1) /
                                  kBiasRowsPerBlock));
    BiasGradKernel<<<grid, block, 0, stream>>>(GradPtr(out), GradPtr(rhs.get()),
                                               rows, cols);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> SubBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    size_t n = lhs->size();
    SubBackwardKernel<<<GridSize(n), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), GradPtr(rhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> MulBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    size_t n = lhs->size();
    MulBackwardKernel<<<GridSize(n), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(lhs.get()), DataPtr(rhs.get()), GradPtr(lhs.get()),
        GradPtr(rhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> DivBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    size_t n = lhs->size();
    DivBackwardKernel<<<GridSize(n), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(lhs.get()), DataPtr(rhs.get()), GradPtr(lhs.get()),
        GradPtr(rhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> AddScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    AccumulateKernel<<<GridSize(n), kBlockSize, 0,
                       CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> SubScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    AccumulateKernel<<<GridSize(n), kBlockSize, 0,
                       CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> MulScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, scalar = args.scalar]() {
    size_t n = lhs->size();
    MulScalarBackwardKernel<<<GridSize(n), kBlockSize, 0,
                              CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), scalar, n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> DivScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, scalar = args.scalar]() {
    size_t n = lhs->size();
    DivScalarBackwardKernel<<<GridSize(n), kBlockSize, 0,
                              CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), scalar, n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> PowBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, exponent = args.scalar]() {
    size_t n = lhs->size();
    PowBackwardKernel<<<GridSize(n), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(lhs.get()), GradPtr(lhs.get()), exponent, n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> ClampMinBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, bound = args.scalar]() {
    size_t n = lhs->size();
    ClampMinBackwardKernel<<<GridSize(n), kBlockSize, 0,
                             CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(lhs.get()), GradPtr(lhs.get()), bound, n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> ClampMaxBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, bound = args.scalar]() {
    size_t n = lhs->size();
    ClampMaxBackwardKernel<<<GridSize(n), kBlockSize, 0,
                             CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(lhs.get()), GradPtr(lhs.get()), bound, n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

}  // namespace

void RegisterArithmeticOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kAdd, Device::CUDA, Add);
  registry.Register(OpId::kAddBias, Device::CUDA, AddBias);
  registry.Register(OpId::kSub, Device::CUDA, Sub);
  registry.Register(OpId::kMul, Device::CUDA, Mul);
  registry.Register(OpId::kDiv, Device::CUDA, Div);
  registry.Register(OpId::kAddScalar, Device::CUDA, AddScalar);
  registry.Register(OpId::kSubScalar, Device::CUDA, SubScalar);
  registry.Register(OpId::kMulScalar, Device::CUDA, MulScalar);
  registry.Register(OpId::kDivScalar, Device::CUDA, DivScalar);
  registry.Register(OpId::kPow, Device::CUDA, Pow);
  registry.Register(OpId::kClampMin, Device::CUDA, ClampMin);
  registry.Register(OpId::kClampMax, Device::CUDA, ClampMax);
  registry.RegisterBackward(OpId::kAdd, Device::CUDA, AddBackward);
  registry.RegisterBackward(OpId::kAddBias, Device::CUDA, AddBiasBackward);
  registry.RegisterBackward(OpId::kSub, Device::CUDA, SubBackward);
  registry.RegisterBackward(OpId::kMul, Device::CUDA, MulBackward);
  registry.RegisterBackward(OpId::kDiv, Device::CUDA, DivBackward);
  registry.RegisterBackward(OpId::kAddScalar, Device::CUDA, AddScalarBackward);
  registry.RegisterBackward(OpId::kSubScalar, Device::CUDA, SubScalarBackward);
  registry.RegisterBackward(OpId::kMulScalar, Device::CUDA, MulScalarBackward);
  registry.RegisterBackward(OpId::kDivScalar, Device::CUDA, DivScalarBackward);
  registry.RegisterBackward(OpId::kPow, Device::CUDA, PowBackward);
  registry.RegisterBackward(OpId::kClampMin, Device::CUDA, ClampMinBackward);
  registry.RegisterBackward(OpId::kClampMax, Device::CUDA, ClampMaxBackward);
}

}  // namespace micrograd::cuda::ops

#endif
