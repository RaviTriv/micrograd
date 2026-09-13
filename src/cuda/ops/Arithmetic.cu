#include "micrograd/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <algorithm>
#include <cstddef>
#include <functional>

#include "micrograd/Tensor.h"
#include "micrograd/cuda/ops/Ops.h"
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
  args.out->to(Backend::CUDA);
  size_t n = args.lhs->size();
  AddKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), n);
}

void Sub(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  size_t n = args.lhs->size();
  SubKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), n);
}

void Mul(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  size_t n = args.lhs->size();
  MulKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), n);
}

void Div(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  size_t n = args.lhs->size();
  DivKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), n);
}

void AddScalar(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  size_t n = args.lhs->size();
  AddScalarKernel<<<GridSize(n), kBlockSize, 0,
                    CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), args.scalar, n);
}

void SubScalar(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  size_t n = args.lhs->size();
  SubScalarKernel<<<GridSize(n), kBlockSize, 0,
                    CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), args.scalar, n);
}

void MulScalar(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  size_t n = args.lhs->size();
  MulScalarKernel<<<GridSize(n), kBlockSize, 0,
                    CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), args.scalar, n);
}

void DivScalar(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  size_t n = args.lhs->size();
  DivScalarKernel<<<GridSize(n), kBlockSize, 0,
                    CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), args.scalar, n);
}

void Pow(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  size_t n = args.lhs->size();
  PowKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), args.scalar, n);
}

std::function<void()> AddBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    size_t n = lhs->size();
    AddBackwardKernel<<<GridSize(n), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), GradPtr(rhs.get()), n);
  };
}

std::function<void()> SubBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    size_t n = lhs->size();
    SubBackwardKernel<<<GridSize(n), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), GradPtr(rhs.get()), n);
  };
}

std::function<void()> MulBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    size_t n = lhs->size();
    MulBackwardKernel<<<GridSize(n), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(lhs.get()), DataPtr(rhs.get()), GradPtr(lhs.get()),
        GradPtr(rhs.get()), n);
  };
}

std::function<void()> DivBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    size_t n = lhs->size();
    DivBackwardKernel<<<GridSize(n), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(lhs.get()), DataPtr(rhs.get()), GradPtr(lhs.get()),
        GradPtr(rhs.get()), n);
  };
}

std::function<void()> AddScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    AccumulateKernel<<<GridSize(n), kBlockSize, 0,
                       CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), n);
  };
}

std::function<void()> SubScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    AccumulateKernel<<<GridSize(n), kBlockSize, 0,
                       CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), n);
  };
}

std::function<void()> MulScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, scalar = args.scalar]() {
    size_t n = lhs->size();
    MulScalarBackwardKernel<<<GridSize(n), kBlockSize, 0,
                              CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), scalar, n);
  };
}

std::function<void()> DivScalarBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, scalar = args.scalar]() {
    size_t n = lhs->size();
    DivScalarBackwardKernel<<<GridSize(n), kBlockSize, 0,
                              CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), scalar, n);
  };
}

std::function<void()> PowBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, exponent = args.scalar]() {
    size_t n = lhs->size();
    PowBackwardKernel<<<GridSize(n), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(lhs.get()), GradPtr(lhs.get()), exponent, n);
  };
}

}  // namespace

void RegisterArithmeticOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kAdd, Device::CUDA, Add);
  registry.Register(OpId::kSub, Device::CUDA, Sub);
  registry.Register(OpId::kMul, Device::CUDA, Mul);
  registry.Register(OpId::kDiv, Device::CUDA, Div);
  registry.Register(OpId::kAddScalar, Device::CUDA, AddScalar);
  registry.Register(OpId::kSubScalar, Device::CUDA, SubScalar);
  registry.Register(OpId::kMulScalar, Device::CUDA, MulScalar);
  registry.Register(OpId::kDivScalar, Device::CUDA, DivScalar);
  registry.Register(OpId::kPow, Device::CUDA, Pow);
  registry.RegisterBackward(OpId::kAdd, Device::CUDA, AddBackward);
  registry.RegisterBackward(OpId::kSub, Device::CUDA, SubBackward);
  registry.RegisterBackward(OpId::kMul, Device::CUDA, MulBackward);
  registry.RegisterBackward(OpId::kDiv, Device::CUDA, DivBackward);
  registry.RegisterBackward(OpId::kAddScalar, Device::CUDA, AddScalarBackward);
  registry.RegisterBackward(OpId::kSubScalar, Device::CUDA, SubScalarBackward);
  registry.RegisterBackward(OpId::kMulScalar, Device::CUDA, MulScalarBackward);
  registry.RegisterBackward(OpId::kDivScalar, Device::CUDA, DivScalarBackward);
  registry.RegisterBackward(OpId::kPow, Device::CUDA, PowBackward);
}

}  // namespace micrograd::cuda::ops

#endif
