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

__global__ void ReluKernel(const scalar_t *lhs, scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = lhs[i] <= 0 ? 0.0f : lhs[i];
  }
}

__global__ void SigmoidKernel(const scalar_t *lhs, scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    scalar_t z = expf(-fabsf(lhs[i]));
    out[i] = lhs[i] >= 0 ? 1.0f / (1.0f + z) : z / (1.0f + z);
  }
}

__global__ void TanhKernel(const scalar_t *lhs, scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = tanhf(lhs[i]);
  }
}

constexpr scalar_t kGeluSqrt2OverPi = 0.7978845608028654f;
constexpr scalar_t kGeluCubicCoeff = 0.044715f;

__global__ void GeluKernel(const scalar_t *lhs, scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const scalar_t x = lhs[i];
    const scalar_t t =
        tanhf(kGeluSqrt2OverPi * (x + (kGeluCubicCoeff * x * x * x)));
    out[i] = 0.5f * x * (1.0f + t);
  }
}

__global__ void GeluBackwardKernel(const scalar_t *out_grad,
                                   const scalar_t *in_data, scalar_t *a_grad,
                                   size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const scalar_t x = in_data[i];
    const scalar_t t =
        tanhf(kGeluSqrt2OverPi * (x + (kGeluCubicCoeff * x * x * x)));
    const scalar_t du_dx =
        kGeluSqrt2OverPi * (1.0f + (3.0f * kGeluCubicCoeff * x * x));
    const scalar_t dy_dx =
        (0.5f * (1.0f + t)) + (0.5f * x * (1.0f - (t * t)) * du_dx);
    a_grad[i] += out_grad[i] * dy_dx;
  }
}

__global__ void ExpKernel(const scalar_t *lhs, scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = expf(lhs[i]);
  }
}

__global__ void LogKernel(const scalar_t *lhs, scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = logf(lhs[i]);
  }
}

__global__ void SqrtKernel(const scalar_t *lhs, scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = sqrtf(lhs[i]);
  }
}

__global__ void NegKernel(const scalar_t *lhs, scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = -lhs[i];
  }
}

__global__ void ReluBackwardKernel(const scalar_t *out_grad,
                                   const scalar_t *a_data, scalar_t *a_grad,
                                   size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    a_grad[i] += out_grad[i] * (a_data[i] > 0 ? 1.0f : 0.0f);
  }
}

__global__ void SigmoidBackwardKernel(const scalar_t *out_grad,
                                      const scalar_t *out_data,
                                      scalar_t *a_grad, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    scalar_t sigmoid_val = out_data[i];
    a_grad[i] += out_grad[i] * sigmoid_val * (1.0f - sigmoid_val);
  }
}

__global__ void TanhBackwardKernel(const scalar_t *out_grad,
                                   const scalar_t *out_data, scalar_t *a_grad,
                                   size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    scalar_t tanh_val = out_data[i];
    a_grad[i] += out_grad[i] * (1.0f - tanh_val * tanh_val);
  }
}

__global__ void ExpBackwardKernel(const scalar_t *out_grad,
                                  const scalar_t *out_data, scalar_t *a_grad,
                                  size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    a_grad[i] += out_grad[i] * out_data[i];
  }
}

__global__ void LogBackwardKernel(const scalar_t *out_grad,
                                  const scalar_t *a_data, scalar_t *a_grad,
                                  size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    a_grad[i] += out_grad[i] / a_data[i];
  }
}

__global__ void SqrtBackwardKernel(const scalar_t *out_grad,
                                   const scalar_t *out_data, scalar_t *a_grad,
                                   size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    a_grad[i] += out_grad[i] * 0.5f / out_data[i];
  }
}

__global__ void NegBackwardKernel(const scalar_t *out_grad, scalar_t *a_grad,
                                  size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    a_grad[i] -= out_grad[i];
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

void Relu(const OpArgs &args) {
  size_t n = args.lhs->size();
  ReluKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void Sigmoid(const OpArgs &args) {
  size_t n = args.lhs->size();
  SigmoidKernel<<<GridSize(n), kBlockSize, 0,
                  CudaContext::instance().stream()>>>(DataPtr(args.lhs),
                                                      DataPtr(args.out), n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void Tanh(const OpArgs &args) {
  size_t n = args.lhs->size();
  TanhKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void Gelu(const OpArgs &args) {
  size_t n = args.lhs->size();
  GeluKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void Exp(const OpArgs &args) {
  size_t n = args.lhs->size();
  ExpKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void Log(const OpArgs &args) {
  size_t n = args.lhs->size();
  LogKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void Sqrt(const OpArgs &args) {
  size_t n = args.lhs->size();
  SqrtKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

void Neg(const OpArgs &args) {
  size_t n = args.lhs->size();
  NegKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
}

std::function<void()> ReluBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    ReluBackwardKernel<<<GridSize(n), kBlockSize, 0,
                         CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(lhs.get()), GradPtr(lhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> SigmoidBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    SigmoidBackwardKernel<<<GridSize(n), kBlockSize, 0,
                            CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(out), GradPtr(lhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> TanhBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    TanhBackwardKernel<<<GridSize(n), kBlockSize, 0,
                         CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(out), GradPtr(lhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> GeluBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    GeluBackwardKernel<<<GridSize(n), kBlockSize, 0,
                         CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(lhs.get()), GradPtr(lhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> ExpBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    ExpBackwardKernel<<<GridSize(n), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(out), GradPtr(lhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> LogBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    LogBackwardKernel<<<GridSize(n), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(lhs.get()), GradPtr(lhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> SqrtBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    SqrtBackwardKernel<<<GridSize(n), kBlockSize, 0,
                         CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(out), GradPtr(lhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

std::function<void()> NegBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    NegBackwardKernel<<<GridSize(n), kBlockSize, 0,
                        CudaContext::instance().stream()>>>(
        GradPtr(out), GradPtr(lhs.get()), n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  };
}

}  // namespace

void RegisterActivationOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kRelu, Device::CUDA, Relu);
  registry.Register(OpId::kSigmoid, Device::CUDA, Sigmoid);
  registry.Register(OpId::kTanh, Device::CUDA, Tanh);
  registry.Register(OpId::kGelu, Device::CUDA, Gelu);
  registry.Register(OpId::kExp, Device::CUDA, Exp);
  registry.Register(OpId::kLog, Device::CUDA, Log);
  registry.Register(OpId::kSqrt, Device::CUDA, Sqrt);
  registry.Register(OpId::kNeg, Device::CUDA, Neg);
  registry.RegisterBackward(OpId::kRelu, Device::CUDA, ReluBackward);
  registry.RegisterBackward(OpId::kSigmoid, Device::CUDA, SigmoidBackward);
  registry.RegisterBackward(OpId::kTanh, Device::CUDA, TanhBackward);
  registry.RegisterBackward(OpId::kGelu, Device::CUDA, GeluBackward);
  registry.RegisterBackward(OpId::kExp, Device::CUDA, ExpBackward);
  registry.RegisterBackward(OpId::kLog, Device::CUDA, LogBackward);
  registry.RegisterBackward(OpId::kSqrt, Device::CUDA, SqrtBackward);
  registry.RegisterBackward(OpId::kNeg, Device::CUDA, NegBackward);
}

}  // namespace micrograd::cuda::ops

#endif
