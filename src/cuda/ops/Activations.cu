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

__global__ void ReluKernel(const scalar_t *lhs, scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = lhs[i] > 0 ? lhs[i] : 0.0f;
  }
}

__global__ void SigmoidKernel(const scalar_t *lhs, scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = 1.0f / (1.0f + expf(-lhs[i]));
  }
}

__global__ void TanhKernel(const scalar_t *lhs, scalar_t *out, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = tanhf(lhs[i]);
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
  args.out->to(Backend::CUDA);
  size_t n = args.lhs->size();
  ReluKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), n);
}

void Sigmoid(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  size_t n = args.lhs->size();
  SigmoidKernel<<<GridSize(n), kBlockSize, 0,
                  CudaContext::instance().stream()>>>(DataPtr(args.lhs),
                                                      DataPtr(args.out), n);
}

void Tanh(const OpArgs &args) {
  args.out->to(Backend::CUDA);
  size_t n = args.lhs->size();
  TanhKernel<<<GridSize(n), kBlockSize, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.out), n);
}

std::function<void()> ReluBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    ReluBackwardKernel<<<GridSize(n), kBlockSize, 0,
                         CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(lhs.get()), GradPtr(lhs.get()), n);
  };
}

std::function<void()> SigmoidBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    SigmoidBackwardKernel<<<GridSize(n), kBlockSize, 0,
                            CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(out), GradPtr(lhs.get()), n);
  };
}

std::function<void()> TanhBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs]() {
    size_t n = lhs->size();
    TanhBackwardKernel<<<GridSize(n), kBlockSize, 0,
                         CudaContext::instance().stream()>>>(
        GradPtr(out), DataPtr(out), GradPtr(lhs.get()), n);
  };
}

}  // namespace

void RegisterActivationOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kRelu, Device::CUDA, Relu);
  registry.Register(OpId::kSigmoid, Device::CUDA, Sigmoid);
  registry.Register(OpId::kTanh, Device::CUDA, Tanh);
  registry.RegisterBackward(OpId::kRelu, Device::CUDA, ReluBackward);
  registry.RegisterBackward(OpId::kSigmoid, Device::CUDA, SigmoidBackward);
  registry.RegisterBackward(OpId::kTanh, Device::CUDA, TanhBackward);
}

}  // namespace micrograd::cuda::ops

#endif
