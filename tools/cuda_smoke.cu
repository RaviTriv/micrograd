#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "cuda_smoke.h"

namespace micrograd::tools {
namespace {

__global__ void iota_kernel(int* out, int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    out[i] = i;
  }
}

}  // namespace

int cudnn_version() { return static_cast<int>(cudnnGetVersion()); }

int cublaslt_version() { return static_cast<int>(cublasLtGetVersion()); }

bool cuda_device_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::vector<int> cuda_iota(int n) {
  std::vector<int> host(static_cast<size_t>(n));
  const size_t bytes = host.size() * sizeof(int);
  int* device = nullptr;
  if (cudaMalloc(&device, bytes) != cudaSuccess) {
    return {};
  }
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  iota_kernel<<<blocks, threads>>>(device, n);
  const cudaError_t launched = cudaGetLastError();
  const cudaError_t copied =
      cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost);
  cudaFree(device);
  if (launched != cudaSuccess || copied != cudaSuccess) {
    return {};
  }
  return host;
}

}  // namespace micrograd::tools
