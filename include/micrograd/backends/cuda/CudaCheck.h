#pragma once

#ifdef MICROGRAD_CUDA_ENABLED

#include <cuda_runtime.h>

#ifdef MICROGRAD_CUBLAS_ENABLED
#include <cublas_v2.h>
#endif

#include <stdexcept>
#include <string>

namespace micrograd::cuda {

[[noreturn]] inline void ThrowCudaError(const std::string &what,
                                        const char *expr, const char *file,
                                        int line) {
  throw std::runtime_error(
      what + " in `" + expr + "` at " + file + ":" + std::to_string(line) +
      ". Kernel errors can surface at a later call; rerun with "
      "CUDA_LAUNCH_BLOCKING=1 to find the launch that failed.");
}

inline void CheckCuda(cudaError_t status, const char *expr, const char *file,
                      int line) {
  if (status != cudaSuccess) {
    ThrowCudaError(std::string("CUDA error: ") + cudaGetErrorString(status),
                   expr, file, line);
  }
}

#ifdef MICROGRAD_CUBLAS_ENABLED
inline void CheckCublas(cublasStatus_t status, const char *expr,
                        const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    ThrowCudaError(
        std::string("cuBLAS error: ") + cublasGetStatusString(status), expr,
        file, line);
  }
}
#endif

}  // namespace micrograd::cuda

#define MICROGRAD_CUDA_CHECK(expr) \
  ::micrograd::cuda::CheckCuda((expr), #expr, __FILE__, __LINE__)

#define MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK() \
  MICROGRAD_CUDA_CHECK(cudaGetLastError())

#ifdef MICROGRAD_CUBLAS_ENABLED
#define MICROGRAD_CUBLAS_CHECK(expr) \
  ::micrograd::cuda::CheckCublas((expr), #expr, __FILE__, __LINE__)
#endif

#endif
