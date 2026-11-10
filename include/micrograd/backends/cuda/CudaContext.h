#pragma once

#ifdef MICROGRAD_CUDA_ENABLED

#include <cuda_runtime.h>

#ifdef MICROGRAD_CUBLAS_ENABLED
#include <cublasLt.h>
#include <cublas_v2.h>
#endif

#include <cstddef>
#include <string>

#include "micrograd/CachingAllocator.h"

namespace micrograd {

class CudaContext {
 public:
  static CudaContext &instance();

  CudaContext(const CudaContext &) = delete;
  CudaContext &operator=(const CudaContext &) = delete;

  bool initialize();
  void shutdown();
  bool isAvailable() const;

  void *allocate(size_t bytes);
  void deallocate(void *ptr, size_t bytes);
  CachingAllocator::Stats memoryStats() const;
  std::string memorySummary() const;

  cudaStream_t stream() const;
  int device() const;
  void setDevice(int device);

#ifdef MICROGRAD_CUBLAS_ENABLED
  cublasHandle_t cublasHandle() const;
  cublasLtHandle_t cublasLtHandle() const;
#endif

  void synchronize();

  void setAllowTf32(bool allow);
  bool allowTf32() const;
  void setAllowBf16Matmul(bool allow);
  bool allowBf16Matmul() const;

 private:
  CudaContext();
  ~CudaContext();

  int device_ = 0;
  cudaStream_t stream_ = nullptr;
#ifdef MICROGRAD_CUBLAS_ENABLED
  cublasHandle_t cublas_handle_ = nullptr;
  cublasLtHandle_t cublas_lt_handle_ = nullptr;
#endif
  CachingAllocator allocator_;
  bool initialized_ = false;
  bool allow_tf32_ = false;
  bool allow_bf16_matmul_ = false;
};

}  // namespace micrograd

#endif
