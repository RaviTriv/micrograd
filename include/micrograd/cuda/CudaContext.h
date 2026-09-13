#pragma once

#ifdef MICROGRAD_CUDA_ENABLED

#include <cuda_runtime.h>

#ifdef MICROGRAD_CUBLAS_ENABLED
#include <cublas_v2.h>
#endif

#include <cstddef>
#include <unordered_map>
#include <vector>

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

  cudaStream_t stream() const;
  int device() const;

#ifdef MICROGRAD_CUBLAS_ENABLED
  cublasHandle_t cublasHandle() const;
#endif

  void synchronize();

 private:
  CudaContext();
  ~CudaContext();

  static size_t bucketSize(size_t bytes);

  int device_ = 0;
  cudaStream_t stream_ = nullptr;
#ifdef MICROGRAD_CUBLAS_ENABLED
  cublasHandle_t cublas_handle_ = nullptr;
#endif
  std::unordered_map<size_t, std::vector<void *>> free_blocks_;
  bool initialized_ = false;
};

}  // namespace micrograd

#endif
