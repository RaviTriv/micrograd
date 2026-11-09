#include "micrograd/backends/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <iostream>

#include "micrograd/backends/cuda/CudaCheck.h"

namespace micrograd {

CudaContext &CudaContext::instance() {
  static CudaContext context;
  return context;
}

CudaContext::CudaContext()
    : allocator_(
          [](size_t bytes) -> void * {
            void *ptr = nullptr;
            if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
              cudaGetLastError();
              return nullptr;
            }
            return ptr;
          },
          [](void *ptr) { cudaFree(ptr); }) {}

CudaContext::~CudaContext() { shutdown(); }

bool CudaContext::initialize() {
  if (initialized_) {
    return true;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cerr << "NO GPU FOUND!!!!!\n";
    return false;
  }

  if (cudaSetDevice(device_) != cudaSuccess) {
    std::cerr << "Failed to set CUDA device.\n";
    return false;
  }

  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    std::cerr << "Failed to create CUDA stream.\n";
    return false;
  }

#ifdef MICROGRAD_CUBLAS_ENABLED
  if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Failed to create cuBLAS handle.\n";
    return false;
  }
  MICROGRAD_CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
  MICROGRAD_CUBLAS_CHECK(
      cublasSetMathMode(cublas_handle_, allow_tf32_ ? CUBLAS_TF32_TENSOR_OP_MATH
                                                    : CUBLAS_DEFAULT_MATH));

  if (cublasLtCreate(&cublas_lt_handle_) != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Failed to create cuBLASLt handle.\n";
    return false;
  }
#endif

  initialized_ = true;
  return true;
}

void CudaContext::synchronize() {
  MICROGRAD_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CudaContext::setAllowTf32(bool allow) {
  allow_tf32_ = allow;
#ifdef MICROGRAD_CUBLAS_ENABLED
  if (cublas_handle_) {
    MICROGRAD_CUBLAS_CHECK(
        cublasSetMathMode(cublas_handle_, allow ? CUBLAS_TF32_TENSOR_OP_MATH
                                                : CUBLAS_DEFAULT_MATH));
  }
#endif
}

bool CudaContext::allowTf32() const { return allow_tf32_; }

void *CudaContext::allocate(size_t bytes) { return allocator_.allocate(bytes); }

void CudaContext::deallocate(void *ptr, size_t /*bytes*/) {
  allocator_.deallocate(ptr);
}

CachingAllocator::Stats CudaContext::memoryStats() const {
  return allocator_.stats();
}

std::string CudaContext::memorySummary() const { return allocator_.summary(); }

void CudaContext::shutdown() {
  allocator_.releaseAll();

#ifdef MICROGRAD_CUBLAS_ENABLED
  if (cublas_lt_handle_) {
    cublasLtDestroy(cublas_lt_handle_);
  }
  cublas_lt_handle_ = nullptr;

  if (cublas_handle_) {
    cublasDestroy(cublas_handle_);
  }
  cublas_handle_ = nullptr;
#endif

  if (stream_) {
    cudaStreamDestroy(stream_);
  }

  stream_ = nullptr;
  initialized_ = false;
}

bool CudaContext::isAvailable() const {
  return initialized_ && stream_ != nullptr;
}

cudaStream_t CudaContext::stream() const { return stream_; }
int CudaContext::device() const { return device_; }

#ifdef MICROGRAD_CUBLAS_ENABLED
cublasHandle_t CudaContext::cublasHandle() const { return cublas_handle_; }
cublasLtHandle_t CudaContext::cublasLtHandle() const {
  return cublas_lt_handle_;
}
#endif

}  // namespace micrograd

#endif
