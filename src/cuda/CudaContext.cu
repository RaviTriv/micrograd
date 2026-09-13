#include "micrograd/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#include <iostream>

namespace micrograd {

CudaContext &CudaContext::instance() {
  static CudaContext context;
  return context;
}

CudaContext::CudaContext() = default;

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
  cublasSetStream(cublas_handle_, stream_);
#endif

  initialized_ = true;
  return true;
}

void CudaContext::synchronize() { cudaStreamSynchronize(stream_); }

size_t CudaContext::bucketSize(size_t bytes) {
  size_t size = 1;
  while (size < bytes) {
    size <<= 1;
  }
  return size;
}

void *CudaContext::allocate(size_t bytes) {
  size_t bucket = bucketSize(bytes);
  std::vector<void *> &blocks = free_blocks_[bucket];
  if (!blocks.empty()) {
    void *ptr = blocks.back();
    blocks.pop_back();
    return ptr;
  }

  void *ptr = nullptr;
  if (cudaMalloc(&ptr, bucket) != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

void CudaContext::deallocate(void *ptr, size_t bytes) {
  if (!ptr) {
    return;
  }
  free_blocks_[bucketSize(bytes)].push_back(ptr);
}

void CudaContext::shutdown() {
  for (auto &entry : free_blocks_) {
    for (void *ptr : entry.second) {
      cudaFree(ptr);
    }
  }
  free_blocks_.clear();

#ifdef MICROGRAD_CUBLAS_ENABLED
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
#endif

}  // namespace micrograd

#endif
