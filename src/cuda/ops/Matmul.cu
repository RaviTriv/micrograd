#include "micrograd/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#ifdef MICROGRAD_CUBLAS_ENABLED
#include <cublas_v2.h>
#endif

#include <cstddef>
#include <functional>

#include "micrograd/Tensor.h"
#include "micrograd/cuda/ops/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::cuda::ops {
namespace {

constexpr unsigned int kTile = 32;

unsigned int GridDim(size_t n) {
  return static_cast<unsigned int>((n + kTile - 1) / kTile);
}

__global__ void MatmulNNKernel(const scalar_t *a, const scalar_t *b,
                               scalar_t *c, size_t m, size_t k, size_t n) {
  __shared__ scalar_t a_tile[kTile][kTile];
  __shared__ scalar_t b_tile[kTile][kTile];

  const size_t batch_idx = blockIdx.z;
  const scalar_t *a_b = a + batch_idx * m * k;
  const scalar_t *b_b = b + batch_idx * k * n;
  scalar_t *c_b = c + batch_idx * m * n;

  const size_t row = blockIdx.y * kTile + threadIdx.y;
  const size_t col = blockIdx.x * kTile + threadIdx.x;

  scalar_t acc = 0.0f;
  const size_t tiles = (k + kTile - 1) / kTile;
  for (size_t t = 0; t < tiles; t++) {
    const size_t a_col = t * kTile + threadIdx.x;
    const size_t b_row = t * kTile + threadIdx.y;
    a_tile[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a_b[row * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b_b[b_row * n + col] : 0.0f;
    __syncthreads();

    for (size_t kk = 0; kk < kTile; kk++) {
      acc += a_tile[threadIdx.y][kk] * b_tile[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c_b[row * n + col] = acc;
  }
}

__global__ void MatmulNTKernel(const scalar_t *a, const scalar_t *b,
                               scalar_t *c, size_t m, size_t p, size_t n) {
  __shared__ scalar_t a_tile[kTile][kTile];
  __shared__ scalar_t b_tile[kTile][kTile];

  const size_t batch_idx = blockIdx.z;
  const scalar_t *a_b = a + batch_idx * m * p;
  const scalar_t *b_b = b + batch_idx * n * p;
  scalar_t *c_b = c + batch_idx * m * n;

  const size_t row = blockIdx.y * kTile + threadIdx.y;
  const size_t col = blockIdx.x * kTile + threadIdx.x;

  scalar_t acc = 0.0f;
  const size_t tiles = (p + kTile - 1) / kTile;
  for (size_t t = 0; t < tiles; t++) {
    const size_t a_col = t * kTile + threadIdx.x;
    const size_t b_col = t * kTile + threadIdx.y;
    a_tile[threadIdx.y][threadIdx.x] =
        (row < m && a_col < p) ? a_b[row * p + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        (col < n && b_col < p) ? b_b[col * p + b_col] : 0.0f;
    __syncthreads();

    for (size_t kk = 0; kk < kTile; kk++) {
      acc += a_tile[threadIdx.y][kk] * b_tile[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c_b[row * n + col] += acc;
  }
}

__global__ void MatmulTNKernel(const scalar_t *a, const scalar_t *b,
                               scalar_t *c, size_t m, size_t k, size_t n) {
  __shared__ scalar_t a_tile[kTile][kTile];
  __shared__ scalar_t b_tile[kTile][kTile];

  const size_t batch_idx = blockIdx.z;
  const scalar_t *a_b = a + batch_idx * m * k;
  const scalar_t *b_b = b + batch_idx * m * n;
  scalar_t *c_b = c + batch_idx * k * n;

  const size_t row = blockIdx.y * kTile + threadIdx.y;
  const size_t col = blockIdx.x * kTile + threadIdx.x;

  scalar_t acc = 0.0f;
  const size_t tiles = (m + kTile - 1) / kTile;
  for (size_t t = 0; t < tiles; t++) {
    const size_t a_row = t * kTile + threadIdx.x;
    const size_t b_row = t * kTile + threadIdx.y;
    a_tile[threadIdx.y][threadIdx.x] =
        (a_row < m && row < k) ? a_b[a_row * k + row] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        (b_row < m && col < n) ? b_b[b_row * n + col] : 0.0f;
    __syncthreads();

    for (size_t kk = 0; kk < kTile; kk++) {
      acc += a_tile[threadIdx.y][kk] * b_tile[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < k && col < n) {
    c_b[row * n + col] += acc;
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

#ifdef MICROGRAD_CUBLAS_ENABLED
void MatmulCublas(const OpArgs &args, size_t batch, size_t m, size_t k,
                  size_t n) {
  const scalar_t alpha = 1.0f;
  const scalar_t beta = 0.0f;
  cublasSgemmStridedBatched(
      CudaContext::instance().cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N,
      static_cast<int>(n), static_cast<int>(m), static_cast<int>(k), &alpha,
      DataPtr(args.rhs), static_cast<int>(n), static_cast<long long>(k * n),
      DataPtr(args.lhs), static_cast<int>(k), static_cast<long long>(m * k),
      &beta, DataPtr(args.out), static_cast<int>(n),
      static_cast<long long>(m * n), static_cast<int>(batch));
}
#endif

void Matmul(const OpArgs &args) {
  args.out->to(Backend::CUDA);

  const auto &lhs_shape = args.lhs->shape();
  const auto &rhs_shape = args.rhs->shape();
  const size_t rank = lhs_shape.size();
  const size_t batch = rank == 3 ? lhs_shape[0] : 1;
  const size_t m = lhs_shape[rank - 2];
  const size_t k = lhs_shape[rank - 1];
  const size_t n = rhs_shape[rank - 1];

#ifdef MICROGRAD_CUBLAS_ENABLED
  MatmulCublas(args, batch, m, k, n);
#else
  const dim3 block(kTile, kTile);
  const dim3 grid(GridDim(n), GridDim(m), static_cast<unsigned int>(batch));
  MatmulNNKernel<<<grid, block, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), m, k, n);
#endif
}

std::function<void()> MatmulBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    const auto &lhs_shape = lhs->shape();
    const auto &rhs_shape = rhs->shape();
    const size_t rank = lhs_shape.size();
    const size_t batch = rank == 3 ? lhs_shape[0] : 1;
    const size_t m = lhs_shape[rank - 2];
    const size_t k = lhs_shape[rank - 1];
    const size_t n = rhs_shape[rank - 1];

    const dim3 block(kTile, kTile);
    cudaStream_t stream = CudaContext::instance().stream();

    const dim3 grid_a(GridDim(k), GridDim(m), static_cast<unsigned int>(batch));
    MatmulNTKernel<<<grid_a, block, 0, stream>>>(
        GradPtr(out), DataPtr(rhs.get()), GradPtr(lhs.get()), m, n, k);

    const dim3 grid_b(GridDim(n), GridDim(k), static_cast<unsigned int>(batch));
    MatmulTNKernel<<<grid_b, block, 0, stream>>>(
        DataPtr(lhs.get()), GradPtr(out), GradPtr(rhs.get()), m, k, n);
  };
}

}  // namespace

void RegisterMatmulOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kMatmul, Device::CUDA, Matmul);
  registry.RegisterBackward(OpId::kMatmul, Device::CUDA, MatmulBackward);
}

}  // namespace micrograd::cuda::ops

#endif
