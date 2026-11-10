#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"

#ifdef MICROGRAD_CUDA_ENABLED

#ifdef MICROGRAD_CUBLAS_ENABLED
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#endif

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <type_traits>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cuda/ops/Bf16.h"
#include "micrograd/backends/cuda/ops/Gemm.h"
#include "micrograd/backends/cuda/ops/Grad.h"
#include "micrograd/backends/cuda/ops/Ops.h"
#include "micrograd/ops/Dispatch.h"
#include "micrograd/ops/Matmul.h"

namespace micrograd::cuda::ops {
namespace {

using micrograd::ops::ResolveMatmulDims;

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

#ifdef MICROGRAD_CUBLAS_ENABLED
constexpr int kCastBlockSize = 256;
constexpr size_t kCastMaxBlocks = 65535;

__global__ void CastToBf16Kernel(const scalar_t *in, __nv_bfloat16 *out,
                                 size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = __float2bfloat16(in[i]);
  }
}

class Bf16Copy {
 public:
  Bf16Copy(const scalar_t *source, size_t count)
      : bytes_(count * sizeof(__nv_bfloat16)),
        ptr_(static_cast<__nv_bfloat16 *>(
            CudaContext::instance().allocate(bytes_))) {
    const size_t blocks = std::min(
        (count + kCastBlockSize - 1) / kCastBlockSize, kCastMaxBlocks);
    CastToBf16Kernel<<<static_cast<unsigned int>(blocks), kCastBlockSize, 0,
                       CudaContext::instance().stream()>>>(source, ptr_,
                                                           count);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
  }
  ~Bf16Copy() { CudaContext::instance().deallocate(ptr_, bytes_); }
  Bf16Copy(const Bf16Copy &) = delete;
  Bf16Copy &operator=(const Bf16Copy &) = delete;
  Bf16Copy(Bf16Copy &&) = delete;
  Bf16Copy &operator=(Bf16Copy &&) = delete;
  const __nv_bfloat16 *get() const { return ptr_; }

 private:
  size_t bytes_;
  __nv_bfloat16 *ptr_;
};

size_t OperandExtent(cublasOperation_t trans, int rows, int cols, int ld,
                     long long stride, int batch) {
  const auto stored_cols =
      static_cast<size_t>(trans == CUBLAS_OP_N ? cols : rows);
  const auto stored_rows =
      static_cast<size_t>(trans == CUBLAS_OP_N ? rows : cols);
  return (static_cast<size_t>(batch - 1) * static_cast<size_t>(stride)) +
         ((stored_cols - 1) * static_cast<size_t>(ld)) + stored_rows;
}

bool UseBf16Matmul() { return CudaContext::instance().allowBf16Matmul(); }

void SgemmStridedBatched(cublasHandle_t handle, cublasOperation_t trans_a,
                         cublasOperation_t trans_b, int m, int n, int k,
                         const scalar_t *alpha, const scalar_t *a, int lda,
                         long long stride_a, const scalar_t *b, int ldb,
                         long long stride_b, const scalar_t *beta, scalar_t *c,
                         int ldc, long long stride_c, int batch,
                         const __nv_bfloat16 *a_bf16 = nullptr,
                         const __nv_bfloat16 *b_bf16 = nullptr) {
  if (!UseBf16Matmul()) {
    MICROGRAD_CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, trans_a, trans_b, m, n, k, alpha, a, lda, stride_a, b, ldb,
        stride_b, beta, c, ldc, stride_c, batch));
    return;
  }
  std::optional<Bf16Copy> a_copy;
  std::optional<Bf16Copy> b_copy;
  if (a_bf16 == nullptr) {
    a_copy.emplace(a, OperandExtent(trans_a, m, k, lda, stride_a, batch));
    a_bf16 = a_copy->get();
  }
  if (b_bf16 == nullptr) {
    b_copy.emplace(b, OperandExtent(trans_b, k, n, ldb, stride_b, batch));
    b_bf16 = b_copy->get();
  }
  MICROGRAD_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, trans_a, trans_b, m, n, k, alpha, a_bf16, CUDA_R_16BF, lda,
      stride_a, b_bf16, CUDA_R_16BF, ldb, stride_b, beta, c, CUDA_R_32F, ldc,
      stride_c, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void MaybeBf16(std::optional<Bf16Copy> &copy, const scalar_t *source,
               size_t count) {
  if (UseBf16Matmul()) {
    copy.emplace(source, count);
  }
}

const __nv_bfloat16 *Bf16Ptr(const std::optional<Bf16Copy> &copy) {
  return copy ? copy->get() : nullptr;
}
#endif

constexpr unsigned int kGemmTile = 16;

__global__ void StridedGemmKernel(GemmShape shape, scalar_t alpha,
                                  GemmOperand a, GemmOperand b, scalar_t beta,
                                  GemmOutput c) {
  const size_t z = blockIdx.z;
  const size_t outer = z / shape.inner;
  const size_t inner = z % shape.inner;
  const size_t row = (blockIdx.y * kGemmTile) + threadIdx.y;
  const size_t col = (blockIdx.x * kGemmTile) + threadIdx.x;
  if (row >= shape.m || col >= shape.n) {
    return;
  }

  const scalar_t *a_base =
      a.ptr + (outer * a.outer_stride) + (inner * a.inner_stride);
  const scalar_t *b_base =
      b.ptr + (outer * b.outer_stride) + (inner * b.inner_stride);
  scalar_t acc = 0.0f;
  for (size_t l = 0; l < shape.k; l++) {
    const scalar_t a_value =
        a.trans ? a_base[(l * a.ld) + row] : a_base[(row * a.ld) + l];
    const scalar_t b_value =
        b.trans ? b_base[(col * b.ld) + l] : b_base[(l * b.ld) + col];
    acc += a_value * b_value;
  }

  scalar_t *out = c.ptr + (outer * c.outer_stride) + (inner * c.inner_stride) +
                  (row * c.ld) + col;
  *out = beta == 0.0f ? alpha * acc : (alpha * acc) + (beta * *out);
}

}  // namespace

void StridedGemm(const GemmShape &shape, scalar_t alpha, const GemmOperand &a,
                 const GemmOperand &b, scalar_t beta, const GemmOutput &c) {
  if (shape.m == 0 || shape.n == 0 || shape.outer == 0 || shape.inner == 0) {
    return;
  }
#ifdef MICROGRAD_CUBLAS_ENABLED
  cublasHandle_t handle = CudaContext::instance().cublasHandle();
  const auto offset = [](const GemmOperand &operand, size_t o) {
    return operand.bf16_copy == nullptr
               ? nullptr
               : static_cast<const __nv_bfloat16 *>(operand.bf16_copy) +
                     (o * operand.outer_stride);
  };
  for (size_t o = 0; o < shape.outer; o++) {
    SgemmStridedBatched(
        handle, b.trans ? CUBLAS_OP_T : CUBLAS_OP_N,
        a.trans ? CUBLAS_OP_T : CUBLAS_OP_N, static_cast<int>(shape.n),
        static_cast<int>(shape.m), static_cast<int>(shape.k), &alpha,
        b.ptr + (o * b.outer_stride), static_cast<int>(b.ld),
        static_cast<long long>(b.inner_stride), a.ptr + (o * a.outer_stride),
        static_cast<int>(a.ld), static_cast<long long>(a.inner_stride), &beta,
        c.ptr + (o * c.outer_stride), static_cast<int>(c.ld),
        static_cast<long long>(c.inner_stride), static_cast<int>(shape.inner),
        offset(b, o), offset(a, o));
  }
#else
  const dim3 block(kGemmTile, kGemmTile);
  const dim3 grid(
      static_cast<unsigned int>((shape.n + kGemmTile - 1) / kGemmTile),
      static_cast<unsigned int>((shape.m + kGemmTile - 1) / kGemmTile),
      static_cast<unsigned int>(shape.outer * shape.inner));
  StridedGemmKernel<<<grid, block, 0, CudaContext::instance().stream()>>>(
      shape, alpha, a, b, beta, c);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

namespace {

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
  SgemmStridedBatched(
      CudaContext::instance().cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N,
      static_cast<int>(n), static_cast<int>(m), static_cast<int>(k), &alpha,
      DataPtr(args.rhs), static_cast<int>(n), static_cast<long long>(k * n),
      DataPtr(args.lhs), static_cast<int>(k), static_cast<long long>(m * k),
      &beta, DataPtr(args.out), static_cast<int>(n),
      static_cast<long long>(m * n), static_cast<int>(batch));
}

void MatmulBackwardCublas(const scalar_t *grad_out, const scalar_t *lhs,
                          const scalar_t *rhs, GradTarget grad_lhs,
                          GradTarget grad_rhs, size_t batch, size_t m, size_t k,
                          size_t n) {
  cublasHandle_t handle = CudaContext::instance().cublasHandle();
  const scalar_t alpha = 1.0f;
  const scalar_t beta_lhs = grad_lhs.accumulate ? 1.0f : 0.0f;
  const scalar_t beta_rhs = grad_rhs.accumulate ? 1.0f : 0.0f;
  const int mi = static_cast<int>(m);
  const int ki = static_cast<int>(k);
  const int ni = static_cast<int>(n);
  const int batch_count = static_cast<int>(batch);
  const auto stride_lhs = static_cast<long long>(m * k);
  const auto stride_rhs = static_cast<long long>(k * n);
  const auto stride_out = static_cast<long long>(m * n);
  std::optional<Bf16Copy> grad_out_bf16;
  MaybeBf16(grad_out_bf16, grad_out, batch * m * n);

  if (grad_lhs.ptr != nullptr) {
    std::optional<Bf16Copy> rhs_bf16;
    MaybeBf16(rhs_bf16, rhs, batch * k * n);
    SgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, ki, mi, ni, &alpha,
                        rhs, ni, stride_rhs, grad_out, ni, stride_out,
                        &beta_lhs, grad_lhs.ptr, ki, stride_lhs, batch_count,
                        Bf16Ptr(rhs_bf16), Bf16Ptr(grad_out_bf16));
  }

  if (grad_rhs.ptr != nullptr) {
    std::optional<Bf16Copy> lhs_bf16;
    MaybeBf16(lhs_bf16, lhs, batch * m * k);
    SgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, ni, ki, mi, &alpha,
                        grad_out, ni, stride_out, lhs, ki, stride_lhs,
                        &beta_rhs, grad_rhs.ptr, ni, stride_rhs, batch_count,
                        Bf16Ptr(grad_out_bf16), Bf16Ptr(lhs_bf16));
  }
}

#endif

constexpr unsigned int kBf16GemmTile = 16;

__device__ inline float ToFloat(float value) { return value; }
__device__ inline float ToFloat(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
__device__ inline void StoreOut(float *out, float value) { *out = value; }
__device__ inline void StoreOut(__nv_bfloat16 *out, float value) {
  *out = __float2bfloat16(value);
}

template <typename Out>
__global__ void GemmBf16Kernel(bool trans_a, bool trans_b, size_t m, size_t n,
                               size_t k, const __nv_bfloat16 *a, size_t lda,
                               size_t stride_a, const __nv_bfloat16 *b,
                               size_t ldb, size_t stride_b, bool accumulate,
                               Out *c, size_t ldc, size_t stride_c) {
  const size_t row = (blockIdx.x * kBf16GemmTile) + threadIdx.x;
  const size_t col = (blockIdx.y * kBf16GemmTile) + threadIdx.y;
  if (row >= m || col >= n) {
    return;
  }
  const __nv_bfloat16 *a_batch = a + (blockIdx.z * stride_a);
  const __nv_bfloat16 *b_batch = b + (blockIdx.z * stride_b);
  float acc = 0.0f;
  for (size_t l = 0; l < k; l++) {
    const float a_value = __bfloat162float(
        trans_a ? a_batch[l + (row * lda)] : a_batch[row + (l * lda)]);
    const float b_value = __bfloat162float(
        trans_b ? b_batch[col + (l * ldb)] : b_batch[l + (col * ldb)]);
    acc += a_value * b_value;
  }
  Out *out = c + (blockIdx.z * stride_c) + row + (col * ldc);
  StoreOut(out, accumulate ? acc + ToFloat(*out) : acc);
}

template <typename Out>
void GemmBf16(bool trans_a, bool trans_b, size_t m, size_t n, size_t k,
              const __nv_bfloat16 *a, size_t lda, size_t stride_a,
              const __nv_bfloat16 *b, size_t ldb, size_t stride_b,
              bool accumulate, Out *c, size_t ldc, size_t stride_c,
              size_t batch) {
  if (m == 0 || n == 0 || batch == 0) {
    return;
  }
#ifdef MICROGRAD_CUBLAS_ENABLED
  const scalar_t alpha = 1.0f;
  const scalar_t beta = accumulate ? 1.0f : 0.0f;
  const cudaDataType_t c_type =
      std::is_same_v<Out, float> ? CUDA_R_32F : CUDA_R_16BF;
  MICROGRAD_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
      CudaContext::instance().cublasHandle(),
      trans_a ? CUBLAS_OP_T : CUBLAS_OP_N, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
      static_cast<int>(m),
      static_cast<int>(n), static_cast<int>(k), &alpha, a, CUDA_R_16BF,
      static_cast<int>(lda), static_cast<long long>(stride_a), b, CUDA_R_16BF,
      static_cast<int>(ldb), static_cast<long long>(stride_b), &beta, c,
      c_type, static_cast<int>(ldc), static_cast<long long>(stride_c),
      static_cast<int>(batch), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
  const dim3 block(kBf16GemmTile, kBf16GemmTile);
  const dim3 grid(
      static_cast<unsigned int>((m + kBf16GemmTile - 1) / kBf16GemmTile),
      static_cast<unsigned int>((n + kBf16GemmTile - 1) / kBf16GemmTile),
      static_cast<unsigned int>(batch));
  GemmBf16Kernel<<<grid, block, 0, CudaContext::instance().stream()>>>(
      trans_a, trans_b, m, n, k, a, lda, stride_a, b, ldb, stride_b, accumulate,
      c, ldc, stride_c);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

void MatmulBf16(const OpArgs &args) {
  const auto [batch, m, k, n] =
      ResolveMatmulDims(args.lhs->shape(), args.rhs->shape());
  GemmBf16(false, false, n, m, k, Bf16Data(args.rhs), n, k * n,
           Bf16Data(args.lhs), k, m * k, false, Bf16Data(args.out), n,
           m * n, batch);
}

std::function<void()> MatmulBackwardBf16(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    const auto [batch, m, k, n] =
        ResolveMatmulDims(lhs->shape(), rhs->shape());
    const __nv_bfloat16 *grad_out = Bf16Grad(out);
    const __nv_bfloat16 *lhs_data =
        Bf16Data(static_cast<const Tensor *>(lhs.get()));
    const __nv_bfloat16 *rhs_data =
        Bf16Data(static_cast<const Tensor *>(rhs.get()));
    if (lhs->requires_grad()) {
      const Bf16GradTarget grad_lhs = Bf16GradForWrite(lhs.get());
      GemmBf16(true, false, k, m, n, rhs_data, n, k * n, grad_out,
               n, m * n, grad_lhs.accumulate, grad_lhs.ptr, k, m * k, batch);
    }
    if (rhs->requires_grad()) {
      const Bf16GradTarget grad_rhs = Bf16GradForWrite(rhs.get());
      GemmBf16(false, true, n, k, m, grad_out, n, m * n, lhs_data,
               k, m * k, grad_rhs.accumulate, grad_rhs.ptr, n, k * n, batch);
    }
  };
}

struct AddBiasInPlaceBf16Op {
  __nv_bfloat16 *out;
  const float *bias;
  size_t cols;
  __device__ void vector(size_t i) const {
    Bf16x8 values = LoadBf16x8(out + i);
    for (int j = 0; j < kBf16Vector; j++) {
      values.v[j] += bias[(i + j) % cols];
    }
    StoreBf16x8(out + i, values);
  }
  __device__ void scalar(size_t i) const {
    out[i] = __float2bfloat16(__bfloat162float(out[i]) + bias[i % cols]);
  }
};

const __nv_bfloat16 *LinearWeight(const Tensor *weight, const Tensor *copy) {
  return weight->dtype() == DType::kBFloat16 ? Bf16Data(weight)
                                             : Bf16Data(copy);
}

void LinearBf16(const OpArgs &args) {
  const size_t k = args.lhs->shape().back();
  const size_t m = args.lhs->size() / k;
  const size_t n = args.rhs->shape()[1];
  const __nv_bfloat16 *weight = LinearWeight(args.rhs, args.aux);
  GemmBf16(false, false, n, m, k, weight, n, 0, Bf16Data(args.lhs), k, 0,
           false, Bf16Data(args.out), n, 0, 1);
  LaunchBf16Map(m * n, AddBiasInPlaceBf16Op{Bf16Data(args.out),
                                             DataPtr(args.extra), n});
}

std::function<void()> LinearBackwardBf16(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs, bias = args.extra,
          copy = args.aux]() {
    const size_t k = lhs->shape().back();
    const size_t m = lhs->size() / k;
    const size_t n = rhs->shape()[1];
    const __nv_bfloat16 *grad_out = Bf16Grad(out);
    const __nv_bfloat16 *weight = LinearWeight(rhs.get(), copy.get());
    if (lhs->requires_grad()) {
      const Bf16GradTarget grad_lhs = Bf16GradForWrite(lhs.get());
      GemmBf16(true, false, k, m, n, weight, n, 0, grad_out, n, 0,
               grad_lhs.accumulate, grad_lhs.ptr, k, 0, 1);
    }
    const __nv_bfloat16 *input =
        Bf16Data(static_cast<const Tensor *>(lhs.get()));
    if (rhs->requires_grad() && rhs->dtype() == DType::kFloat32) {
      const GradTarget grad_rhs = GradForWrite(rhs.get());
      GemmBf16(false, true, n, k, m, grad_out, n, 0, input, k, 0,
               grad_rhs.accumulate, grad_rhs.ptr, n, 0, 1);
    } else if (rhs->requires_grad()) {
      const Bf16GradTarget grad_rhs = Bf16GradForWrite(rhs.get());
      GemmBf16(false, true, n, k, m, grad_out, n, 0, input, k, 0,
               grad_rhs.accumulate, grad_rhs.ptr, n, 0, 1);
    }
    if (bias->requires_grad()) {
      Bf16BiasGrad(grad_out, GradPtr(bias.get()), m, n);
    }
  };
}

void MatmulNTBf16(const OpArgs &args) {
  const size_t k = args.lhs->shape().back();
  const size_t m = args.lhs->size() / k;
  const size_t n = args.rhs->shape()[0];
  GemmBf16(true, false, n, m, k, LinearWeight(args.rhs, args.aux), k, 0,
           Bf16Data(args.lhs), k, 0, false, Bf16Data(args.out), n, 0, 1);
}

std::function<void()> MatmulNTBackwardBf16(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs, copy = args.aux]() {
    const size_t k = lhs->shape().back();
    const size_t m = lhs->size() / k;
    const size_t n = rhs->shape()[0];
    const __nv_bfloat16 *grad_out = Bf16Grad(out);
    const __nv_bfloat16 *lhs_data =
        Bf16Data(static_cast<const Tensor *>(lhs.get()));
    const __nv_bfloat16 *rhs_data = LinearWeight(rhs.get(), copy.get());
    if (lhs->requires_grad()) {
      const Bf16GradTarget grad_lhs = Bf16GradForWrite(lhs.get());
      GemmBf16(false, false, k, m, n, rhs_data, k, 0, grad_out, n, 0,
               grad_lhs.accumulate, grad_lhs.ptr, k, 0, 1);
    }
    if (rhs->requires_grad() && rhs->dtype() == DType::kFloat32) {
      const GradTarget grad_rhs = GradForWrite(rhs.get());
      GemmBf16(false, true, k, n, m, lhs_data, k, 0, grad_out, n, 0,
               grad_rhs.accumulate, grad_rhs.ptr, k, 0, 1);
    } else if (rhs->requires_grad()) {
      const Bf16GradTarget grad_rhs = Bf16GradForWrite(rhs.get());
      GemmBf16(false, true, k, n, m, lhs_data, k, 0, grad_out, n, 0,
               grad_rhs.accumulate, grad_rhs.ptr, k, 0, 1);
    }
  };
}

void Matmul(const OpArgs &args) {
  const auto &lhs_shape = args.lhs->shape();
  const auto &rhs_shape = args.rhs->shape();
  const auto [batch, m, k, n] = ResolveMatmulDims(lhs_shape, rhs_shape);

#ifdef MICROGRAD_CUBLAS_ENABLED
  MatmulCublas(args, batch, m, k, n);
#else
  const dim3 block(kTile, kTile);
  const dim3 grid(GridDim(n), GridDim(m), static_cast<unsigned int>(batch));
  MatmulNNKernel<<<grid, block, 0, CudaContext::instance().stream()>>>(
      DataPtr(args.lhs), DataPtr(args.rhs), DataPtr(args.out), m, k, n);
  MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

std::function<void()> MatmulBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    const auto &lhs_shape = lhs->shape();
    const auto &rhs_shape = rhs->shape();
    const auto [batch, m, k, n] = ResolveMatmulDims(lhs_shape, rhs_shape);

#ifdef MICROGRAD_CUBLAS_ENABLED
    const scalar_t *grad_out = GradPtr(out);
    const GradTarget grad_lhs =
        lhs->requires_grad() ? GradForWrite(lhs.get()) : GradTarget{};
    const GradTarget grad_rhs =
        rhs->requires_grad() && rhs.get() != lhs.get() ? GradForWrite(rhs.get())
                                                       : GradTarget{};
    MatmulBackwardCublas(grad_out, DataPtr(lhs.get()), DataPtr(rhs.get()),
                         grad_lhs, grad_rhs, batch, m, k, n);
    if (rhs.get() == lhs.get() && rhs->requires_grad()) {
      MatmulBackwardCublas(grad_out, DataPtr(lhs.get()), DataPtr(rhs.get()),
                           GradTarget{}, GradForWrite(rhs.get()), batch, m, k,
                           n);
    }
#else
    const dim3 block(kTile, kTile);
    cudaStream_t stream = CudaContext::instance().stream();

    const dim3 grid_a(GridDim(k), GridDim(m), static_cast<unsigned int>(batch));
    MatmulNTKernel<<<grid_a, block, 0, stream>>>(
        GradPtr(out), DataPtr(rhs.get()), GradPtr(lhs.get()), m, n, k);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 grid_b(GridDim(n), GridDim(k), static_cast<unsigned int>(batch));
    MatmulTNKernel<<<grid_b, block, 0, stream>>>(
        DataPtr(lhs.get()), GradPtr(out), GradPtr(rhs.get()), m, k, n);
    MICROGRAD_CUDA_KERNEL_LAUNCH_CHECK();
#endif
  };
}

GemmShape NTShape(const Tensor &lhs, const Tensor &rhs) {
  const size_t k = lhs.shape().back();
  return {.m = lhs.size() / k, .n = rhs.shape()[0], .k = k};
}

void MatmulNT(const OpArgs &args) {
  const GemmShape shape = NTShape(*args.lhs, *args.rhs);
  StridedGemm(shape, 1.0f, {.ptr = DataPtr(args.lhs), .ld = shape.k},
              {.ptr = DataPtr(args.rhs), .trans = true, .ld = shape.k}, 0.0f,
              {.ptr = DataPtr(args.out), .ld = shape.n});
}

std::function<void()> MatmulNTBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    const GemmShape shape = NTShape(*lhs, *rhs);
    const scalar_t *grad_out = GradPtr(out);
    const scalar_t *lhs_data = DataPtr(static_cast<const Tensor *>(lhs.get()));
    const scalar_t *rhs_data = DataPtr(static_cast<const Tensor *>(rhs.get()));
#ifdef MICROGRAD_CUBLAS_ENABLED
    std::optional<Bf16Copy> grad_out_bf16;
    MaybeBf16(grad_out_bf16, grad_out, shape.m * shape.n);
    const void *grad_out_copy = Bf16Ptr(grad_out_bf16);
#else
    const void *grad_out_copy = nullptr;
#endif
    if (lhs->requires_grad()) {
      const GradTarget grad_lhs = GradForWrite(lhs.get());
      StridedGemm({.m = shape.m, .n = shape.k, .k = shape.n}, 1.0f,
                  {.ptr = grad_out, .ld = shape.n, .bf16_copy = grad_out_copy},
                  {.ptr = rhs_data, .ld = shape.k},
                  grad_lhs.accumulate ? 1.0f : 0.0f,
                  {.ptr = grad_lhs.ptr, .ld = shape.k});
    }
    if (rhs->requires_grad()) {
      const GradTarget grad_rhs = GradForWrite(rhs.get());
      StridedGemm({.m = shape.n, .n = shape.k, .k = shape.m}, 1.0f,
                  {.ptr = grad_out,
                   .trans = true,
                   .ld = shape.n,
                   .bf16_copy = grad_out_copy},
                  {.ptr = lhs_data, .ld = shape.k},
                  grad_rhs.accumulate ? 1.0f : 0.0f,
                  {.ptr = grad_rhs.ptr, .ld = shape.k});
    }
  };
}

}  // namespace

void RegisterMatmulOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kMatmul, Device::CUDA, Matmul);
  registry.RegisterBackward(OpId::kMatmul, Device::CUDA, MatmulBackward);
  registry.Register(OpId::kMatmulNT, Device::CUDA, MatmulNT);
  registry.RegisterBackward(OpId::kMatmulNT, Device::CUDA, MatmulNTBackward);
  registry.Register(OpId::kMatmul, Device::CUDA, MatmulBf16, DType::kBFloat16);
  registry.RegisterBackward(OpId::kMatmul, Device::CUDA, MatmulBackwardBf16,
                            DType::kBFloat16);
  registry.Register(OpId::kMatmulNT, Device::CUDA, MatmulNTBf16,
                    DType::kBFloat16);
  registry.RegisterBackward(OpId::kMatmulNT, Device::CUDA,
                            MatmulNTBackwardBf16, DType::kBFloat16);
  registry.Register(OpId::kLinear, Device::CUDA, LinearBf16, DType::kBFloat16);
  registry.RegisterBackward(OpId::kLinear, Device::CUDA, LinearBackwardBf16,
                            DType::kBFloat16);
}

}  // namespace micrograd::cuda::ops

#endif
