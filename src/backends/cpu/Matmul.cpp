#include "micrograd/ops/Matmul.h"

#include <algorithm>
#include <array>
#include <functional>
#include <thread>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cpu/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::ops::cpu {
namespace {

constexpr size_t kBlock = 64;
constexpr size_t kDotBlock = 256;
constexpr size_t kLanes = 8;
constexpr size_t kMinWorkPerThread = size_t{1} << 18;

template <typename RowsFn>
void ParallelRows(size_t batch, size_t rows, size_t work_per_row,
                  const RowsFn &fn) {
  const size_t total_rows = batch * rows;
  const size_t hardware =
      std::max<size_t>(1, std::thread::hardware_concurrency());
  const size_t threads = std::min(
      {hardware, total_rows, total_rows * work_per_row / kMinWorkPerThread});

  const auto run = [&](size_t begin, size_t end) {
    while (begin < end) {
      const size_t bidx = begin / rows;
      const size_t batch_end = std::min(end, (bidx + 1) * rows);
      fn(bidx, begin - (bidx * rows), batch_end - (bidx * rows));
      begin = batch_end;
    }
  };

  if (threads <= 1) {
    run(0, total_rows);
    return;
  }
  const size_t chunk = (total_rows + threads - 1) / threads;
  std::vector<std::jthread> workers;
  workers.reserve(threads - 1);
  for (size_t begin = chunk; begin < total_rows; begin += chunk) {
    workers.emplace_back(run, begin, std::min(begin + chunk, total_rows));
  }
  run(0, chunk);
}

void RowsNN(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t inner,
            size_t cols, size_t row_begin, size_t row_end) {
  for (size_t ii = row_begin; ii < row_end; ii += kBlock) {
    const size_t i_max = std::min(ii + kBlock, row_end);
    for (size_t pp = 0; pp < inner; pp += kBlock) {
      const size_t p_max = std::min(pp + kBlock, inner);
      for (size_t jj = 0; jj < cols; jj += kBlock) {
        const size_t j_max = std::min(jj + kBlock, cols);
        for (size_t i = ii; i < i_max; i++) {
          for (size_t p = pp; p < p_max; p++) {
            const scalar_t a_ip = a[(i * inner) + p];
            for (size_t j = jj; j < j_max; j++) {
              out[(i * cols) + j] += a_ip * b[(p * cols) + j];
            }
          }
        }
      }
    }
  }
}

scalar_t Dot(const scalar_t *x, const scalar_t *y, size_t count) {
  std::array<scalar_t, kLanes> partial{};
  size_t p = 0;
  for (; p + kLanes <= count; p += kLanes) {
    for (size_t lane = 0; lane < kLanes; lane++) {
      partial[lane] += x[p + lane] * y[p + lane];
    }
  }
  scalar_t sum = 0.0f;
  for (; p < count; p++) {
    sum += x[p] * y[p];
  }
  for (const scalar_t value : partial) {
    sum += value;
  }
  return sum;
}

void RowsNT(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t inner,
            size_t cols, size_t row_begin, size_t row_end) {
  for (size_t jj = 0; jj < cols; jj += kBlock) {
    const size_t j_max = std::min(jj + kBlock, cols);
    for (size_t pp = 0; pp < inner; pp += kDotBlock) {
      const size_t p_len = std::min(kDotBlock, inner - pp);
      for (size_t i = row_begin; i < row_end; i++) {
        for (size_t j = jj; j < j_max; j++) {
          out[(i * cols) + j] +=
              Dot(a + (i * inner) + pp, b + (j * inner) + pp, p_len);
        }
      }
    }
  }
}

void RowsTN(const scalar_t *a, const scalar_t *d, scalar_t *out, size_t depth,
            size_t a_cols, size_t cols, size_t row_begin, size_t row_end) {
  for (size_t ii = 0; ii < depth; ii += kBlock) {
    const size_t i_max = std::min(ii + kBlock, depth);
    for (size_t jj = 0; jj < cols; jj += kBlock) {
      const size_t j_max = std::min(jj + kBlock, cols);
      for (size_t p = row_begin; p < row_end; p++) {
        for (size_t i = ii; i < i_max; i++) {
          const scalar_t a_ip = a[(i * a_cols) + p];
          for (size_t j = jj; j < j_max; j++) {
            out[(p * cols) + j] += a_ip * d[(i * cols) + j];
          }
        }
      }
    }
  }
}

void Matmul(const OpArgs &args) {
  const MatmulDims d =
      ResolveMatmulDims(args.lhs->shape(), args.rhs->shape());
  const scalar_t *lhs = args.lhs->data().data();
  const scalar_t *rhs = args.rhs->data().data();
  scalar_t *out = args.out->data().data();

  ParallelRows(d.batch, d.m, d.k * d.n,
               [&](size_t bidx, size_t row_begin, size_t row_end) {
                 scalar_t *out_b = out + (bidx * d.m * d.n);
                 std::fill(out_b + (row_begin * d.n), out_b + (row_end * d.n),
                           0.0f);
                 RowsNN(lhs + (bidx * d.m * d.k), rhs + (bidx * d.k * d.n),
                        out_b, d.k, d.n, row_begin, row_end);
               });
}

std::function<void()> MatmulBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    const MatmulDims d = ResolveMatmulDims(lhs->shape(), rhs->shape());
    const scalar_t *out_grad = out->grad().data();

    if (lhs->requires_grad()) {
      const scalar_t *rhs_data = rhs->data().data();
      scalar_t *lhs_grad = lhs->grad().data();
      ParallelRows(d.batch, d.m, d.n * d.k,
                   [&](size_t bidx, size_t row_begin, size_t row_end) {
                     RowsNT(out_grad + (bidx * d.m * d.n),
                            rhs_data + (bidx * d.k * d.n),
                            lhs_grad + (bidx * d.m * d.k), d.n, d.k, row_begin,
                            row_end);
                   });
    }

    if (rhs->requires_grad()) {
      const scalar_t *lhs_data = lhs->data().data();
      scalar_t *rhs_grad = rhs->grad().data();
      ParallelRows(d.batch, d.k, d.m * d.n,
                   [&](size_t bidx, size_t row_begin, size_t row_end) {
                     RowsTN(lhs_data + (bidx * d.m * d.k),
                            out_grad + (bidx * d.m * d.n),
                            rhs_grad + (bidx * d.k * d.n), d.m, d.k, d.n,
                            row_begin, row_end);
                   });
    }
  };
}

struct NTDims {
  size_t m;
  size_t k;
  size_t n;
};

NTDims MakeNTDims(const Tensor &lhs, const Tensor &rhs) {
  const size_t k = lhs.shape().back();
  return {.m = lhs.size() / k, .k = k, .n = rhs.shape()[0]};
}

void MatmulNT(const OpArgs &args) {
  const NTDims d = MakeNTDims(*args.lhs, *args.rhs);
  const scalar_t *lhs = args.lhs->data().data();
  const scalar_t *rhs = args.rhs->data().data();
  scalar_t *out = args.out->data().data();

  ParallelRows(1, d.m, d.k * d.n,
               [&](size_t, size_t row_begin, size_t row_end) {
                 std::fill(out + (row_begin * d.n), out + (row_end * d.n),
                           0.0f);
                 RowsNT(lhs, rhs, out, d.k, d.n, row_begin, row_end);
               });
}

std::function<void()> MatmulNTBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, rhs = args.rhs]() {
    const NTDims d = MakeNTDims(*lhs, *rhs);
    const scalar_t *out_grad = out->grad().data();

    if (lhs->requires_grad()) {
      const scalar_t *rhs_data = rhs->data().data();
      scalar_t *lhs_grad = lhs->grad().data();
      ParallelRows(1, d.m, d.n * d.k,
                   [&](size_t, size_t row_begin, size_t row_end) {
                     RowsNN(out_grad, rhs_data, lhs_grad, d.n, d.k, row_begin,
                            row_end);
                   });
    }

    if (rhs->requires_grad()) {
      const scalar_t *lhs_data = lhs->data().data();
      scalar_t *rhs_grad = rhs->grad().data();
      ParallelRows(1, d.n, d.m * d.k,
                   [&](size_t, size_t row_begin, size_t row_end) {
                     RowsTN(out_grad, lhs_data, rhs_grad, d.m, d.n, d.k,
                            row_begin, row_end);
                   });
    }
  };
}

}  // namespace

void RegisterMatmulOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kMatmul, Device::CPU, Matmul);
  registry.RegisterBackward(OpId::kMatmul, Device::CPU, MatmulBackward);
  registry.Register(OpId::kMatmulNT, Device::CPU, MatmulNT);
  registry.RegisterBackward(OpId::kMatmulNT, Device::CPU, MatmulNTBackward);
}

}  // namespace micrograd::ops::cpu
