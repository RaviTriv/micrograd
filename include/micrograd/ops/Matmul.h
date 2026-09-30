#pragma once

#include <cstddef>
#include <vector>

namespace micrograd::ops {

struct MatmulDims {
  size_t batch;
  size_t m;
  size_t k;
  size_t n;
};

// A contiguous [B, M, K] is laid out as [B*M, K], so a rank 3 lhs against a
// rank 2 rhs runs as one [B*M, K] x [K, N] matmul with no copies of the rhs.
inline MatmulDims ResolveMatmulDims(const std::vector<size_t> &lhs_shape,
                                    const std::vector<size_t> &rhs_shape) {
  if (lhs_shape.size() == 3 && rhs_shape.size() == 2) {
    const size_t folded_rows = lhs_shape[0] * lhs_shape[1];
    return {.batch = 1, .m = folded_rows, .k = lhs_shape[2], .n = rhs_shape[1]};
  }
  const size_t rank = lhs_shape.size();
  const size_t batch = rank == 3 ? lhs_shape[0] : 1;
  return {.batch = batch,
          .m = lhs_shape[rank - 2],
          .k = lhs_shape[rank - 1],
          .n = rhs_shape[rank - 1]};
}

}  // namespace micrograd::ops
