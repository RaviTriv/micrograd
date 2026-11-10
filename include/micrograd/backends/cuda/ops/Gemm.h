#pragma once

#include <cstddef>

#include "micrograd/Scalar.h"

namespace micrograd::cuda::ops {

struct GemmOperand {
  const scalar_t *ptr = nullptr;
  bool trans = false;
  size_t ld = 0;
  size_t inner_stride = 0;
  size_t outer_stride = 0;
  const void *bf16_copy = nullptr;
};

struct GemmOutput {
  scalar_t *ptr = nullptr;
  size_t ld = 0;
  size_t inner_stride = 0;
  size_t outer_stride = 0;
};

struct GemmShape {
  size_t outer = 1;
  size_t inner = 1;
  size_t m = 0;
  size_t n = 0;
  size_t k = 0;
};

void StridedGemm(const GemmShape &shape, scalar_t alpha, const GemmOperand &a,
                 const GemmOperand &b, scalar_t beta, const GemmOutput &c);

}  // namespace micrograd::cuda::ops
