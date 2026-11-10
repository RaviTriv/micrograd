#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "micrograd/Scalar.h"

namespace micrograd::cuda::ops {

void FloatToBf16(const scalar_t *in, void *out, size_t n, cudaStream_t stream);
void Bf16ToFloat(const void *in, scalar_t *out, size_t n, cudaStream_t stream);

}  // namespace micrograd::cuda::ops
