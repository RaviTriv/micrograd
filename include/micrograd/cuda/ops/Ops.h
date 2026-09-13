#pragma once

namespace micrograd::cuda::ops {

void RegisterCudaOps();

void RegisterArithmeticOps();
void RegisterActivationOps();

}  // namespace micrograd::cuda::ops
