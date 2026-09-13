#pragma once

namespace micrograd::cuda::ops {

void RegisterCudaOps();

void RegisterArithmeticOps();
void RegisterActivationOps();
void RegisterBroadcastOps();
void RegisterReductionOps();

}  // namespace micrograd::cuda::ops
