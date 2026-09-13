#pragma once

namespace micrograd::cuda::ops {

void RegisterCudaOps();

void RegisterArithmeticOps();
void RegisterActivationOps();
void RegisterBroadcastOps();

}  // namespace micrograd::cuda::ops
