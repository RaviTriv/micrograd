#pragma once

namespace micrograd::metal::ops {

void RegisterMetalOps();

void RegisterArithmeticOps();
void RegisterMatmulOps();
void RegisterReductionOps();
void RegisterActivationOps();

}  // namespace micrograd::metal::ops
