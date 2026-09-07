#pragma once

namespace micrograd::ops::cpu {

void RegisterArithmeticOps();
void RegisterMatmulOps();
void RegisterReductionOps();
void RegisterActivationOps();

}  // namespace micrograd::ops::cpu
