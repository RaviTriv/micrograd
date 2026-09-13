#pragma once

namespace micrograd::ops::cpu {

void RegisterArithmeticOps();
void RegisterMatmulOps();
void RegisterReductionOps();
void RegisterActivationOps();
void RegisterShapeOps();
void RegisterBroadcastOps();

}  // namespace micrograd::ops::cpu
