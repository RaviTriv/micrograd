#pragma once

namespace micrograd::ops::cpu {

void RegisterArithmeticOps();
void RegisterMatmulOps();
void RegisterReductionOps();
void RegisterActivationOps();
void RegisterShapeOps();
void RegisterBroadcastOps();
void RegisterEmbeddingOps();
void RegisterLayerNormOps();
void RegisterRmsNormOps();

}  // namespace micrograd::ops::cpu
