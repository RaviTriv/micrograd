#pragma once

namespace micrograd::cuda::ops {

void RegisterCudaOps();

void RegisterArithmeticOps();
void RegisterActivationOps();
void RegisterBroadcastOps();
void RegisterReductionOps();
void RegisterMatmulOps();
void RegisterShapeOps();
void RegisterEmbeddingOps();
void RegisterLayerNormOps();

}  // namespace micrograd::cuda::ops
