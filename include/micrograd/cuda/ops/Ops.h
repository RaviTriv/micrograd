#pragma once

#include "micrograd/Scalar.h"

namespace micrograd {
class Tensor;
}  // namespace micrograd

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
void RegisterRmsNormOps();
void RegisterRotaryOps();

scalar_t GradNormSquared(const Tensor &param);
void ScaleGrad(Tensor &param, scalar_t scale);

void AdamWStep(Tensor &param, Tensor &m, Tensor &v, scalar_t lr, scalar_t beta1,
               scalar_t beta2, scalar_t eps, scalar_t weight_decay, bool decay,
               scalar_t bias_correction1, scalar_t bias_correction2);

}  // namespace micrograd::cuda::ops
