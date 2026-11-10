#pragma once

#include <cstddef>
#include <vector>

#include "micrograd/Scalar.h"

namespace micrograd {
class Tensor;
enum class OpId;
}  // namespace micrograd

namespace micrograd::cuda::ops {

void RegisterCudaOps();
bool ForwardOverwritesOutput(OpId op);

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
void RegisterAttentionOps();

scalar_t GradNormSquared(const Tensor &param);
void ClipGradNorm(const std::vector<Tensor *> &params, scalar_t max_norm);
const scalar_t *GradNormSquaredTotal(const std::vector<Tensor *> &params);
void ScaleGrad(Tensor &param, scalar_t scale);

void AdamWStep(Tensor &param, Tensor &m, Tensor &v, scalar_t lr, scalar_t beta1,
               scalar_t beta2, scalar_t eps, scalar_t weight_decay, bool decay,
               scalar_t bias_correction1, scalar_t bias_correction2);

struct AdamWTensor {
  scalar_t *data;
  const scalar_t *grad;
  scalar_t *m;
  scalar_t *v;
  size_t n;
  bool decay;
};

void FusedAdamWStep(const std::vector<AdamWTensor> &tensors, scalar_t lr,
                    scalar_t beta1, scalar_t beta2, scalar_t eps,
                    scalar_t weight_decay, scalar_t bias_correction1,
                    scalar_t bias_correction2,
                    const scalar_t *clip_total = nullptr,
                    scalar_t max_norm = 0.0f);

void MuonStep(Tensor &param, Tensor &momentum, scalar_t lr, scalar_t beta,
              scalar_t weight_decay, bool nesterov, size_t ns_steps,
              scalar_t update_scale);

struct MuonTensor {
  scalar_t *data;
  const scalar_t *grad;
  scalar_t *momentum;
};

void MuonStepBatchedBf16(const std::vector<MuonTensor> &tensors, size_t rows,
                         size_t cols, scalar_t lr, scalar_t beta,
                         scalar_t weight_decay, bool nesterov, size_t ns_steps,
                         scalar_t update_scale,
                         const scalar_t *clip_total = nullptr,
                         scalar_t max_norm = 0.0f);

}  // namespace micrograd::cuda::ops
