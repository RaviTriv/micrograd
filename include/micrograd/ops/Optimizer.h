#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "micrograd/Scalar.h"

namespace micrograd {

class Tensor;

namespace ops {

scalar_t GradNormSquared(const Tensor &param);
void ScaleGrad(Tensor &param, scalar_t scale);
scalar_t ClipGradNorm(const std::vector<std::shared_ptr<Tensor>> &params,
                      scalar_t max_norm);

void AdamWStep(Tensor &param, Tensor &m, Tensor &v, scalar_t lr, scalar_t beta1,
               scalar_t beta2, scalar_t eps, scalar_t weight_decay, bool decay,
               scalar_t bias_correction1, scalar_t bias_correction2);

void MuonStep(Tensor &param, Tensor &momentum, scalar_t lr, scalar_t beta,
              scalar_t weight_decay, bool nesterov, size_t ns_steps,
              scalar_t update_scale);

}  // namespace ops
}  // namespace micrograd
