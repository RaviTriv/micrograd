#pragma once

#include <cstddef>
#include <memory>

#include "micrograd/Tensor.h"

namespace micrograd::init {

void kaiming_uniform_(const std::shared_ptr<Tensor> &tensor, size_t fan_in);
void xavier_uniform_(const std::shared_ptr<Tensor> &tensor, size_t fan_in,
                     size_t fan_out);
void normal_(const std::shared_ptr<Tensor> &tensor, scalar_t mean,
             scalar_t std);
void zeros_(const std::shared_ptr<Tensor> &tensor);

}  // namespace micrograd::init
