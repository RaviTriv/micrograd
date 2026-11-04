#pragma once

#include <cstddef>
#include <memory>

#include "micrograd/Tensor.h"

namespace micrograd::init {

void kaiming_uniform_(const std::shared_ptr<Tensor> &tensor, size_t fan_in);
void xavier_uniform_(const std::shared_ptr<Tensor> &tensor, size_t fan_in,
                     size_t fan_out);

}  // namespace micrograd::init
