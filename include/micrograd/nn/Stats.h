#pragma once

#include <cstddef>
#include <cstdint>

#include "micrograd/nn/Module.h"

namespace micrograd {

size_t param_count(const nn::Module &module);
size_t param_bytes(const nn::Module &module);
uint64_t linear_flops(size_t batch, size_t in_features, size_t out_features,
                      bool bias = true);

}  // namespace micrograd
