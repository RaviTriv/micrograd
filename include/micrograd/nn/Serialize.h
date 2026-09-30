#pragma once

#include <string>

#include "micrograd/nn/Module.h"

namespace micrograd {

void save(const std::string &path, nn::Module &module);
void load(const std::string &path, nn::Module &module);

}  // namespace micrograd
