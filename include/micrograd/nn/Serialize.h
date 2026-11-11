#pragma once

#include <map>
#include <string>

#include "micrograd/nn/Module.h"

namespace micrograd {

using StateDictMetadata = std::map<std::string, std::string>;

void save(const std::string &path, nn::Module &module);
void save(const std::string &path, nn::Module &module,
          const StateDictMetadata &metadata);
void load(const std::string &path, nn::Module &module);
StateDictMetadata load_metadata(const std::string &path);

}  // namespace micrograd
