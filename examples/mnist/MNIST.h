#pragma once

#include <cstddef>
#include <string>

#include "micrograd/data/DataLoader.h"

namespace micrograd {

data::TensorDataset load_mnist(const std::string &images_path,
                               const std::string &labels_path,
                               size_t sample_count);

}  // namespace micrograd
