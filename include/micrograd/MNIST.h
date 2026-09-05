#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Tensor.h"

struct MNISTData {
  std::vector<std::shared_ptr<Tensor>> images;
  std::vector<std::shared_ptr<Tensor>> labels;
};

MNISTData load_mnist(const std::string &images_path,
                     const std::string &labels_path, int sample_count);
