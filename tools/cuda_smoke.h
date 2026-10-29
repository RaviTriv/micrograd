#pragma once

#include <vector>

namespace micrograd::tools {

int cudnn_version();
int cublaslt_version();
bool cuda_device_available();
std::vector<int> cuda_iota(int n);

}  // namespace micrograd::tools
