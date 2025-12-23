#pragma once
#include "Tensor.h"
#include <string>

std::string to_dot(const std::shared_ptr<Tensor> &tensor);
std::string format_scalar(const std::string &op, double scalar);
