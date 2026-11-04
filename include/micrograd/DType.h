#pragma once

#include <cstddef>
#include <stdexcept>

namespace micrograd {

enum class DType { kFloat32, kBFloat16 };

inline constexpr size_t kDTypeCount = static_cast<size_t>(DType::kBFloat16) + 1;

inline size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return sizeof(float);
    case DType::kBFloat16:
      return 2;
  }

  throw std::runtime_error("Unknown dtype");
}

}  // namespace micrograd
