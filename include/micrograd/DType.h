#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
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

inline uint16_t float_to_bfloat16_bits(float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<uint16_t>((bits >> 16) | 0x0040U);
  }
  const uint32_t rounded = bits + 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(rounded >> 16);
}

inline float bfloat16_bits_to_float(uint16_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

}  // namespace micrograd
