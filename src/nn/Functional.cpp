#include "micrograd/nn/Functional.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace micrograd {

std::shared_ptr<Tensor> avg_pool2d(const std::shared_ptr<Tensor> &input,
                                   size_t kernel) {
  if (input->shape().size() != 2) {
    throw std::invalid_argument("avg_pool2d expects rank 2 input");
  }
  if (kernel == 0) {
    throw std::invalid_argument("avg_pool2d kernel must be positive");
  }

  size_t batch = input->shape()[0];
  size_t pixels = input->shape()[1];
  auto side =
      static_cast<size_t>(std::lround(std::sqrt(static_cast<double>(pixels))));
  if (side * side != pixels) {
    throw std::invalid_argument("avg_pool2d expects a square image");
  }
  if (side % kernel != 0) {
    throw std::invalid_argument(
        "avg_pool2d kernel must evenly divide the image side");
  }

  size_t pooled_side = side / kernel;
  size_t pooled_pixels = pooled_side * pooled_side;
  auto divisor = static_cast<scalar_t>(kernel * kernel);

  std::vector<scalar_t> pooled(batch * pooled_pixels);
  for (size_t b = 0; b < batch; b++) {
    for (size_t py = 0; py < pooled_side; py++) {
      for (size_t px = 0; px < pooled_side; px++) {
        scalar_t sum = 0.0f;
        for (size_t dy = 0; dy < kernel; dy++) {
          for (size_t dx = 0; dx < kernel; dx++) {
            size_t y = (py * kernel) + dy;
            size_t x = (px * kernel) + dx;
            sum += input->at({b, (y * side) + x});
          }
        }
        pooled[(b * pooled_pixels) + (py * pooled_side) + px] = sum / divisor;
      }
    }
  }

  return std::make_shared<Tensor>(std::vector<size_t>{batch, pooled_pixels},
                                  pooled);
}

std::shared_ptr<Tensor> gelu(const std::shared_ptr<Tensor> &input) {
  constexpr scalar_t kSqrt2OverPi = 0.7978845608028654f;
  constexpr scalar_t kCubicCoeff = 0.044715f;

  auto cubed = input->pow(3.0f);
  auto inner = input->add(cubed->mul(kCubicCoeff))->mul(kSqrt2OverPi);
  auto gate = inner->tanh()->add(1.0f)->mul(0.5f);

  return input->mul(gate);
}

std::shared_ptr<Tensor> relu_squared(const std::shared_ptr<Tensor> &input) {
  return input->relu()->pow(2.0f);
}

std::shared_ptr<Tensor> qk_norm(const std::shared_ptr<Tensor> &input,
                                scalar_t eps) {
  int64_t last_dim = static_cast<int64_t>(input->shape().size()) - 1;
  auto mean_square = input->pow(2.0f)->mean(last_dim, true);
  auto rms = mean_square->add(eps)->sqrt();
  return input->div(rms);
}

}  // namespace micrograd
