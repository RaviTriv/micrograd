#include "micrograd/Tensor.h"

#ifdef MICROGRAD_METAL_ENABLED
#include "micrograd/metal/MetalContext.h"
#endif

namespace micrograd {

std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor> &b) {
  if (shape_.size() != 2 || b->shape_.size() != 2) {
    throw std::invalid_argument("Tensors must be 2D for matmul");
  }

  if (shape_[1] != b->shape_[0]) {
    throw std::invalid_argument("Inner dimensions must match for matmul");
  }

#ifdef MICROGRAD_METAL_ENABLED
  if (backend() == Backend::Metal && b->backend() == Backend::Metal) {
    return matmul_metal(b);
  }
#endif

  size_t m = shape_[0];
  size_t k = shape_[1];
  size_t n = b->shape_[1];

  auto result = std::make_shared<Tensor>(std::vector<size_t>{m, n});

  auto lhs = data();
  auto rhs = b->data();
  auto out = result->data();
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      scalar_t sum = 0.0f;
      for (size_t p = 0; p < k; p++) {
        sum += lhs[i * k + p] * rhs[p * n + j];
      }
      out[i * n + j] = sum;
    }
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result = result.get(), self_ptr, b, m, k, n]() {
    auto a_data = self_ptr->data();
    auto b_data = b->data();
    auto a_grad = self_ptr->grad();
    auto b_grad = b->grad();
    auto out_grad = result->grad();

    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        for (size_t p = 0; p < n; p++) {
          a_grad[i * k + j] += out_grad[i * n + p] * b_data[j * n + p];
        }
      }
    }

    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        for (size_t p = 0; p < m; p++) {
          b_grad[i * n + j] += a_data[p * k + i] * out_grad[p * n + j];
        }
      }
    }
  };

  return result;
}

}  // namespace micrograd
