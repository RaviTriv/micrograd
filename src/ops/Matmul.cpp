#include "micrograd/Tensor.h"

#ifdef MICROGRAD_METAL_ENABLED
#include "micrograd/metal/MetalContext.h"
#endif

std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor> &b) {
  if (shape_.size() != 2 || b->shape_.size() != 2) {
    throw std::invalid_argument("Tensors must be 2D for matmul");
  }

  if (shape_[1] != b->shape_[0]) {
    throw std::invalid_argument("Inner dimensions must match for matmul");
  }

#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal &&
      b->backend_ == micrograd::Backend::Metal) {
    return matmul_metal(b);
  }
#endif

  size_t m = shape_[0];
  size_t k = shape_[1];
  size_t n = b->shape_[1];

  auto result = std::make_shared<Tensor>(std::vector<size_t>{m, n});
  result->op_ = "@";

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      double sum = 0.0;
      for (size_t p = 0; p < k; p++) {
        sum += at({i, p}) * b->at({p, j});
      }
      result->at({i, j}) = sum;
    }
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result, self_ptr, b, m, k, n]() {
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        for (size_t p = 0; p < n; p++) {
          self_ptr->grad_at({i, j}) += result->grad_at({i, p}) * b->at({j, p});
        }
      }
    }

    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        for (size_t p = 0; p < m; p++) {
          b->grad_at({i, j}) += self_ptr->at({p, i}) * result->grad_at({p, j});
        }
      }
    }
  };

  return result;
}