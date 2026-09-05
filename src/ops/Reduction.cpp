#include "micrograd/Tensor.h"

#ifdef MICROGRAD_METAL_ENABLED
#include "micrograd/metal/MetalContext.h"
#endif

namespace micrograd {

std::shared_ptr<Tensor> Tensor::sum() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == Backend::Metal) {
    return sum_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(std::vector<size_t>{1});

  scalar_t total = 0.0f;
  for (scalar_t value : data_) {
    total += value;
  }
  result->data_[0] = total;

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    for (scalar_t &g : self_ptr->grad_) {
      g += result->grad_[0];
    }
  };

  return result;
}

}  // namespace micrograd
