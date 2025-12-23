#include "micrograd/Tensor.h"

#ifdef MICROGRAD_METAL_ENABLED
#include "micrograd/metal/MetalContext.h"
#endif

std::shared_ptr<Tensor> Tensor::sum() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal) {
    return sum_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(std::vector<size_t>{1});
  result->op_ = "sum";

  double total = 0.0;
  for (size_t i = 0; i < data_.size(); i++) {
    total += data_[i];
  }
  result->data_[0] = total;

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[0];
    }
  };

  return result;
}