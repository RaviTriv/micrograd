#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

#ifdef MICROGRAD_METAL_ENABLED
#include "micrograd/metal/MetalContext.h"
#endif

namespace micrograd {

std::shared_ptr<Tensor> Tensor::sum() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend() == Backend::Metal) {
    return sum_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(std::vector<size_t>{1});
  DispatchOp(OpId::kSum, backend(), {.lhs = this, .out = result.get()});

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result = result.get(), self_ptr]() {
    const scalar_t out_grad = result->grad()[0];
    for (scalar_t &g : self_ptr->grad()) {
      g += out_grad;
    }
  };

  return result;
}

}  // namespace micrograd
