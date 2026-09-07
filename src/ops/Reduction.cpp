#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::sum() {
  auto result = std::make_shared<Tensor>(std::vector<size_t>{1});
  DispatchOp(OpId::kSum, backend(), {.lhs = this, .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kSum, backend(),
                                        {.lhs = self_ptr, .out = result.get()});
  }

  return result;
}

}  // namespace micrograd
