#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::relu() {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kRelu, backend(), {.lhs = this, .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kRelu, backend(),
                                        {.lhs = self_ptr, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid() {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kSigmoid, backend(), {.lhs = this, .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kSigmoid, backend(),
                                        {.lhs = self_ptr, .out = result.get()});
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::tanh() {
  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kTanh, backend(), {.lhs = this, .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kTanh, backend(),
                                        {.lhs = self_ptr, .out = result.get()});
  }

  return result;
}

}  // namespace micrograd
