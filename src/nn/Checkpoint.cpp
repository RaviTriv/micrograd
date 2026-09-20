#include <functional>
#include <utility>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::checkpoint(
    const std::shared_ptr<Tensor> &input,
    const std::function<
        std::shared_ptr<Tensor>(const std::shared_ptr<Tensor> &)> &recompute) {
  std::shared_ptr<Tensor> output;
  {
    const NoGradGuard no_grad;
    output = recompute(input);
  }

  output->requires_grad_ = GradEnabled() && input->requires_grad_;
  if (output->requires_grad_) {
    output->children_ = {input};
    Tensor *out = output.get();
    output->backward_fn_ = [input, recompute, out]() {
      std::vector<std::shared_ptr<Tensor>> saved_children =
          std::move(input->children_);
      std::function<void()> saved_backward_fn = std::move(input->backward_fn_);
      input->children_.clear();
      input->backward_fn_ = nullptr;

      std::shared_ptr<Tensor> recomputed = recompute(input);
      recomputed->grad_ = out->grad_.copy_to(recomputed->backend());
      recomputed->propagate_gradients();

      input->children_ = std::move(saved_children);
      input->backward_fn_ = std::move(saved_backward_fn);
    };
  }

  return output;
}

}  // namespace micrograd
