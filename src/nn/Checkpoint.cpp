#include <functional>
#include <random>
#include <utility>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Random.h"
#include "micrograd/Tensor.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::checkpoint(
    const std::shared_ptr<Tensor> &input,
    const std::function<
        std::shared_ptr<Tensor>(const std::shared_ptr<Tensor> &)> &recompute) {
  const std::mt19937_64 forward_rng = global_rng();
  std::shared_ptr<Tensor> output;
  {
    const NoGradGuard no_grad;
    output = recompute(input);
  }

  output->requires_grad_ = GradEnabled() && input->requires_grad_;
  if (output->requires_grad_) {
    output->children_ = {input};
    Tensor *out = output.get();
    output->backward_fn_ = [input, recompute, out, forward_rng]() {
      std::vector<std::shared_ptr<Tensor>> saved_children =
          std::move(input->children_);
      std::function<void()> saved_backward_fn = std::move(input->backward_fn_);
      input->children_.clear();
      input->backward_fn_ = nullptr;

      const std::mt19937_64 backward_rng = global_rng();
      global_rng() = forward_rng;
      std::shared_ptr<Tensor> recomputed = recompute(input);
      global_rng() = backward_rng;
      recomputed->grad_ = out->grad_.copy_to(recomputed->backend());
      recomputed->propagate_gradients();

      input->children_ = std::move(saved_children);
      input->backward_fn_ = std::move(saved_backward_fn);
    };
  }

  return output;
}

}  // namespace micrograd
