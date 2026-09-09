
#include "micrograd/Autograd.h"

#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "micrograd/Tensor.h"

namespace micrograd {
namespace {
thread_local bool grad_enabled = true;
}  // namespace

bool GradEnabled() { return grad_enabled; }

NoGradGuard::NoGradGuard() : previous_(grad_enabled) { grad_enabled = false; }

NoGradGuard::~NoGradGuard() { grad_enabled = previous_; }

void Tensor::backward() {
  if (size() != 1) {
    throw std::invalid_argument(
        "backward() requires a scalar tensor, pass an explicit gradient "
        "instead");
  }

  to(Backend::CPU);
  grad()[0] = 1.0f;
  propagate_gradients();
}

void Tensor::backward(const Tensor &grad_output) {
  if (grad_output.shape() != shape_) {
    throw std::invalid_argument("Gradient and tensor shape mismatch");
  }

  to(Backend::CPU);
  const auto *seed =
      static_cast<const scalar_t *>(grad_output.data_storage().host_pointer());
  std::copy_n(seed, size(), grad().begin());
  propagate_gradients();
}

void Tensor::propagate_gradients() {
  std::vector<std::shared_ptr<Tensor>> ordered;
  std::unordered_set<Tensor *> visited;
  std::vector<std::pair<std::shared_ptr<Tensor>, size_t>> pending;

  visited.insert(this);
  pending.emplace_back(shared_from_this(), 0);
  while (!pending.empty()) {
    auto &[node, next_child] = pending.back();
    if (next_child < node->children_.size()) {
      const std::shared_ptr<Tensor> &child = node->children_[next_child++];
      if (visited.insert(child.get()).second) {
        pending.emplace_back(child, 0);
      }
      continue;
    }
    ordered.push_back(std::move(node));
    pending.pop_back();
  }

  for (const auto &node : std::ranges::reverse_view(ordered)) {
    if (node->backward_fn_) {
      node->backward_fn_();
    }
  }
}

}  // namespace micrograd
