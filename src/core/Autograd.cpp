
#include "micrograd/Autograd.h"

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

void Tensor::require_grad_for_backward() const {
  if (!requires_grad_) {
    throw std::runtime_error(
        "backward() called on a tensor that does not require grad, set "
        "requires_grad on an input before the forward pass");
  }
}

void Tensor::accumulate_into_grad(const Storage &addend) {
  Storage total = grad_.copy_to(Device::CPU);
  const Storage extra = addend.copy_to(Device::CPU);
  auto *dst = static_cast<scalar_t *>(total.host_pointer());
  const auto *src = static_cast<const scalar_t *>(extra.host_pointer());
  for (size_t i = 0; i < size(); i++) {
    dst[i] += src[i];
  }
  grad_ = total.copy_to(backend());
}

void Tensor::backward(bool retain_graph) {
  require_grad_for_backward();
  if (size() != 1) {
    throw std::invalid_argument(
        "backward() requires a scalar tensor, pass an explicit gradient "
        "instead");
  }

  Storage seed(sizeof(scalar_t), Device::CPU);
  *static_cast<scalar_t *>(seed.data()) = 1.0f;
  propagate_gradients(seed.copy_to(backend()), retain_graph);
}

void Tensor::backward(const Tensor &grad_output, bool retain_graph) {
  require_grad_for_backward();
  if (grad_output.shape() != shape_) {
    throw std::invalid_argument("Gradient and tensor shape mismatch");
  }

  propagate_gradients(grad_output.data_storage().copy_to(backend()),
                      retain_graph);
}

void Tensor::propagate_gradients(Storage seed, bool retain_graph) {
  std::vector<std::shared_ptr<Tensor>> ordered;
  std::unordered_set<Tensor *> visited;
  std::vector<std::pair<std::shared_ptr<Tensor>, size_t>> pending;

  const auto check_not_released = [](const Tensor &node) {
    if (node.graph_released_) {
      throw std::runtime_error(
          "backward() reached a graph that an earlier backward() released, "
          "recompute the forward pass or pass retain_graph = true to the "
          "earlier call");
    }
  };

  check_not_released(*this);
  visited.insert(this);
  pending.emplace_back(shared_from_this(), 0);
  while (!pending.empty()) {
    auto &[node, next_child] = pending.back();
    if (next_child < node->children_.size()) {
      const std::shared_ptr<Tensor> &child = node->children_[next_child++];
      if (visited.insert(child.get()).second) {
        check_not_released(*child);
        pending.emplace_back(child, 0);
      }
      continue;
    }
    ordered.push_back(std::move(node));
    pending.pop_back();
  }

  for (const auto &node : ordered) {
    if (node->backward_fn_) {
      node->grad_ = Storage();
    } else {
      node->ensure_grad();
    }
  }
  if (backward_fn_ || !has_grad()) {
    grad_ = std::move(seed);
  } else {
    accumulate_into_grad(seed);
  }

  for (const auto &node : std::ranges::reverse_view(ordered)) {
    if (node->backward_fn_) {
      node->backward_fn_();
      node->grad_ = Storage();
    }
  }

  if (retain_graph) {
    return;
  }
  for (const auto &node : ordered) {
    if (node->backward_fn_) {
      node->graph_released_ = true;
    }
    node->backward_fn_ = nullptr;
    node->children_.clear();
  }
}

}  // namespace micrograd
