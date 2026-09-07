
#include "micrograd/Autograd.h"

#include <ranges>
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

  to(Backend::CPU);

  for (scalar_t &g : grad()) {
    g = 1.0f;
  }

  for (const auto &node : std::ranges::reverse_view(ordered)) {
    if (node->backward_fn_) {
      node->backward_fn_();
    }
  }
}

}  // namespace micrograd
