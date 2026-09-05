
#include <functional>
#include <ranges>
#include <unordered_set>

#include "micrograd/Tensor.h"

namespace micrograd {

void Tensor::backward() {
  std::vector<std::shared_ptr<Tensor>> ordered;
  std::unordered_set<Tensor *> visited;

  std::function<void(const std::shared_ptr<Tensor> &)> findOrder =
      [&](const std::shared_ptr<Tensor> &node) {
        if (visited.contains(node.get())) {
          return;
        }
        visited.insert(node.get());
        for (auto &child : node->children_) {
          findOrder(child);
        }
        ordered.push_back(node);
      };
  findOrder(shared_from_this());

  to(Backend::CPU);

  for (scalar_t &g : grad_) {
    g = 1.0f;
  }

  for (const auto &node : std::ranges::reverse_view(ordered)) {
    if (node->backward_fn_) {
      node->backward_fn_();
    }
  }
}

}  // namespace micrograd
