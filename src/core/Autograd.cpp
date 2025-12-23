
#include "micrograd/Tensor.h"
#include <functional>
#include <unordered_set>

void Tensor::backward() {
  std::vector<std::shared_ptr<Tensor>> ordered;
  std::unordered_set<Tensor *> visited;

  std::function<void(std::shared_ptr<Tensor>)> findOrder =
      [&](std::shared_ptr<Tensor> node) {
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

  for (size_t i = 0; i < grad_.size(); i++) {
    grad_[i] = 1.0;
  }

  for (auto it = ordered.rbegin(); it != ordered.rend(); it++) {
    if ((*it)->backward_fn_) {
      (*it)->backward_fn_();
    }
  }
}