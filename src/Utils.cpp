#include "micrograd/utils.h"
#include <functional>
#include <sstream>
#include <unordered_set>

std::string to_dot(const std::shared_ptr<Tensor> &tensor) {
  std::stringstream ss;
  ss << "digraph {\n";
  ss << "  rankdir=LR;\n";

  std::unordered_set<Tensor *> visited;

  std::function<void(std::shared_ptr<Tensor>)> build =
      [&](std::shared_ptr<Tensor> node) {
        if (visited.contains(node.get()))
          return;
        visited.insert(node.get());

        ss << "  \"" << node.get() << "\" [label=\"data " << node->data_[0];
        if (node->data_.size() > 1)
          ss << "...";
        ss << " | grad " << node->grad_[0];
        if (node->grad_.size() > 1)
          ss << "...";
        ss << "\", shape=record];\n";

        if (!node->op_.empty()) {
          ss << "  \"" << node.get() << "_op\" [label=\"" << node->op_
             << "\"];\n";
          ss << "  \"" << node.get() << "_op\" -> \"" << node.get() << "\";\n";

          for (auto &child : node->children_) {
            ss << "  \"" << child.get() << "\" -> \"" << node.get()
               << "_op\";\n";
            build(child);
          }
        }
      };

  build(tensor);
  ss << "}\n";
  return ss.str();
}