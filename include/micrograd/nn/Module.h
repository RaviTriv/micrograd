#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "micrograd/Tensor.h"

namespace micrograd {
namespace nn {

class Module {
 public:
  virtual ~Module() = default;

  std::vector<std::shared_ptr<Tensor>> parameters() const {
    std::vector<std::shared_ptr<Tensor>> result;
    collect_parameters(result);
    return result;
  }

  std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>
  named_parameters() const {
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> result;
    collect_named_parameters("", result);
    return result;
  }

  virtual void train(bool mode = true) {
    training_ = mode;
    for (const auto &entry : modules_) {
      entry.second->train(mode);
    }
  }

  void eval() { train(false); }

  bool is_training() const { return training_; }

 protected:
  void register_parameter(const std::string &name,
                          std::shared_ptr<Tensor> tensor) {
    parameters_.emplace_back(name, std::move(tensor));
  }

  void register_module(const std::string &name,
                       std::shared_ptr<Module> module) {
    modules_.emplace_back(name, std::move(module));
  }

 private:
  void collect_parameters(std::vector<std::shared_ptr<Tensor>> &result) const {
    for (const auto &entry : parameters_) {
      result.push_back(entry.second);
    }
    for (const auto &entry : modules_) {
      entry.second->collect_parameters(result);
    }
  }

  void collect_named_parameters(
      const std::string &prefix,
      std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> &result)
      const {
    for (const auto &entry : parameters_) {
      result.emplace_back(prefix + entry.first, entry.second);
    }
    for (const auto &entry : modules_) {
      entry.second->collect_named_parameters(prefix + entry.first + ".",
                                             result);
    }
  }

  bool training_ = true;
  std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> parameters_;
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> modules_;
};

}  // namespace nn
}  // namespace micrograd
