#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "micrograd/Tensor.h"

namespace micrograd::nn {

class Module {
 public:
  Module() = default;
  virtual ~Module() = default;

  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;
  Module(Module &&) = delete;
  Module &operator=(Module &&) = delete;

  virtual std::shared_ptr<Tensor> forward(
      const std::shared_ptr<Tensor> &input) = 0;

  std::vector<std::shared_ptr<Tensor>> parameters() const {
    std::vector<std::shared_ptr<Tensor>> result;
    for (auto &entry : named_parameters()) {
      result.push_back(std::move(entry.second));
    }
    return result;
  }

  std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>
  named_parameters() const {
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> result;
    std::unordered_set<const Tensor *> seen;
    collect_named_parameters("", seen, result);
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

  void set_checkpointing(bool enabled = true) { checkpointing_ = enabled; }

  std::shared_ptr<Tensor> maybe_checkpoint(
      const std::shared_ptr<Tensor> &input,
      const std::function<
          std::shared_ptr<Tensor>(const std::shared_ptr<Tensor> &)> &fn) const {
    if (checkpointing_) {
      return Tensor::checkpoint(input, fn);
    }
    return fn(input);
  }

 private:
  void collect_named_parameters(
      const std::string &prefix, std::unordered_set<const Tensor *> &seen,
      std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> &result)
      const {
    for (const auto &entry : parameters_) {
      if (seen.insert(entry.second.get()).second) {
        result.emplace_back(prefix + entry.first, entry.second);
      }
    }
    for (const auto &entry : modules_) {
      entry.second->collect_named_parameters(prefix + entry.first + ".", seen,
                                             result);
    }
  }

  bool training_ = true;
  bool checkpointing_ = false;
  std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> parameters_;
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> modules_;
};

}  // namespace micrograd::nn
