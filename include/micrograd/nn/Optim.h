#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "micrograd/Storage.h"
#include "micrograd/Tensor.h"

namespace micrograd {

class Optimizer {
 public:
  explicit Optimizer(std::vector<std::shared_ptr<Tensor>> parameters);
  virtual ~Optimizer() = default;

  Optimizer(const Optimizer &) = delete;
  Optimizer &operator=(const Optimizer &) = delete;

  virtual void step() = 0;
  void zero_grad();

  size_t step_count() const { return step_count_; }
  const std::vector<std::shared_ptr<Tensor>> &parameters() const {
    return parameters_;
  }

 protected:
  std::vector<std::shared_ptr<Tensor>> parameters_;
  size_t step_count_ = 0;
};

class SGD : public Optimizer {
 public:
  SGD(std::vector<std::shared_ptr<Tensor>> parameters, scalar_t learning_rate,
      scalar_t momentum = 0, scalar_t weight_decay = 0, bool nesterov = false);

  void step() override;

 private:
  scalar_t learning_rate_;
  scalar_t momentum_;
  scalar_t weight_decay_;
  bool nesterov_;
  std::vector<std::vector<scalar_t>> velocity_;
};

class AdamW : public Optimizer {
 public:
  AdamW(std::vector<std::shared_ptr<Tensor>> parameters, scalar_t learning_rate,
        std::pair<scalar_t, scalar_t> betas = {0.9f, 0.999f},
        scalar_t eps = 1e-8f, scalar_t weight_decay = 1e-2f);

  void step() override;

 private:
  scalar_t learning_rate_;
  scalar_t beta1_;
  scalar_t beta2_;
  scalar_t eps_;
  scalar_t weight_decay_;
  std::vector<Storage> m_;
  std::vector<Storage> v_;
  std::vector<std::vector<scalar_t>> master_;
};

class Muon : public Optimizer {
 public:
  Muon(std::vector<std::shared_ptr<Tensor>> parameters, scalar_t learning_rate,
       scalar_t momentum = 0.95f, scalar_t weight_decay = 0.01f,
       bool nesterov = true, size_t ns_steps = 5);

  void step() override;

 private:
  scalar_t learning_rate_;
  scalar_t momentum_;
  scalar_t weight_decay_;
  bool nesterov_;
  size_t ns_steps_;
  std::vector<std::vector<scalar_t>> momentum_buffer_;
};

}  // namespace micrograd
