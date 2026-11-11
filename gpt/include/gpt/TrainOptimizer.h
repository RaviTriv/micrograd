#pragma once

#include <algorithm>
#include <cstddef>
#include <istream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"

namespace micrograd::gpt {

using NamedParameters =
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>;

class TrainOptimizer {
 public:
  TrainOptimizer(const NamedParameters &named_parameters, scalar_t weight_decay,
                 scalar_t beta1, scalar_t beta2, scalar_t eps,
                 scalar_t base_lr = 0.0f, scalar_t muon_lr = 0.0f,
                 bool muon_bf16 = false);

  size_t muon_count() const {
    return static_cast<size_t>(std::count(muon_.begin(), muon_.end(), true));
  }

  size_t decayed_count() const {
    size_t count = 0;
    for (size_t i = 0; i < parameters_.size(); i++) {
      count += decay_[i] && !muon_[i] ? 1u : 0u;
    }
    return count;
  }

  size_t parameter_count() const { return parameters_.size(); }

  void zero_grad() {
    for (auto &p : parameters_) {
      p->release_grad();
    }
  }

  size_t step_count() const { return step_count_; }

  void write_state(std::ostream &out) const;
  void read_state(std::istream &in);

  bool fuses_grad_clip() const;

  void step(scalar_t learning_rate, scalar_t grad_clip = 0.0f,
            const std::vector<std::shared_ptr<Tensor>> &clip_params = {});

 private:
  void step_muon(scalar_t learning_rate, const scalar_t *clip_total = nullptr,
                 scalar_t max_norm = 0.0f);

  static constexpr scalar_t kMuonMomentum = 0.95f;
  static constexpr size_t kMuonNsSteps = 5;

  static std::shared_ptr<Tensor> make_moment(const Tensor &param);
  static void write_tensor(std::ostream &out, const Tensor &tensor);
  static void read_tensor(std::istream &in, std::shared_ptr<Tensor> &tensor);

  std::vector<std::shared_ptr<Tensor>> parameters_;
  std::vector<bool> decay_;
  std::vector<bool> muon_;
  scalar_t weight_decay_;
  scalar_t beta1_;
  scalar_t beta2_;
  scalar_t eps_;
  scalar_t base_lr_;
  scalar_t muon_lr_;
  [[maybe_unused]] bool muon_bf16_;
  size_t step_count_ = 0;
  std::vector<std::shared_ptr<Tensor>> m_;
  std::vector<std::shared_ptr<Tensor>> v_;
};

}  // namespace micrograd::gpt
