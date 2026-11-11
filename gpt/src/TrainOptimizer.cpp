#include "gpt/TrainOptimizer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <map>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "micrograd/Device.h"
#include "micrograd/Storage.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Optimizer.h"

#ifdef MICROGRAD_CUDA_ENABLED
#include "micrograd/backends/cuda/ops/Ops.h"
#endif

namespace micrograd::gpt {

namespace {

void write_u64(std::ostream &out, uint64_t value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

uint64_t read_u64(std::istream &in) {
  uint64_t value = 0;
  in.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error("Optimizer state file is truncated");
  }
  return value;
}

bool is_decayed(const std::string &name) {
  const size_t last_dot = name.rfind('.');
  const size_t leaf_start = last_dot == std::string::npos ? 0 : last_dot + 1;
  return name.substr(leaf_start) == "weight";
}

bool is_muon_matrix(const std::string &name, const Tensor &param) {
  return name.starts_with("h.") && param.shape().size() == 2;
}

}  // namespace

TrainOptimizer::TrainOptimizer(const NamedParameters &named_parameters,
                               scalar_t weight_decay, scalar_t beta1,
                               scalar_t beta2, scalar_t eps, scalar_t base_lr,
                               scalar_t muon_lr, bool muon_bf16)
    : weight_decay_(weight_decay),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      base_lr_(base_lr),
      muon_lr_(muon_lr),
      muon_bf16_(muon_bf16) {
  for (const auto &[name, p] : named_parameters) {
    const bool muon = muon_lr > 0.0f && is_muon_matrix(name, *p);
    parameters_.push_back(p);
    decay_.push_back(is_decayed(name));
    muon_.push_back(muon);
    m_.push_back(make_moment(*p));
    v_.push_back(muon ? Tensor::zeros({1}, p->backend()) : make_moment(*p));
  }
}

void TrainOptimizer::write_state(std::ostream &out) const {
  write_u64(out, static_cast<uint64_t>(step_count_));
  for (size_t i = 0; i < parameters_.size(); i++) {
    write_tensor(out, *m_[i]);
    write_tensor(out, *v_[i]);
  }
}

void TrainOptimizer::read_state(std::istream &in) {
  step_count_ = static_cast<size_t>(read_u64(in));
  for (size_t i = 0; i < parameters_.size(); i++) {
    read_tensor(in, m_[i]);
    read_tensor(in, v_[i]);
  }
}

bool TrainOptimizer::fuses_grad_clip() const {
#ifdef MICROGRAD_CUDA_ENABLED
  return (muon_lr_ <= 0.0f || muon_bf16_) && !parameters_.empty() &&
         parameters_.front()->backend() == Device::CUDA;
#else
  return false;
#endif
}

void TrainOptimizer::step(
    scalar_t learning_rate, [[maybe_unused]] scalar_t grad_clip,
    [[maybe_unused]] const std::vector<std::shared_ptr<Tensor>> &clip_params) {
  step_count_++;
  scalar_t bias_correction1 =
      1.0f - std::pow(beta1_, static_cast<scalar_t>(step_count_));
  scalar_t bias_correction2 =
      1.0f - std::pow(beta2_, static_cast<scalar_t>(step_count_));
#ifdef MICROGRAD_CUDA_ENABLED
  if (!parameters_.empty() && parameters_.front()->backend() == Device::CUDA) {
    const scalar_t *clip_total = nullptr;
    if (grad_clip > 0.0f && !clip_params.empty()) {
      std::vector<Tensor *> raw;
      raw.reserve(clip_params.size());
      for (const auto &p : clip_params) {
        raw.push_back(p.get());
      }
      clip_total = micrograd::cuda::ops::GradNormSquaredTotal(raw);
    }
    step_muon(learning_rate, clip_total, grad_clip);
    std::vector<micrograd::cuda::ops::AdamWTensor> tensors;
    tensors.reserve(parameters_.size());
    for (size_t i = 0; i < parameters_.size(); i++) {
      if (muon_[i]) {
        continue;
      }
      tensors.push_back(micrograd::cuda::ops::AdamWTensor{
          static_cast<scalar_t *>(
              parameters_[i]->data_storage().device_pointer()),
          static_cast<const scalar_t *>(
              parameters_[i]->grad_storage().device_pointer()),
          static_cast<scalar_t *>(m_[i]->data_storage().device_pointer()),
          static_cast<scalar_t *>(v_[i]->data_storage().device_pointer()),
          parameters_[i]->size(), decay_[i]});
    }
    micrograd::cuda::ops::FusedAdamWStep(
        tensors, learning_rate, beta1_, beta2_, eps_, weight_decay_,
        bias_correction1, bias_correction2, clip_total, grad_clip);
    return;
  }
#endif
  step_muon(learning_rate);
  for (size_t i = 0; i < parameters_.size(); i++) {
    if (muon_[i]) {
      continue;
    }
    micrograd::ops::AdamWStep(*parameters_[i], *m_[i], *v_[i], learning_rate,
                              beta1_, beta2_, eps_, weight_decay_, decay_[i],
                              bias_correction1, bias_correction2);
  }
}

void TrainOptimizer::step_muon(scalar_t learning_rate,
                               [[maybe_unused]] const scalar_t *clip_total,
                               [[maybe_unused]] scalar_t max_norm) {
  if (muon_lr_ <= 0.0f) {
    return;
  }
  const scalar_t lr = muon_lr_ * learning_rate / base_lr_;
  const auto update_scale = [](const std::vector<size_t> &shape) {
    const scalar_t fan_in = static_cast<scalar_t>(shape[0]);
    const scalar_t fan_out = static_cast<scalar_t>(shape[1]);
    return std::sqrt(std::max(1.0f, fan_out / fan_in));
  };
#ifdef MICROGRAD_CUDA_ENABLED
  if (muon_bf16_ && !parameters_.empty() &&
      parameters_.front()->backend() == Device::CUDA) {
    std::map<std::vector<size_t>, std::vector<micrograd::cuda::ops::MuonTensor>>
        groups;
    for (size_t i = 0; i < parameters_.size(); i++) {
      if (!muon_[i]) {
        continue;
      }
      groups[parameters_[i]->shape()].push_back(
          {static_cast<scalar_t *>(
               parameters_[i]->data_storage().device_pointer()),
           static_cast<const scalar_t *>(
               parameters_[i]->grad_storage().device_pointer()),
           static_cast<scalar_t *>(m_[i]->data_storage().device_pointer())});
    }
    for (const auto &[shape, tensors] : groups) {
      micrograd::cuda::ops::MuonStepBatchedBf16(
          tensors, shape[0], shape[1], lr, kMuonMomentum, 0.0f, true,
          kMuonNsSteps, update_scale(shape), clip_total, max_norm);
    }
    return;
  }
#endif
  for (size_t i = 0; i < parameters_.size(); i++) {
    if (!muon_[i]) {
      continue;
    }
    micrograd::ops::MuonStep(*parameters_[i], *m_[i], lr, kMuonMomentum, 0.0f,
                             true, kMuonNsSteps,
                             update_scale(parameters_[i]->shape()));
  }
}

std::shared_ptr<Tensor> TrainOptimizer::make_moment(const Tensor &param) {
  auto moment = Tensor::zeros(param.shape());
  moment->to(param.backend());
  return moment;
}

void TrainOptimizer::write_tensor(std::ostream &out, const Tensor &tensor) {
  Storage host = tensor.data_storage().copy_to(Device::CPU);
  out.write(static_cast<const char *>(host.data()),
            static_cast<std::streamsize>(host.bytes()));
}

void TrainOptimizer::read_tensor(std::istream &in,
                                 std::shared_ptr<Tensor> &tensor) {
  std::vector<scalar_t> values(tensor->size());
  in.read(reinterpret_cast<char *>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(scalar_t)));
  if (!in) {
    throw std::runtime_error("Optimizer state file is truncated");
  }
  auto loaded = Tensor::from_values(tensor->shape(), std::move(values));
  loaded->to(tensor->backend());
  tensor = loaded;
}

}  // namespace micrograd::gpt
