#include "micrograd/nn/Optim.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "micrograd/ops/Optimizer.h"

#ifdef MICROGRAD_CUDA_ENABLED
#include "micrograd/backends/cuda/ops/Ops.h"
#endif

namespace micrograd {

namespace {

void require_nonnegative(const char *name, scalar_t value) {
  if (!std::isfinite(value) || value < 0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and nonnegative");
  }
}

void require_beta(const char *name, scalar_t value) {
  if (!std::isfinite(value) || value < 0 || value >= 1) {
    throw std::invalid_argument(std::string(name) + " must be in [0, 1)");
  }
}

}  // namespace

Optimizer::Optimizer(std::vector<std::shared_ptr<Tensor>> parameters,
                     scalar_t learning_rate)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate) {
  require_nonnegative("learning_rate", learning_rate_);
  std::unordered_set<const Tensor *> seen;
  for (const auto &p : parameters_) {
    if (!p) {
      throw std::invalid_argument("Optimizer parameters must not be null");
    }
    if (!seen.insert(p.get()).second) {
      throw std::invalid_argument(
          "Optimizer parameters must not contain duplicates");
    }
  }
}

void Optimizer::set_learning_rate(scalar_t learning_rate) {
  require_nonnegative("learning_rate", learning_rate);
  learning_rate_ = learning_rate;
}

void Optimizer::zero_grad() {
  for (auto &p : parameters_) {
    p->zero_grad();
  }
}

SGD::SGD(std::vector<std::shared_ptr<Tensor>> parameters,
         scalar_t learning_rate, scalar_t momentum, scalar_t weight_decay,
         bool nesterov)
    : Optimizer(std::move(parameters), learning_rate),
      momentum_(momentum),
      weight_decay_(weight_decay),
      nesterov_(nesterov) {
  require_nonnegative("momentum", momentum_);
  require_nonnegative("weight_decay", weight_decay_);
  if (momentum_ != 0) {
    velocity_.reserve(parameters_.size());
    for (auto &p : parameters_) {
      velocity_.emplace_back(p->size(), static_cast<scalar_t>(0));
    }
  }
}

void SGD::step() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    auto &p = parameters_[i];
    auto data = p->data();
    auto grad = p->grad();
    for (size_t j = 0; j < p->size(); j++) {
      scalar_t g = grad[j] + weight_decay_ * data[j];
      scalar_t update = g;
      if (momentum_ != 0) {
        auto &velocity = velocity_[i];
        velocity[j] = momentum_ * velocity[j] + g;
        update = nesterov_ ? g + momentum_ * velocity[j] : velocity[j];
      }
      data[j] -= learning_rate_ * update;
    }
  }
  step_count_++;
}

Adam::Adam(std::vector<std::shared_ptr<Tensor>> parameters,
           scalar_t learning_rate, std::pair<scalar_t, scalar_t> betas,
           scalar_t eps, scalar_t weight_decay)
    : Optimizer(std::move(parameters), learning_rate),
      beta1_(betas.first),
      beta2_(betas.second),
      eps_(eps),
      weight_decay_(weight_decay) {
  require_beta("beta1", beta1_);
  require_beta("beta2", beta2_);
  require_nonnegative("eps", eps_);
  require_nonnegative("weight_decay", weight_decay_);
  m_.reserve(parameters_.size());
  v_.reserve(parameters_.size());
  for (auto &p : parameters_) {
    if (p->backend() != Device::CPU ||
        p->data_storage().dtype() != DType::kFloat32) {
      throw std::invalid_argument("Adam requires float32 CPU parameters");
    }
    m_.emplace_back(p->size(), static_cast<scalar_t>(0));
    v_.emplace_back(p->size(), static_cast<scalar_t>(0));
  }
}

void Adam::step() {
  step_count_++;
  scalar_t bias_correction1 =
      1.0f - std::pow(beta1_, static_cast<scalar_t>(step_count_));
  scalar_t bias_correction2 =
      1.0f - std::pow(beta2_, static_cast<scalar_t>(step_count_));

  for (size_t i = 0; i < parameters_.size(); i++) {
    auto &p = parameters_[i];
    auto data = p->data();
    auto grad = p->grad();
    auto &m = m_[i];
    auto &v = v_[i];
    for (size_t j = 0; j < p->size(); j++) {
      scalar_t g = grad[j] + weight_decay_ * data[j];
      m[j] = beta1_ * m[j] + (1.0f - beta1_) * g;
      v[j] = beta2_ * v[j] + (1.0f - beta2_) * g * g;
      scalar_t m_hat = m[j] / bias_correction1;
      scalar_t v_hat = v[j] / bias_correction2;
      data[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + eps_);
    }
  }
}

namespace {

void require_float32(const Tensor &p, const char *name) {
  if (p.dtype() != DType::kFloat32 ||
      p.grad_storage().dtype() != DType::kFloat32) {
    throw std::invalid_argument(std::string(name) +
                                " requires float32 parameters and gradients");
  }
}

void match_device(Tensor &buffer, const Tensor &param) {
  if (buffer.backend() != param.backend()) {
    buffer.to(param.backend());
  }
}

BufferState save_buffer(const Tensor &buffer) {
  Storage host = buffer.data_storage().copy_to(Device::CPU);
  const auto *values = static_cast<const scalar_t *>(host.data());
  return {buffer.shape(),
          std::vector<scalar_t>(values, values + buffer.size())};
}

std::vector<std::shared_ptr<Tensor>> load_buffers(
    const std::vector<BufferState> &buffers,
    const std::vector<std::shared_ptr<Tensor>> &parameters, const char *name) {
  if (buffers.size() != parameters.size()) {
    throw std::invalid_argument(std::string(name) +
                                " state has the wrong number of buffers");
  }
  std::vector<std::shared_ptr<Tensor>> loaded;
  loaded.reserve(buffers.size());
  for (size_t i = 0; i < buffers.size(); i++) {
    const BufferState &buffer = buffers[i];
    if (buffer.shape != parameters[i]->shape() ||
        buffer.values.size() != parameters[i]->size()) {
      throw std::invalid_argument(std::string(name) +
                                  " state buffer does not match its parameter");
    }
    loaded.push_back(Tensor::from_values(buffer.shape, buffer.values));
  }
  return loaded;
}

}  // namespace

AdamW::AdamW(std::vector<std::shared_ptr<Tensor>> parameters,
             scalar_t learning_rate, std::pair<scalar_t, scalar_t> betas,
             scalar_t eps, scalar_t weight_decay)
    : AdamW(parameters, std::vector<bool>(parameters.size(), true),
            learning_rate, betas, eps, weight_decay) {}

AdamW::AdamW(std::vector<std::shared_ptr<Tensor>> parameters,
             std::vector<bool> decay, scalar_t learning_rate,
             std::pair<scalar_t, scalar_t> betas, scalar_t eps,
             scalar_t weight_decay)
    : Optimizer(std::move(parameters), learning_rate),
      beta1_(betas.first),
      beta2_(betas.second),
      eps_(eps),
      weight_decay_(weight_decay),
      decay_(std::move(decay)) {
  require_beta("beta1", beta1_);
  require_beta("beta2", beta2_);
  require_nonnegative("eps", eps_);
  require_nonnegative("weight_decay", weight_decay_);
  if (decay_.size() != parameters_.size()) {
    throw std::invalid_argument("AdamW needs one decay flag per parameter");
  }
  m_.reserve(parameters_.size());
  v_.reserve(parameters_.size());
  for (auto &p : parameters_) {
    m_.push_back(Tensor::zeros(p->shape(), p->backend()));
    v_.push_back(Tensor::zeros(p->shape(), p->backend()));
  }
}

void AdamW::step() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    require_float32(*parameters_[i], "AdamW");
    match_device(*m_[i], *parameters_[i]);
    match_device(*v_[i], *parameters_[i]);
  }

  step_count_++;
  scalar_t bias_correction1 =
      1.0f - std::pow(beta1_, static_cast<scalar_t>(step_count_));
  scalar_t bias_correction2 =
      1.0f - std::pow(beta2_, static_cast<scalar_t>(step_count_));

#ifdef MICROGRAD_CUDA_ENABLED
  std::vector<cuda::ops::AdamWTensor> cuda_tensors;
#endif

  for (size_t i = 0; i < parameters_.size(); i++) {
    auto &p = parameters_[i];
    if (p->backend() != Device::CPU) {
#ifdef MICROGRAD_CUDA_ENABLED
      if (p->backend() == Device::CUDA) {
        cuda_tensors.push_back(cuda::ops::AdamWTensor{
            static_cast<scalar_t *>(p->data_storage().device_pointer()),
            static_cast<const scalar_t *>(p->grad_storage().device_pointer()),
            static_cast<scalar_t *>(m_[i]->data_storage().device_pointer()),
            static_cast<scalar_t *>(v_[i]->data_storage().device_pointer()),
            p->size(), decay_[i]});
        continue;
      }
#endif
      throw std::runtime_error("AdamW: unsupported device");
    }
    ops::AdamWStep(*p, *m_[i], *v_[i], learning_rate_, beta1_, beta2_, eps_,
                   weight_decay_, decay_[i], bias_correction1,
                   bias_correction2);
  }

#ifdef MICROGRAD_CUDA_ENABLED
  if (!cuda_tensors.empty()) {
    cuda::ops::FusedAdamWStep(cuda_tensors, learning_rate_, beta1_, beta2_,
                              eps_, weight_decay_, bias_correction1,
                              bias_correction2);
  }
#endif
}

AdamWState AdamW::state() const {
  AdamWState saved;
  saved.step_count = step_count_;
  for (size_t i = 0; i < parameters_.size(); i++) {
    saved.m.push_back(save_buffer(*m_[i]));
    saved.v.push_back(save_buffer(*v_[i]));
  }
  return saved;
}

void AdamW::load_state(const AdamWState &saved) {
  auto m = load_buffers(saved.m, parameters_, "AdamW");
  auto v = load_buffers(saved.v, parameters_, "AdamW");
  m_ = std::move(m);
  v_ = std::move(v);
  step_count_ = saved.step_count;
}

Muon::Muon(std::vector<std::shared_ptr<Tensor>> parameters,
           scalar_t learning_rate, scalar_t momentum, scalar_t weight_decay,
           bool nesterov, size_t ns_steps)
    : Optimizer(std::move(parameters), learning_rate),
      momentum_(momentum),
      weight_decay_(weight_decay),
      nesterov_(nesterov),
      ns_steps_(ns_steps) {
  require_nonnegative("momentum", momentum_);
  require_nonnegative("weight_decay", weight_decay_);
  momentum_buffer_.reserve(parameters_.size());
  for (auto &p : parameters_) {
    if (p->shape().size() != 2) {
      throw std::invalid_argument("Muon requires 2-D parameters");
    }
    momentum_buffer_.push_back(Tensor::zeros(p->shape(), p->backend()));
  }
}

void Muon::step() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    auto &p = parameters_[i];
    auto &buf = momentum_buffer_[i];
    match_device(*buf, *p);
    const scalar_t fan_in = static_cast<scalar_t>(p->shape()[0]);
    const scalar_t fan_out = static_cast<scalar_t>(p->shape()[1]);
    const scalar_t scale = std::sqrt(std::max(1.0f, fan_out / fan_in));
    ops::MuonStep(*p, *buf, learning_rate_, momentum_, weight_decay_,
                  nesterov_, ns_steps_, scale);
  }
  step_count_++;
}

MuonState Muon::state() const {
  MuonState saved;
  saved.step_count = step_count_;
  for (const auto &buffer : momentum_buffer_) {
    saved.momentum.push_back(save_buffer(*buffer));
  }
  return saved;
}

void Muon::load_state(const MuonState &saved) {
  momentum_buffer_ = load_buffers(saved.momentum, parameters_, "Muon");
  step_count_ = saved.step_count;
}

}  // namespace micrograd
