#include "micrograd/nn/Optim.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

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

Optimizer::Optimizer(std::vector<std::shared_ptr<Tensor>> parameters)
    : parameters_(std::move(parameters)) {
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

void Optimizer::zero_grad() {
  for (auto &p : parameters_) {
    p->zero_grad();
  }
}

SGD::SGD(std::vector<std::shared_ptr<Tensor>> parameters,
         scalar_t learning_rate, scalar_t momentum, scalar_t weight_decay,
         bool nesterov)
    : Optimizer(std::move(parameters)),
      learning_rate_(learning_rate),
      momentum_(momentum),
      weight_decay_(weight_decay),
      nesterov_(nesterov) {
  require_nonnegative("learning_rate", learning_rate_);
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
    : Optimizer(std::move(parameters)),
      learning_rate_(learning_rate),
      beta1_(betas.first),
      beta2_(betas.second),
      eps_(eps),
      weight_decay_(weight_decay) {
  require_nonnegative("learning_rate", learning_rate_);
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

scalar_t read_master_weight(const Storage &storage, size_t index) {
  if (storage.dtype() == DType::kBFloat16) {
    uint16_t bits = static_cast<const uint16_t *>(storage.data())[index];
    uint32_t widened = static_cast<uint32_t>(bits) << 16;
    scalar_t value = 0;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
  }
  return static_cast<const scalar_t *>(storage.data())[index];
}

void write_master_weight(Storage &storage, size_t index, scalar_t value) {
  if (storage.dtype() == DType::kBFloat16) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    static_cast<uint16_t *>(storage.data())[index] =
        static_cast<uint16_t>(rounded >> 16);
    return;
  }
  static_cast<scalar_t *>(storage.data())[index] = value;
}

Storage zero_moment(size_t n, Device device) {
  Storage host(n * sizeof(scalar_t), Device::CPU);
  std::memset(host.data(), 0, host.bytes());
  if (device == Device::CPU) {
    return host;
  }
  return host.copy_to(device);
}

}  // namespace

AdamW::AdamW(std::vector<std::shared_ptr<Tensor>> parameters,
             scalar_t learning_rate, std::pair<scalar_t, scalar_t> betas,
             scalar_t eps, scalar_t weight_decay)
    : Optimizer(std::move(parameters)),
      learning_rate_(learning_rate),
      beta1_(betas.first),
      beta2_(betas.second),
      eps_(eps),
      weight_decay_(weight_decay) {
  require_nonnegative("learning_rate", learning_rate_);
  require_beta("beta1", beta1_);
  require_beta("beta2", beta2_);
  require_nonnegative("eps", eps_);
  require_nonnegative("weight_decay", weight_decay_);
  m_.reserve(parameters_.size());
  v_.reserve(parameters_.size());
  master_.resize(parameters_.size());
  for (size_t i = 0; i < parameters_.size(); i++) {
    auto &p = parameters_[i];
    m_.push_back(zero_moment(p->size(), p->backend()));
    v_.push_back(zero_moment(p->size(), p->backend()));
    if (p->backend() == Device::CPU &&
        p->data_storage().dtype() == DType::kBFloat16) {
      std::vector<scalar_t> weights(p->size());
      const Storage &storage = p->data_storage();
      for (size_t j = 0; j < p->size(); j++) {
        weights[j] = read_master_weight(storage, j);
      }
      master_[i] = std::move(weights);
    }
  }
}

void AdamW::step() {
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
            static_cast<scalar_t *>(m_[i].device_pointer()),
            static_cast<scalar_t *>(v_[i].device_pointer()), p->size(), true});
        continue;
      }
#endif
      throw std::runtime_error("AdamW: unsupported device");
    }
    Storage &storage = p->data_storage();
    const bool bf16 = storage.dtype() == DType::kBFloat16;
    auto *weights = bf16 ? master_[i].data()
                         : static_cast<scalar_t *>(storage.data());
    auto *m = static_cast<scalar_t *>(m_[i].data());
    auto *v = static_cast<scalar_t *>(v_[i].data());
    auto grad = p->grad();
    for (size_t j = 0; j < p->size(); j++) {
      scalar_t weight =
          weights[j] - learning_rate_ * weight_decay_ * weights[j];
      m[j] = beta1_ * m[j] + (1.0f - beta1_) * grad[j];
      v[j] = beta2_ * v[j] + (1.0f - beta2_) * grad[j] * grad[j];
      scalar_t m_hat = m[j] / bias_correction1;
      scalar_t v_hat = v[j] / bias_correction2;
      weight -= learning_rate_ * m_hat / (std::sqrt(v_hat) + eps_);
      weights[j] = weight;
      if (bf16) {
        write_master_weight(storage, j, weight);
      }
    }
  }

#ifdef MICROGRAD_CUDA_ENABLED
  if (!cuda_tensors.empty()) {
    cuda::ops::FusedAdamWStep(cuda_tensors, learning_rate_, beta1_, beta2_,
                              eps_, weight_decay_, bias_correction1,
                              bias_correction2);
  }
#endif
}

namespace {

std::vector<scalar_t> transpose_matrix(const std::vector<scalar_t> &m,
                                       size_t rows, size_t cols) {
  std::vector<scalar_t> result(rows * cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      result[j * rows + i] = m[i * cols + j];
    }
  }
  return result;
}

void matmul_transpose_b(const std::vector<scalar_t> &a, size_t a_rows,
                        size_t a_cols, const std::vector<scalar_t> &b,
                        size_t b_rows, std::vector<scalar_t> &out) {
  for (size_t i = 0; i < a_rows; i++) {
    for (size_t j = 0; j < b_rows; j++) {
      scalar_t sum = 0;
      for (size_t k = 0; k < a_cols; k++) {
        sum += a[i * a_cols + k] * b[j * a_cols + k];
      }
      out[i * b_rows + j] = sum;
    }
  }
}

void matmul(const std::vector<scalar_t> &a, size_t a_rows, size_t a_cols,
            const std::vector<scalar_t> &b, size_t b_cols,
            std::vector<scalar_t> &out) {
  for (size_t i = 0; i < a_rows; i++) {
    for (size_t j = 0; j < b_cols; j++) {
      scalar_t sum = 0;
      for (size_t k = 0; k < a_cols; k++) {
        sum += a[i * a_cols + k] * b[k * b_cols + j];
      }
      out[i * b_cols + j] = sum;
    }
  }
}

std::vector<scalar_t> zeropower_via_newton_schulz5(
    const std::vector<scalar_t> &g, size_t rows, size_t cols, size_t steps) {
  constexpr scalar_t a = 3.4445f;
  constexpr scalar_t b = -4.7750f;
  constexpr scalar_t c = 2.0315f;

  scalar_t norm = 0;
  for (auto v : g) {
    norm += v * v;
  }
  norm = std::sqrt(norm) + static_cast<scalar_t>(1e-7);

  bool wide = rows <= cols;
  size_t r = wide ? rows : cols;
  size_t cc = wide ? cols : rows;
  std::vector<scalar_t> x = wide ? g : transpose_matrix(g, rows, cols);
  for (auto &v : x) {
    v /= norm;
  }

  std::vector<scalar_t> a_mat(r * r);
  std::vector<scalar_t> a2(r * r);
  std::vector<scalar_t> b_mat(r * r);
  std::vector<scalar_t> bx(r * cc);
  for (size_t step = 0; step < steps; step++) {
    matmul_transpose_b(x, r, cc, x, r, a_mat);
    matmul(a_mat, r, r, a_mat, r, a2);
    for (size_t i = 0; i < r * r; i++) {
      b_mat[i] = b * a_mat[i] + c * a2[i];
    }
    matmul(b_mat, r, r, x, cc, bx);
    for (size_t i = 0; i < r * cc; i++) {
      x[i] = a * x[i] + bx[i];
    }
  }
  return wide ? x : transpose_matrix(x, r, cc);
}

}  // namespace

Muon::Muon(std::vector<std::shared_ptr<Tensor>> parameters,
           scalar_t learning_rate, scalar_t momentum, scalar_t weight_decay,
           bool nesterov, size_t ns_steps)
    : Optimizer(std::move(parameters)),
      learning_rate_(learning_rate),
      momentum_(momentum),
      weight_decay_(weight_decay),
      nesterov_(nesterov),
      ns_steps_(ns_steps) {
  require_nonnegative("learning_rate", learning_rate_);
  require_nonnegative("momentum", momentum_);
  require_nonnegative("weight_decay", weight_decay_);
  momentum_buffer_.reserve(parameters_.size());
  for (auto &p : parameters_) {
    if (p->shape().size() != 2) {
      throw std::invalid_argument("Muon requires 2-D parameters");
    }
    momentum_buffer_.emplace_back(p->size(), static_cast<scalar_t>(0));
  }
}

void Muon::step() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    auto &p = parameters_[i];
    auto &buf = momentum_buffer_[i];
    size_t rows = p->shape()[0];
    size_t cols = p->shape()[1];
    std::vector<scalar_t> update(p->size());
    for (size_t j = 0; j < p->size(); j++) {
      scalar_t grad = p->grad()[j];
      buf[j] = momentum_ * buf[j] + (1.0f - momentum_) * grad;
      update[j] =
          nesterov_ ? momentum_ * buf[j] + (1.0f - momentum_) * grad : buf[j];
    }
    update = zeropower_via_newton_schulz5(update, rows, cols, ns_steps_);
    scalar_t scale = std::sqrt(
        std::max(static_cast<scalar_t>(1),
                 static_cast<scalar_t>(rows) / static_cast<scalar_t>(cols)));
    for (size_t j = 0; j < p->size(); j++) {
      p->data()[j] -= learning_rate_ * weight_decay_ * p->data()[j];
      p->data()[j] -= learning_rate_ * scale * update[j];
    }
  }
  step_count_++;
}

}  // namespace micrograd
