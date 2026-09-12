#include "micrograd/NN.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "micrograd/Init.h"
#include "micrograd/Random.h"

namespace micrograd {

namespace {

constexpr uint32_t kStateDictMagic = 0x4d47534e;
constexpr uint32_t kStateDictVersion = 1;

std::string format_shape(const std::vector<size_t> &shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i > 0) {
      result += ", ";
    }
    result += std::to_string(shape[i]);
  }
  result += "]";
  return result;
}

void write_u32(std::ofstream &file, uint32_t value) {
  file.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_u64(std::ofstream &file, uint64_t value) {
  file.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

uint32_t read_u32(std::ifstream &file) {
  uint32_t value = 0;
  file.read(reinterpret_cast<char *>(&value), sizeof(value));
  return value;
}

uint64_t read_u64(std::ifstream &file) {
  uint64_t value = 0;
  file.read(reinterpret_cast<char *>(&value), sizeof(value));
  return value;
}

}  // namespace

std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor> &prediction,
                                 const std::shared_ptr<Tensor> &target) {
  auto diff = prediction->sub(target);
  auto squared = diff->pow(2.0f);

  auto sum = squared->sum();
  auto mean = sum->div(static_cast<scalar_t>(prediction->size()));

  return mean;
}

std::shared_ptr<Tensor> cross_entropy(
    const std::shared_ptr<Tensor> &logits,
    const std::vector<size_t> &target_indices) {
  if (logits->shape().size() != 2) {
    throw std::invalid_argument("cross_entropy expects rank 2 logits");
  }

  size_t batch = logits->shape()[0];
  size_t classes = logits->shape()[1];
  if (target_indices.size() != batch) {
    throw std::invalid_argument(
        "cross_entropy target count does not match the batch size");
  }

  std::vector<scalar_t> selected(batch * classes, 0.0f);
  for (size_t i = 0; i < batch; i++) {
    if (target_indices[i] >= classes) {
      throw std::out_of_range("cross_entropy target index is out of range");
    }
    selected[(i * classes) + target_indices[i]] = 1.0f;
  }

  auto selector =
      std::make_shared<Tensor>(std::vector<size_t>{batch, classes}, selected);
  selector->to(logits->backend());

  auto log_probs = logits->log_softmax(1);
  auto picked = log_probs->mul(selector);

  return picked->sum()->neg()->div(static_cast<scalar_t>(batch));
}

std::shared_ptr<Tensor> avg_pool_2x2(const std::shared_ptr<Tensor> &input) {
  std::vector<scalar_t> pooled(196);

  for (size_t py = 0; py < 14; py++) {
    for (size_t px = 0; px < 14; px++) {
      scalar_t sum = 0.0f;
      for (size_t dy = 0; dy < 2; dy++) {
        for (size_t dx = 0; dx < 2; dx++) {
          size_t y = (py * 2) + dy;
          size_t x = (px * 2) + dx;
          sum += input->at({0, (y * 28) + x});
        }
      }
      pooled[(py * 14) + px] = sum / 4.0f;
    }
  }

  return std::make_shared<Tensor>(std::vector<size_t>{1, 196}, pooled);
}

std::shared_ptr<Tensor> gelu(const std::shared_ptr<Tensor> &input) {
  constexpr scalar_t kSqrt2OverPi = 0.7978845608028654f;
  constexpr scalar_t kCubicCoeff = 0.044715f;

  auto cubed = input->pow(3.0f);
  auto inner = input->add(cubed->mul(kCubicCoeff))->mul(kSqrt2OverPi);
  auto gate = inner->tanh()->add(1.0f)->mul(0.5f);

  return input->mul(gate);
}

Linear::Linear(size_t in_features, size_t out_features) {
  weights_ =
      std::make_shared<Tensor>(std::vector<size_t>{in_features, out_features});
  init::kaiming_uniform_(weights_, in_features);

  std::vector<scalar_t> b_data(out_features, 0.0f);
  bias_ =
      std::make_shared<Tensor>(std::vector<size_t>{1, out_features}, b_data);

  weights_->set_requires_grad(true);
  bias_->set_requires_grad(true);

  register_parameter("weight", weights_);
  register_parameter("bias", bias_);
}

std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor> &input) {
  return input->matmul(weights_)->add(bias_);
}

std::shared_ptr<Tensor> Linear::weights() { return weights_; }
std::shared_ptr<Tensor> Linear::bias() { return bias_; }

Embedding::Embedding(size_t num_embeddings, size_t dim) {
  std::uniform_real_distribution<scalar_t> dis(-0.1f, 0.1f);

  std::vector<scalar_t> w_data(num_embeddings * dim);
  for (auto &w : w_data) {
    w = dis(global_rng());
  }

  weight_ = std::make_shared<Tensor>(std::vector<size_t>{num_embeddings, dim},
                                     w_data);
  weight_->set_requires_grad(true);

  register_parameter("weight", weight_);
}

std::shared_ptr<Tensor> Embedding::forward(
    const std::shared_ptr<Tensor> &input) {
  return weight_->embedding_lookup(input);
}

std::shared_ptr<Tensor> Embedding::weight() { return weight_; }

LayerNorm::LayerNorm(std::vector<size_t> normalized_shape, scalar_t eps)
    : normalized_shape_(std::move(normalized_shape)), eps_(eps) {
  size_t count = 1;
  for (auto dim : normalized_shape_) {
    count *= dim;
  }

  gain_ = std::make_shared<Tensor>(normalized_shape_,
                                   std::vector<scalar_t>(count, 1.0f));
  bias_ = std::make_shared<Tensor>(normalized_shape_,
                                   std::vector<scalar_t>(count, 0.0f));

  gain_->set_requires_grad(true);
  bias_->set_requires_grad(true);

  register_parameter("gain", gain_);
  register_parameter("bias", bias_);
}

std::shared_ptr<Tensor> LayerNorm::forward(
    const std::shared_ptr<Tensor> &input) {
  return input->layer_norm(normalized_shape_, gain_, bias_, eps_);
}

std::shared_ptr<Tensor> LayerNorm::gain() { return gain_; }
std::shared_ptr<Tensor> LayerNorm::bias() { return bias_; }

std::shared_ptr<Tensor> ReLU::forward(const std::shared_ptr<Tensor> &input) {
  return input->relu();
}

Dropout::Dropout(scalar_t p) : p_(p) {
  if (p_ < 0.0f || p_ >= 1.0f) {
    throw std::invalid_argument("Dropout probability must be in [0, 1)");
  }
}

std::shared_ptr<Tensor> Dropout::forward(const std::shared_ptr<Tensor> &input) {
  if (!is_training() || p_ == 0.0f) {
    return input;
  }

  std::bernoulli_distribution keep(1.0 - p_);
  scalar_t scale = 1.0f / (1.0f - p_);

  std::vector<scalar_t> mask_data(input->size());
  for (auto &value : mask_data) {
    value = keep(global_rng()) ? scale : 0.0f;
  }

  auto mask = std::make_shared<Tensor>(input->shape(), mask_data);
  mask->to(input->backend());

  return input->mul(mask);
}

Sequential::Sequential(std::vector<std::shared_ptr<nn::Module>> layers)
    : layers_(std::move(layers)) {
  for (size_t i = 0; i < layers_.size(); i++) {
    register_module(std::to_string(i), layers_[i]);
  }
}

std::shared_ptr<Tensor> Sequential::forward(
    const std::shared_ptr<Tensor> &input) {
  auto output = input;
  for (auto &layer : layers_) {
    output = layer->forward(output);
  }
  return output;
}

SGD::SGD(std::vector<std::shared_ptr<Tensor>> parameters,
         scalar_t learning_rate, scalar_t momentum, scalar_t weight_decay,
         bool nesterov)
    : parameters_(std::move(parameters)),
      learning_rate_(learning_rate),
      momentum_(momentum),
      weight_decay_(weight_decay),
      nesterov_(nesterov) {
  velocity_.reserve(parameters_.size());
  for (auto &p : parameters_) {
    velocity_.emplace_back(p->size(), scalar_t(0));
  }
}

void SGD::zero_grad() {
  for (auto &p : parameters_) {
    p->zero_grad();
  }
}

void SGD::step() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    auto &p = parameters_[i];
    auto &velocity = velocity_[i];
    for (size_t j = 0; j < p->size(); j++) {
      scalar_t grad = p->grad()[j] + weight_decay_ * p->data()[j];
      velocity[j] = momentum_ * velocity[j] + grad;
      scalar_t update =
          nesterov_ ? grad + momentum_ * velocity[j] : velocity[j];
      p->data()[j] -= learning_rate_ * update;
    }
  }
}

AdamW::AdamW(std::vector<std::shared_ptr<Tensor>> parameters,
             scalar_t learning_rate, std::pair<scalar_t, scalar_t> betas,
             scalar_t eps, scalar_t weight_decay)
    : parameters_(std::move(parameters)),
      learning_rate_(learning_rate),
      beta1_(betas.first),
      beta2_(betas.second),
      eps_(eps),
      weight_decay_(weight_decay) {
  m_.reserve(parameters_.size());
  v_.reserve(parameters_.size());
  for (auto &p : parameters_) {
    m_.emplace_back(p->size(), scalar_t(0));
    v_.emplace_back(p->size(), scalar_t(0));
  }
}

void AdamW::zero_grad() {
  for (auto &p : parameters_) {
    p->zero_grad();
  }
}

void AdamW::step() {
  step_count_++;
  scalar_t bias_correction1 =
      1.0f - std::pow(beta1_, static_cast<scalar_t>(step_count_));
  scalar_t bias_correction2 =
      1.0f - std::pow(beta2_, static_cast<scalar_t>(step_count_));
  for (size_t i = 0; i < parameters_.size(); i++) {
    auto &p = parameters_[i];
    auto &m = m_[i];
    auto &v = v_[i];
    for (size_t j = 0; j < p->size(); j++) {
      p->data()[j] -= learning_rate_ * weight_decay_ * p->data()[j];
      scalar_t grad = p->grad()[j];
      m[j] = beta1_ * m[j] + (1.0f - beta1_) * grad;
      v[j] = beta2_ * v[j] + (1.0f - beta2_) * grad * grad;
      scalar_t m_hat = m[j] / bias_correction1;
      scalar_t v_hat = v[j] / bias_correction2;
      p->data()[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + eps_);
    }
  }
}

void save(const std::string &path, nn::Module &module) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for saving: " + path);
  }

  auto named = module.named_parameters();

  write_u32(file, kStateDictMagic);
  write_u32(file, kStateDictVersion);
  write_u32(file, static_cast<uint32_t>(named.size()));

  for (auto &[name, tensor] : named) {
    write_u32(file, static_cast<uint32_t>(name.size()));
    file.write(name.data(), static_cast<std::streamsize>(name.size()));

    const auto &shape = tensor->shape();
    write_u32(file, static_cast<uint32_t>(shape.size()));
    for (auto dim : shape) {
      write_u64(file, static_cast<uint64_t>(dim));
    }

    auto data = tensor->data();
    file.write(reinterpret_cast<const char *>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(scalar_t)));
  }

  if (!file) {
    throw std::runtime_error("Failed while writing state dict: " + path);
  }
}

void load(const std::string &path, nn::Module &module) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for loading: " + path);
  }

  uint32_t magic = read_u32(file);
  if (magic != kStateDictMagic) {
    throw std::runtime_error("Not a micrograd state dict file: " + path);
  }

  uint32_t version = read_u32(file);
  if (version != kStateDictVersion) {
    throw std::runtime_error("Unsupported state dict version " +
                             std::to_string(version) + " in " + path);
  }

  uint32_t count = read_u32(file);

  std::unordered_map<std::string,
                     std::pair<std::vector<size_t>, std::vector<scalar_t>>>
      stored;
  stored.reserve(count);

  for (uint32_t i = 0; i < count; i++) {
    uint32_t name_len = read_u32(file);
    std::string name(name_len, '\0');
    file.read(name.data(), name_len);

    uint32_t ndim = read_u32(file);
    std::vector<size_t> shape(ndim);
    size_t total = 1;
    for (uint32_t d = 0; d < ndim; d++) {
      shape[d] = static_cast<size_t>(read_u64(file));
      total *= shape[d];
    }

    std::vector<scalar_t> values(total);
    file.read(reinterpret_cast<char *>(values.data()),
              static_cast<std::streamsize>(total * sizeof(scalar_t)));

    stored.emplace(std::move(name),
                   std::make_pair(std::move(shape), std::move(values)));
  }

  if (!file) {
    throw std::runtime_error("State dict file is truncated: " + path);
  }

  auto named = module.named_parameters();
  if (named.size() != stored.size()) {
    throw std::runtime_error("State dict parameter count mismatch: model has " +
                             std::to_string(named.size()) +
                             " parameters, file has " +
                             std::to_string(stored.size()));
  }

  for (auto &[name, tensor] : named) {
    auto it = stored.find(name);
    if (it == stored.end()) {
      throw std::runtime_error("State dict is missing parameter: " + name);
    }

    const auto &shape = it->second.first;
    const auto &values = it->second.second;
    if (shape != tensor->shape()) {
      throw std::runtime_error(
          "State dict shape mismatch for parameter " + name + ": model has " +
          format_shape(tensor->shape()) + ", file has " + format_shape(shape));
    }

    auto data = tensor->data();
    std::copy(values.begin(), values.end(), data.begin());
  }
}

}  // namespace micrograd
