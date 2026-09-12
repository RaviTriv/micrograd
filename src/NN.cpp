#include "micrograd/NN.h"

#include <cstddef>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "micrograd/Init.h"
#include "micrograd/Random.h"

namespace micrograd {

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
         scalar_t learning_rate)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate) {}

void SGD::zero_grad() {
  for (auto &p : parameters_) {
    p->zero_grad();
  }
}

void SGD::step() {
  for (auto &p : parameters_) {
    for (size_t i = 0; i < p->size(); i++) {
      p->data()[i] -= learning_rate_ * p->grad()[i];
    }
  }
}

void save_model(const std::string &path, Linear &l1, Linear &l2) {
  std::ofstream file(path, std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for saving: " + path);
  }

  auto w1 = l1.weights()->data();
  auto b1 = l1.bias()->data();
  auto w2 = l2.weights()->data();
  auto b2 = l2.bias()->data();

  file.write(reinterpret_cast<char *>(w1.data()),
             static_cast<std::streamsize>(w1.size() * sizeof(scalar_t)));
  file.write(reinterpret_cast<char *>(b1.data()),
             static_cast<std::streamsize>(b1.size() * sizeof(scalar_t)));
  file.write(reinterpret_cast<char *>(w2.data()),
             static_cast<std::streamsize>(w2.size() * sizeof(scalar_t)));
  file.write(reinterpret_cast<char *>(b2.data()),
             static_cast<std::streamsize>(b2.size() * sizeof(scalar_t)));

  file.close();
}

void load_model(const std::string &path, Linear &l1, Linear &l2) {
  std::ifstream file(path, std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for loading: " + path);
  }

  auto w1 = l1.weights()->data();
  auto b1 = l1.bias()->data();
  auto w2 = l2.weights()->data();
  auto b2 = l2.bias()->data();

  file.read(reinterpret_cast<char *>(w1.data()),
            static_cast<std::streamsize>(w1.size() * sizeof(scalar_t)));
  file.read(reinterpret_cast<char *>(b1.data()),
            static_cast<std::streamsize>(b1.size() * sizeof(scalar_t)));
  file.read(reinterpret_cast<char *>(w2.data()),
            static_cast<std::streamsize>(w2.size() * sizeof(scalar_t)));
  file.read(reinterpret_cast<char *>(b2.data()),
            static_cast<std::streamsize>(b2.size() * sizeof(scalar_t)));

  file.close();
}

}  // namespace micrograd
