#include "micrograd/nn/Layers.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "micrograd/Init.h"
#include "micrograd/Random.h"
#include "micrograd/nn/Functional.h"

namespace micrograd {

Linear::Linear(size_t in_features, size_t out_features, bool bias) {
  weights_ = Tensor::zeros(std::vector<size_t>{in_features, out_features});
  init::normal_(weights_, 0.0f,
                std::sqrt(1.0f / static_cast<scalar_t>(in_features)));
  weights_->set_requires_grad(true);
  register_parameter("weight", weights_);

  if (bias) {
    bias_ = Tensor::zeros(std::vector<size_t>{out_features});
    bias_->set_requires_grad(true);
    register_parameter("bias", bias_);
  }
}

std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor> &input) {
  auto output = input->matmul(weights_);
  return bias_ ? output->add_bias(bias_) : output;
}

std::shared_ptr<Tensor> Linear::weights() { return weights_; }
std::shared_ptr<Tensor> Linear::bias() { return bias_; }

Embedding::Embedding(size_t num_embeddings, size_t dim) {
  std::uniform_real_distribution<scalar_t> dis(-0.1f, 0.1f);

  std::vector<scalar_t> w_data(num_embeddings * dim);
  for (auto &w : w_data) {
    w = dis(global_rng());
  }

  weight_ =
      Tensor::from_values(std::vector<size_t>{num_embeddings, dim}, w_data);
  weight_->set_requires_grad(true);

  register_parameter("weight", weight_);
}

std::shared_ptr<Tensor> Embedding::forward(
    const std::shared_ptr<Tensor> &input) {
  return weight_->embedding_lookup(input);
}

std::shared_ptr<Tensor> Embedding::weight() { return weight_; }

LMHead::LMHead(std::shared_ptr<Tensor> embedding_weight, bool tied)
    : embedding_weight_(std::move(embedding_weight)), tied_(tied) {
  if (!tied_) {
    auto shape = embedding_weight_->shape();
    weight_ = Tensor::zeros(shape);
    init::kaiming_uniform_(weight_, shape[1]);
    weight_->set_requires_grad(true);

    register_parameter("weight", weight_);
  }
}

std::shared_ptr<Tensor> LMHead::forward(const std::shared_ptr<Tensor> &input) {
  const auto &weight = tied_ ? embedding_weight_ : weight_;
  return input->matmul(weight->transpose(0, 1));
}

std::shared_ptr<Tensor> LMHead::weight() {
  return tied_ ? embedding_weight_ : weight_;
}

LayerNorm::LayerNorm(std::vector<size_t> normalized_shape, scalar_t eps)
    : normalized_shape_(std::move(normalized_shape)), eps_(eps) {
  size_t count = 1;
  for (auto dim : normalized_shape_) {
    count *= dim;
  }

  gain_ = Tensor::from_values(normalized_shape_,
                              std::vector<scalar_t>(count, 1.0f));
  bias_ = Tensor::from_values(normalized_shape_,
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

RMSNorm::RMSNorm(std::vector<size_t> normalized_shape, scalar_t eps)
    : normalized_shape_(std::move(normalized_shape)), eps_(eps) {
  size_t count = 1;
  for (auto dim : normalized_shape_) {
    count *= dim;
  }

  gain_ = Tensor::from_values(normalized_shape_,
                              std::vector<scalar_t>(count, 1.0f));
  gain_->set_requires_grad(true);

  register_parameter("gain", gain_);
}

std::shared_ptr<Tensor> RMSNorm::forward(const std::shared_ptr<Tensor> &input) {
  return input->rms_norm(normalized_shape_, gain_, eps_);
}

std::shared_ptr<Tensor> RMSNorm::gain() { return gain_; }

std::shared_ptr<Tensor> ReLU::forward(const std::shared_ptr<Tensor> &input) {
  return input->relu();
}

std::shared_ptr<Tensor> Sigmoid::forward(const std::shared_ptr<Tensor> &input) {
  return input->sigmoid();
}

std::shared_ptr<Tensor> Tanh::forward(const std::shared_ptr<Tensor> &input) {
  return input->tanh();
}

std::shared_ptr<Tensor> GELU::forward(const std::shared_ptr<Tensor> &input) {
  return gelu(input);
}

Softmax::Softmax(int64_t dim) : dim_(dim) {}

std::shared_ptr<Tensor> Softmax::forward(const std::shared_ptr<Tensor> &input) {
  return input->softmax(dim_);
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

  auto mask = Tensor::from_values(input->shape(), mask_data);
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

}  // namespace micrograd
