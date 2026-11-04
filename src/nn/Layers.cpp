#include "micrograd/nn/Layers.h"

#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "micrograd/Init.h"
#include "micrograd/Random.h"

namespace micrograd {

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

LMHead::LMHead(std::shared_ptr<Tensor> embedding_weight, bool tied)
    : embedding_weight_(std::move(embedding_weight)), tied_(tied) {
  if (!tied_) {
    auto shape = embedding_weight_->shape();
    weight_ = std::make_shared<Tensor>(shape);
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

RMSNorm::RMSNorm(std::vector<size_t> normalized_shape, scalar_t eps)
    : normalized_shape_(std::move(normalized_shape)), eps_(eps) {
  size_t count = 1;
  for (auto dim : normalized_shape_) {
    count *= dim;
  }

  gain_ = std::make_shared<Tensor>(normalized_shape_,
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

}  // namespace micrograd
