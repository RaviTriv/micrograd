#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "Tensor.h"
#include "micrograd/nn/Module.h"

namespace micrograd {

std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor> &prediction,
                                 const std::shared_ptr<Tensor> &target);
std::shared_ptr<Tensor> cross_entropy(
    const std::shared_ptr<Tensor> &logits,
    const std::vector<size_t> &target_indices);
std::shared_ptr<Tensor> avg_pool_2x2(const std::shared_ptr<Tensor> &input);
std::shared_ptr<Tensor> gelu(const std::shared_ptr<Tensor> &input);

class Linear : public nn::Module {
 public:
  Linear(size_t in_features, size_t out_features);
  std::shared_ptr<Tensor> forward(
      const std::shared_ptr<Tensor> &input) override;
  std::shared_ptr<Tensor> weights();
  std::shared_ptr<Tensor> bias();

 private:
  std::shared_ptr<Tensor> weights_;
  std::shared_ptr<Tensor> bias_;
};

class Embedding : public nn::Module {
 public:
  Embedding(size_t num_embeddings, size_t dim);
  std::shared_ptr<Tensor> forward(
      const std::shared_ptr<Tensor> &input) override;
  std::shared_ptr<Tensor> weight();

 private:
  std::shared_ptr<Tensor> weight_;
};

class LayerNorm : public nn::Module {
 public:
  LayerNorm(std::vector<size_t> normalized_shape, scalar_t eps = 1e-5f);
  std::shared_ptr<Tensor> forward(
      const std::shared_ptr<Tensor> &input) override;
  std::shared_ptr<Tensor> gain();
  std::shared_ptr<Tensor> bias();

 private:
  std::vector<size_t> normalized_shape_;
  scalar_t eps_;
  std::shared_ptr<Tensor> gain_;
  std::shared_ptr<Tensor> bias_;
};

class ReLU : public nn::Module {
 public:
  std::shared_ptr<Tensor> forward(
      const std::shared_ptr<Tensor> &input) override;
};

class Dropout : public nn::Module {
 public:
  explicit Dropout(scalar_t p);
  std::shared_ptr<Tensor> forward(
      const std::shared_ptr<Tensor> &input) override;

 private:
  scalar_t p_;
};

class Sequential : public nn::Module {
 public:
  explicit Sequential(std::vector<std::shared_ptr<nn::Module>> layers);
  std::shared_ptr<Tensor> forward(
      const std::shared_ptr<Tensor> &input) override;

 private:
  std::vector<std::shared_ptr<nn::Module>> layers_;
};

class SGD {
 public:
  SGD(std::vector<std::shared_ptr<Tensor>> parameters, scalar_t learning_rate,
      scalar_t momentum = 0, scalar_t weight_decay = 0, bool nesterov = false);

  void step();
  void zero_grad();

 private:
  std::vector<std::shared_ptr<Tensor>> parameters_;
  scalar_t learning_rate_;
  scalar_t momentum_;
  scalar_t weight_decay_;
  bool nesterov_;
  std::vector<std::vector<scalar_t>> velocity_;
};

void save_model(const std::string &path, Linear &l1, Linear &l2);
void load_model(const std::string &path, Linear &l1, Linear &l2);

}  // namespace micrograd
