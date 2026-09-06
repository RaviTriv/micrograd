#include "micrograd/NN.h"

#include <fstream>
#include <memory>
#include <random>
#include <utility>

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

Linear::Linear(size_t in_features, size_t out_features) {
  std::uniform_real_distribution<scalar_t> dis(-0.1f, 0.1f);

  std::vector<scalar_t> w_data(in_features * out_features);
  for (auto &w : w_data) {
    w = dis(global_rng());
  }

  weights_ = std::make_shared<Tensor>(
      std::vector<size_t>{in_features, out_features}, w_data);

  std::vector<scalar_t> b_data(out_features, 0.0f);
  bias_ =
      std::make_shared<Tensor>(std::vector<size_t>{1, out_features}, b_data);
}

std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor> &input) {
  return input->matmul(weights_)->add(bias_);
}

std::shared_ptr<Tensor> Linear::weights() { return weights_; }
std::shared_ptr<Tensor> Linear::bias() { return bias_; }

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
