#include "micrograd/nn.h"
#include <fstream>
#include <memory>
#include <random>

std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor> &prediction,
                                 const std::shared_ptr<Tensor> &target) {
  auto diff = prediction->sub(target);
  auto squared = diff->pow(2.0);

  auto sum = squared->sum();
  auto mean = sum->div(static_cast<double>(prediction->size()));

  return mean;
}

std::shared_ptr<Tensor> avg_pool_2x2(const std::shared_ptr<Tensor> &input) {
  std::vector<double> pooled(196);

  for (int py = 0; py < 14; py++) {
    for (int px = 0; px < 14; px++) {
      double sum = 0.0;
      for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
          int y = py * 2 + dy;
          int x = px * 2 + dx;
          sum += input->at({0, static_cast<size_t>(y * 28 + x)});
        }
      }
      pooled[static_cast<size_t>(py * 14 + px)] = sum / 4.0;
    }
  }

  return std::make_shared<Tensor>(std::vector<size_t>{1, 196}, pooled);
}

Linear::Linear(size_t in_features, size_t out_features) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-0.1, 0.1);

  std::vector<double> w_data(in_features * out_features);
  for (auto &w : w_data) {
    w = dis(gen);
  }

  weights_ = std::make_shared<Tensor>(
      std::vector<size_t>{in_features, out_features}, w_data);

  std::vector<double> b_data(out_features, 0.0);
  bias_ =
      std::make_shared<Tensor>(std::vector<size_t>{1, out_features}, b_data);
}

std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor> &input) {
  return input->matmul(weights_)->add(bias_);
}

std::shared_ptr<Tensor> Linear::weights() { return weights_; }
std::shared_ptr<Tensor> Linear::bias() { return bias_; }

SGD::SGD(std::vector<std::shared_ptr<Tensor>> parameters, double learning_rate)
    : parameters_(parameters), learning_rate_(learning_rate) {}

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

  auto &w1 = l1.weights()->data();
  auto &b1 = l1.bias()->data();
  auto &w2 = l2.weights()->data();
  auto &b2 = l2.bias()->data();

  file.write(reinterpret_cast<char *>(w1.data()),
             static_cast<std::streamsize>(w1.size() * sizeof(double)));
  file.write(reinterpret_cast<char *>(b1.data()),
             static_cast<std::streamsize>(b1.size() * sizeof(double)));
  file.write(reinterpret_cast<char *>(w2.data()),
             static_cast<std::streamsize>(w2.size() * sizeof(double)));
  file.write(reinterpret_cast<char *>(b2.data()),
             static_cast<std::streamsize>(b2.size() * sizeof(double)));

  file.close();
}

void load_model(const std::string &path, Linear &l1, Linear &l2) {
  std::ifstream file(path, std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for loading: " + path);
  }

  auto &w1 = l1.weights()->data();
  auto &b1 = l1.bias()->data();
  auto &w2 = l2.weights()->data();
  auto &b2 = l2.bias()->data();

  file.read(reinterpret_cast<char *>(w1.data()),
            static_cast<std::streamsize>(w1.size() * sizeof(double)));
  file.read(reinterpret_cast<char *>(b1.data()),
            static_cast<std::streamsize>(b1.size() * sizeof(double)));
  file.read(reinterpret_cast<char *>(w2.data()),
            static_cast<std::streamsize>(w2.size() * sizeof(double)));
  file.read(reinterpret_cast<char *>(b2.data()),
            static_cast<std::streamsize>(b2.size() * sizeof(double)));

  file.close();
}