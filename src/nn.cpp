#include "micrograd/nn.h"
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