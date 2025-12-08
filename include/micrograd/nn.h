#pragma once

#include "Tensor.h"
#include <memory>

std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor> &prediction,
                                 const std::shared_ptr<Tensor> &target);

class Linear {
public:
  Linear(size_t in_features, size_t out_features);
  std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input);
  std::shared_ptr<Tensor> weights();
  std::shared_ptr<Tensor> bias();

private:
  std::shared_ptr<Tensor> weights_;
  std::shared_ptr<Tensor> bias_;
};