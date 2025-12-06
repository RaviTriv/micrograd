#pragma once

#include <vector>
#include <memory>
#include <functional>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  Tensor(std::vector<size_t> shape);
  ~Tensor();
private:
  std::vector<double> data_;
  std::vector<double> grad_;
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;

  std::vector<std::shared_ptr<Tensor>> children_;
  std::function<void()> backward_fn_;
};
