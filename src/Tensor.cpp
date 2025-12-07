#include "micrograd/Tensor.h"
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_set>

Tensor::Tensor(std::vector<size_t> shape) : shape_(shape) {
  size_t total = 1;
  for (auto dim : shape_) {
    total *= dim;
  }
  data_.resize(total, 0.0);
  grad_.resize(total, 0.0);
  compute_strides();
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<double> data)
    : data_(data), shape_(shape) {
  size_t total = 1;
  for (auto dim : shape_) {
    total *= dim;
  }
  if (data_.size() != total) {
    throw std::invalid_argument("Tensor and data size mismatch");
  }
  grad_.resize(total, 0.0);
  compute_strides();
}

void Tensor::compute_strides() {
  strides_.resize(shape_.size());
  size_t stride = 1;
  for (size_t i = shape_.size(); i > 0; i--) {
    strides_[i - 1] = stride;
    stride *= shape_[i - 1];
  }
}

std::shared_ptr<Tensor> Tensor::add(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "+";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] + b->data_[i];
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result, self_ptr, b]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
      b->grad_[i] += result->grad_[i];
    }
  };
  return result;
}

std::shared_ptr<Tensor> Tensor::sub(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "-";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] - b->data_[i];
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result, self_ptr, b]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
      b->grad_[i] -= result->grad_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::mul(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "*";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] * b->data_[i];
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result, self_ptr, b]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] * b->data_[i];
      b->grad_[i] += result->grad_[i] * self_ptr->data_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::div(const std::shared_ptr<Tensor> &b) {
  if (shape_ != b->shape_) {
    throw std::invalid_argument("Tensor shapes do not match");
  }

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "/";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] / b->data_[i];
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result, self_ptr, b]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] / b->data_[i];
      b->grad_[i] -=
          result->grad_[i] * self_ptr->data_[i] / (b->data_[i] * b->data_[i]);
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::add(double scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "+";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] + scalar;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sub(double scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "-";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] - scalar;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::mul(double scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "*";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] * scalar;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr, scalar]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] * scalar;
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::div(double scalar) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "/";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] / scalar;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr, scalar]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] / scalar;
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::pow(double exponent) {
  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "^";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = std::pow(data_[i], exponent);
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr, exponent]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[i] * exponent *
                            std::pow(self_ptr->data_[i], exponent - 1);
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sum() {
  auto result = std::make_shared<Tensor>(std::vector<size_t>{1});

  double total = 0.0;
  for (size_t i = 0; i < data_.size(); i++) {
    total += data_[i];
  }
  result->data_[0] = total;

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] += result->grad_[0];
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor> &b) {
  if (shape_.size() != 2 || b->shape_.size() != 2) {
    throw std::invalid_argument("Tensors must be 2D for matmul");
  }

  if (shape_[1] != b->shape_[0]) {
    throw std::invalid_argument("Inner dimensions must match for matmul");
  }

  size_t m = shape_[0];
  size_t k = shape_[1];
  size_t n = b->shape_[1];

  auto result = std::make_shared<Tensor>(std::vector<size_t>{m, n});

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      double sum = 0.0;
      for (size_t p = 0; p < k; p++) {
        sum += at({i, p}) * b->at({p, j});
      }
      result->at({i, j}) = sum;
    }
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr, b};

  result->backward_fn_ = [result, self_ptr, b, m, k, n]() {
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        for (size_t p = 0; p < n; p++) {
          self_ptr->grad_at({i, j}) += result->grad_at({i, p}) * b->at({j, p});
        }
      }
    }

    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        for (size_t p = 0; p < m; p++) {
          b->grad_at({i, j}) += self_ptr->at({p, i}) * result->grad_at({p, j});
        }
      }
    }
  };

  return result;
}

void Tensor::backward() {
  std::vector<std::shared_ptr<Tensor>> ordered;
  std::unordered_set<Tensor *> visited;

  std::function<void(std::shared_ptr<Tensor>)> findOrder =
      [&](std::shared_ptr<Tensor> node) {
        if (visited.contains(node.get())) {
          return;
        }
        visited.insert(node.get());
        for (auto &child : node->children_) {
          findOrder(child);
        }
        ordered.push_back(node);
      };
  findOrder(shared_from_this());

  for (size_t i = 0; i < grad_.size(); i++) {
    grad_[i] = 1.0;
  }

  for (auto it = ordered.rbegin(); it != ordered.rend(); it++) {
    if ((*it)->backward_fn_) {
      (*it)->backward_fn_();
    }
  }
}

void Tensor::zero_grad() {
  for (size_t i = 0; i < grad_.size(); i++) {
    grad_[i] = 0.0;
  }
}

const std::vector<size_t> &Tensor::shape() const { return shape_; }

size_t Tensor::size() const { return data_.size(); }

double &Tensor::at(const std::vector<size_t> &indices) {
  return data_[flat_index(indices)];
}

double Tensor::at(const std::vector<size_t> &indices) const {
  return data_[flat_index(indices)];
}

double &Tensor::grad_at(const std::vector<size_t> &indices) {
  return grad_[flat_index(indices)];
}

double Tensor::grad_at(const std::vector<size_t> &indices) const {
  return grad_[flat_index(indices)];
}

size_t Tensor::flat_index(const std::vector<size_t> &indices) const {
  size_t idx = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    idx += indices[i] * strides_[i];
  }
  return idx;
}