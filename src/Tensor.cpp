#include "micrograd/Tensor.h"
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>

#ifdef MICROGRAD_METAL_ENABLED
#include "micrograd/metal/MetalContext.h"
#endif

std::string format_scalar(const std::string &op, double scalar);
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

#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal &&
      b->backend_ == micrograd::Backend::Metal) {
    return add_metal(b);
  }
#endif

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

#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal &&
      b->backend_ == micrograd::Backend::Metal) {
    return sub_metal(b);
  }
#endif

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

#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal &&
      b->backend_ == micrograd::Backend::Metal) {
    return mul_metal(b);
  }
#endif

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

#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal &&
      b->backend_ == micrograd::Backend::Metal) {
    return div_metal(b);
  }
#endif

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
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal) {
    return add_scalar_metal(scalar);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = format_scalar("+", scalar);

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
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal) {
    return sub_scalar_metal(scalar);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = format_scalar("-", scalar);

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
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal) {
    return mul_scalar_metal(scalar);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = format_scalar("*", scalar);

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
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal) {
    return div_scalar_metal(scalar);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = format_scalar("/", scalar);

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
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal) {
    return pow_metal(exponent);
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = format_scalar("^", exponent);

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
  result->op_ = "sum";

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

#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal &&
      b->backend_ == micrograd::Backend::Metal) {
    return matmul_metal(b);
  }
#endif

  size_t m = shape_[0];
  size_t k = shape_[1];
  size_t n = b->shape_[1];

  auto result = std::make_shared<Tensor>(std::vector<size_t>{m, n});
  result->op_ = "@";

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

std::shared_ptr<Tensor> Tensor::relu() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal) {
    return relu_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "relu";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = data_[i] > 0 ? data_[i] : 0;
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      self_ptr->grad_[i] +=
          result->grad_[i] * (self_ptr->data_[i] > 0 ? 1.0 : 0.0);
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal) {
    return sigmoid_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "sigmoid";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = 1.0 / (1.0 + std::exp(-data_[i]));
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      double sigmoid_val = result->data_[i];
      self_ptr->grad_[i] +=
          result->grad_[i] * sigmoid_val * (1.0 - sigmoid_val);
    }
  };

  return result;
}

std::shared_ptr<Tensor> Tensor::tanh() {
#ifdef MICROGRAD_METAL_ENABLED
  if (backend_ == micrograd::Backend::Metal) {
    return tanh_metal();
  }
#endif

  auto result = std::make_shared<Tensor>(shape_);
  result->op_ = "tanh";

  for (size_t i = 0; i < data_.size(); i++) {
    result->data_[i] = std::tanh(data_[i]);
  }

  auto self_ptr = shared_from_this();
  result->children_ = {self_ptr};

  result->backward_fn_ = [result, self_ptr]() {
    for (size_t i = 0; i < self_ptr->grad_.size(); i++) {
      double tanh_val = result->data_[i];
      self_ptr->grad_[i] += result->grad_[i] * (1.0 - tanh_val * tanh_val);
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

std::string format_scalar(const std::string &op, double scalar) {
  std::ostringstream oss;
  oss << op << " " << std::fixed << std::setprecision(2) << scalar;
  return oss.str();
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

std::vector<double> &Tensor::data() { return data_; }

std::vector<double> &Tensor::grad() { return grad_; }

void Tensor::to(micrograd::Backend b) {
  if (backend_ == b) {
    return;
  }

#ifdef MICROGRAD_METAL_ENABLED
  if (b == micrograd::Backend::Metal) {
    auto &ctx = MetalContext::instance();
    if (!ctx.isAvailable()) {
      ctx.initialize();
    }

    gpu_data_ = ctx.createBuffer(size() * sizeof(float));
    gpu_grad_ = ctx.createBuffer(size() * sizeof(float));

    float *gpu_ptr = static_cast<float *>(gpu_data_->contents());
    for (size_t i = 0; i < size(); i++) {
      gpu_ptr[i] = static_cast<float>(data_[i]);
    }

    float *grad_ptr = static_cast<float *>(gpu_grad_->contents());
    for (size_t i = 0; i < size(); i++) {
      grad_ptr[i] = 0.0f;
    }

    backend_ = micrograd::Backend::Metal;
  } else {
    float *gpu_ptr = static_cast<float *>(gpu_data_->contents());
    for (size_t i = 0; i < size(); i++) {
      data_[i] = static_cast<double>(gpu_ptr[i]);
    }

    float *grad_ptr = static_cast<float *>(gpu_grad_->contents());
    for (size_t i = 0; i < size(); i++) {
      grad_[i] = static_cast<double>(grad_ptr[i]);
    }

    auto &ctx = MetalContext::instance();
    ctx.releaseBuffer(gpu_data_);
    ctx.releaseBuffer(gpu_grad_);
    gpu_data_ = nullptr;
    gpu_grad_ = nullptr;

    backend_ = micrograd::Backend::CPU;
  }
#else
  (void)b;
#endif
}

micrograd::Backend Tensor::backend() const { return backend_; }