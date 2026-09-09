#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"

namespace micrograd {
namespace {

std::vector<size_t> ResolveShape(const std::vector<int64_t> &shape,
                                 size_t total) {
  std::vector<size_t> resolved(shape.size());
  size_t inferred_count = 0;
  size_t inferred_position = 0;
  size_t known = 1;

  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] == -1) {
      inferred_count++;
      inferred_position = i;
      continue;
    }
    if (shape[i] < 0) {
      throw std::invalid_argument("Reshape dimensions must be -1 or positive");
    }
    resolved[i] = static_cast<size_t>(shape[i]);
    known *= resolved[i];
  }

  if (inferred_count > 1) {
    throw std::invalid_argument(
        "Reshape allows at most one inferred dimension");
  }

  if (inferred_count == 1) {
    if (known == 0 || total % known != 0) {
      throw std::invalid_argument("Cannot infer reshape dimension");
    }
    resolved[inferred_position] = total / known;
    known = total;
  }

  if (known != total) {
    throw std::invalid_argument("Reshape element count mismatch");
  }

  return resolved;
}

}  // namespace

std::shared_ptr<Tensor> Tensor::reshape(const std::vector<int64_t> &shape) {
  std::vector<size_t> resolved = ResolveShape(shape, size());

  auto result = std::make_shared<Tensor>(resolved);
  result->data_ = data_.copy_to(backend());
  result->grad_ = Storage(grad_.bytes(), backend());
  result->zero_grad();

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = [source = self_ptr, out = result.get()]() {
      source->to(Backend::CPU);
      out->to(Backend::CPU);
      std::ranges::transform(source->grad(), out->grad(),
                             source->grad().begin(), std::plus<>{});
    };
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::view(const std::vector<int64_t> &shape) {
  return reshape(shape);
}

}  // namespace micrograd
