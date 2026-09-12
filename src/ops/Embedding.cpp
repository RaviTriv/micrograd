#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::embedding_lookup(
    const std::shared_ptr<Tensor> &indices) {
  if (shape_.size() != 2) {
    throw std::invalid_argument("embedding_lookup expects a rank 2 weight");
  }

  size_t num_embeddings = shape_[0];
  size_t dim = shape_[1];
  size_t count = indices->size();

  std::vector<size_t> out_shape = indices->shape();
  out_shape.push_back(dim);

  auto result = std::make_shared<Tensor>(out_shape);
  const auto *weight = static_cast<const scalar_t *>(data_.host_pointer());
  const auto *index_values =
      static_cast<const scalar_t *>(indices->data_storage().host_pointer());
  std::span<scalar_t> out_values = result->data();

  for (size_t p = 0; p < count; p++) {
    auto index = static_cast<size_t>(std::lround(index_values[p]));
    if (index >= num_embeddings) {
      throw std::out_of_range("embedding_lookup index is out of range");
    }
    std::copy_n(&weight[index * dim], dim, &out_values[p * dim]);
  }
  result->to(backend());

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = [self_ptr, indices, out = result.get(), dim,
                            count]() {
      out->to(Backend::CPU);
      std::span<scalar_t> gradient(
          static_cast<scalar_t *>(self_ptr->grad_storage().host_pointer()),
          self_ptr->size());
      const auto *grad_index_values =
          static_cast<const scalar_t *>(indices->data_storage().host_pointer());
      std::span<const scalar_t> incoming = out->grad();
      for (size_t p = 0; p < count; p++) {
        auto index = static_cast<size_t>(std::lround(grad_index_values[p]));
        for (size_t d = 0; d < dim; d++) {
          gradient[(index * dim) + d] += incoming[(p * dim) + d];
        }
      }
    };
  }

  return result;
}

}  // namespace micrograd
