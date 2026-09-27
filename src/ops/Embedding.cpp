#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Storage.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::embedding_lookup(
    const std::shared_ptr<Tensor> &indices) {
  if (shape_.size() != 2) {
    throw std::invalid_argument("embedding_lookup expects a rank 2 weight");
  }

  size_t num_embeddings = shape_[0];
  size_t dim = shape_[1];
  size_t count = indices->size();

  Storage host_indices = indices->data_storage().copy_to(Device::CPU);
  const auto *index_values =
      static_cast<const scalar_t *>(host_indices.host_pointer());
  for (size_t p = 0; p < count; p++) {
    auto index = static_cast<size_t>(std::lround(index_values[p]));
    if (index >= num_embeddings) {
      throw std::out_of_range("embedding_lookup index is out of range");
    }
  }

  std::vector<size_t> out_shape = indices->shape();
  out_shape.push_back(dim);

  auto result = std::make_shared<Tensor>(out_shape);
  DispatchOp(OpId::kEmbeddingLookup, backend(),
             {.lhs = this, .rhs = indices.get(), .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ =
        MakeBackward(OpId::kEmbeddingLookup, backend(),
                     {.lhs = self_ptr, .rhs = indices, .out = result.get()});
  }

  return result;
}

}  // namespace micrograd
