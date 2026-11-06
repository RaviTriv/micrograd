#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

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

  return apply(OpId::kEmbeddingLookup, std::move(out_shape),
               {.lhs = shared_from_this(),
                .rhs = indices,
                .grad_through_lhs_only = true});
}

std::shared_ptr<Tensor> Tensor::gather_per_row(
    const std::shared_ptr<Tensor> &indices) {
  if (shape_.size() != 2) {
    throw std::invalid_argument("gather_per_row expects a rank 2 input");
  }

  size_t rows = shape_[0];
  size_t cols = shape_[1];
  if (indices->size() != rows) {
    throw std::invalid_argument(
        "gather_per_row index count does not match the row count");
  }

  Storage host_indices = indices->data_storage().copy_to(Device::CPU);
  const auto *index_values =
      static_cast<const scalar_t *>(host_indices.host_pointer());
  for (size_t i = 0; i < rows; i++) {
    auto index = static_cast<size_t>(std::lround(index_values[i]));
    if (index >= cols) {
      throw std::out_of_range("gather_per_row index is out of range");
    }
  }

  return apply(OpId::kGatherPerRow, {rows},
               {.lhs = shared_from_this(),
                .rhs = indices,
                .grad_through_lhs_only = true});
}

}  // namespace micrograd
