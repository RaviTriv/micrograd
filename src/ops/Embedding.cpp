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
namespace {

void RequireRowGather(const std::vector<size_t> &shape, size_t index_count) {
  if (shape.size() != 2) {
    throw std::invalid_argument("gather_per_row expects a rank 2 input");
  }
  if (index_count != shape[0]) {
    throw std::invalid_argument(
        "gather_per_row index count does not match the row count");
  }
}

}  // namespace

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
  RequireRowGather(shape_, indices->size());

  size_t rows = shape_[0];
  size_t cols = shape_[1];
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

std::shared_ptr<Tensor> Tensor::gather_per_row(
    const std::vector<size_t> &indices) {
  RequireRowGather(shape_, indices.size());

  size_t rows = shape_[0];
  size_t cols = shape_[1];
  std::vector<scalar_t> index_values(rows);
  for (size_t i = 0; i < rows; i++) {
    if (indices[i] >= cols) {
      throw std::out_of_range("gather_per_row index is out of range");
    }
    index_values[i] = static_cast<scalar_t>(indices[i]);
  }
  auto index_tensor =
      Tensor::from_values(std::vector<size_t>{rows}, std::move(index_values));
  index_tensor->to(backend());

  return apply(OpId::kGatherPerRow, {rows},
               {.lhs = shared_from_this(),
                .rhs = index_tensor,
                .grad_through_lhs_only = true});
}

std::shared_ptr<Tensor> Tensor::cross_entropy_rows(
    const std::vector<size_t> &targets) {
  if (shape_.size() < 2) {
    throw std::invalid_argument("cross_entropy expects logits of rank 2 or more");
  }
  size_t cols = shape_.back();
  size_t rows = cols == 0 ? 0 : size() / cols;
  if (targets.size() != rows) {
    throw std::invalid_argument(
        "gather_per_row index count does not match the row count");
  }
  std::vector<scalar_t> target_values(rows);
  for (size_t i = 0; i < rows; i++) {
    if (targets[i] >= cols) {
      throw std::out_of_range("cross_entropy target is out of range");
    }
    target_values[i] = static_cast<scalar_t>(targets[i]);
  }
  auto target_tensor =
      Tensor::from_values(std::vector<size_t>{rows}, std::move(target_values));
  target_tensor->to(backend());

  return apply(OpId::kCrossEntropy, {rows},
               {.lhs = shared_from_this(),
                .rhs = target_tensor,
                .grad_through_lhs_only = true});
}

}  // namespace micrograd
