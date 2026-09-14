#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::rotary_embedding(scalar_t base) {
  if (shape_.size() < 2) {
    throw std::invalid_argument(
        "rotary_embedding expects an input of rank 2 or higher");
  }

  size_t head_dim = shape_.back();
  if (head_dim == 0 || head_dim % 2 != 0) {
    throw std::invalid_argument(
        "rotary_embedding expects an even, non-zero head dimension");
  }

  size_t seq_len = shape_[shape_.size() - 2];
  size_t half = head_dim / 2;
  size_t outer = size() / (seq_len * head_dim);

  auto cos_table = std::make_shared<std::vector<scalar_t>>(seq_len * half);
  auto sin_table = std::make_shared<std::vector<scalar_t>>(seq_len * half);
  for (size_t t = 0; t < seq_len; t++) {
    for (size_t j = 0; j < half; j++) {
      scalar_t exponent =
          static_cast<scalar_t>(2 * j) / static_cast<scalar_t>(head_dim);
      scalar_t freq = 1.0f / std::pow(base, exponent);
      scalar_t angle = static_cast<scalar_t>(t) * freq;
      (*cos_table)[(t * half) + j] = std::cos(angle);
      (*sin_table)[(t * half) + j] = std::sin(angle);
    }
  }

  auto result = std::make_shared<Tensor>(shape_);
  const auto *source = static_cast<const scalar_t *>(data_.host_pointer());
  std::span<scalar_t> out_values = result->data();

  for (size_t o = 0; o < outer; o++) {
    for (size_t t = 0; t < seq_len; t++) {
      const scalar_t *row = source + (((o * seq_len) + t) * head_dim);
      scalar_t *out_row = &out_values[((o * seq_len) + t) * head_dim];
      const scalar_t *cos_row = cos_table->data() + (t * half);
      const scalar_t *sin_row = sin_table->data() + (t * half);
      for (size_t j = 0; j < half; j++) {
        scalar_t x1 = row[j];
        scalar_t x2 = row[j + half];
        out_row[j] = (x1 * cos_row[j]) + (x2 * sin_row[j]);
        out_row[j + half] = (x2 * cos_row[j]) - (x1 * sin_row[j]);
      }
    }
  }
  result->to(backend());

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = [self_ptr, out = result.get(), cos_table, sin_table,
                            outer, seq_len, head_dim, half]() {
      out->to(Backend::CPU);
      std::span<scalar_t> gradient(
          static_cast<scalar_t *>(self_ptr->grad_storage().host_pointer()),
          self_ptr->size());
      std::span<const scalar_t> incoming = out->grad();

      for (size_t o = 0; o < outer; o++) {
        for (size_t t = 0; t < seq_len; t++) {
          const scalar_t *grad_row =
              incoming.data() + (((o * seq_len) + t) * head_dim);
          scalar_t *in_grad_row =
              gradient.data() + (((o * seq_len) + t) * head_dim);
          const scalar_t *cos_row = cos_table->data() + (t * half);
          const scalar_t *sin_row = sin_table->data() + (t * half);
          for (size_t j = 0; j < half; j++) {
            scalar_t dy1 = grad_row[j];
            scalar_t dy2 = grad_row[j + half];
            in_grad_row[j] += (dy1 * cos_row[j]) - (dy2 * sin_row[j]);
            in_grad_row[j + half] += (dy1 * sin_row[j]) + (dy2 * cos_row[j]);
          }
        }
      }
    };
  }

  return result;
}

}  // namespace micrograd
