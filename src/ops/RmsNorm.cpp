#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::rms_norm(
    const std::vector<size_t> &normalized_shape,
    const std::shared_ptr<Tensor> &gain, scalar_t eps) {
  if (normalized_shape.size() > shape_.size()) {
    throw std::invalid_argument(
        "rms_norm normalized_shape has more dimensions than the input");
  }

  size_t leading = shape_.size() - normalized_shape.size();
  for (size_t i = 0; i < normalized_shape.size(); i++) {
    if (shape_[leading + i] != normalized_shape[i]) {
      throw std::invalid_argument(
          "rms_norm input shape does not end with normalized_shape");
    }
  }

  size_t n = 1;
  for (auto dim : normalized_shape) {
    n *= dim;
  }
  if (gain->size() != n) {
    throw std::invalid_argument("rms_norm gain must match normalized_shape");
  }
  size_t outer = size() / n;

  auto result = std::make_shared<Tensor>(shape_);
  auto xhat = std::make_shared<std::vector<scalar_t>>(size());
  auto rstd = std::make_shared<std::vector<scalar_t>>(outer);

  const auto *source = static_cast<const scalar_t *>(data_.host_pointer());
  const auto *gain_values =
      static_cast<const scalar_t *>(gain->data_storage().host_pointer());
  std::span<scalar_t> out_values = result->data();

  for (size_t o = 0; o < outer; o++) {
    const scalar_t *row = source + (o * n);

    scalar_t mean_square = 0.0f;
    for (size_t i = 0; i < n; i++) {
      mean_square += row[i] * row[i];
    }
    mean_square /= static_cast<scalar_t>(n);

    scalar_t row_rstd = 1.0f / std::sqrt(mean_square + eps);
    (*rstd)[o] = row_rstd;

    for (size_t i = 0; i < n; i++) {
      scalar_t normalized = row[i] * row_rstd;
      (*xhat)[(o * n) + i] = normalized;
      out_values[(o * n) + i] = normalized * gain_values[i];
    }
  }
  result->to(backend());

  result->requires_grad_ =
      GradEnabled() && (requires_grad_ || gain->requires_grad());
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr, gain};
    result->backward_fn_ = [self_ptr, gain, out = result.get(), xhat, rstd, n,
                            outer]() {
      out->to(Backend::CPU);
      std::span<const scalar_t> incoming = out->grad();
      const auto *backward_gain_values =
          static_cast<const scalar_t *>(gain->data_storage().host_pointer());

      std::span<scalar_t> input_grad(
          static_cast<scalar_t *>(self_ptr->grad_storage().host_pointer()),
          self_ptr->size());
      std::span<scalar_t> gain_grad(
          static_cast<scalar_t *>(gain->grad_storage().host_pointer()),
          gain->size());

      std::vector<scalar_t> dxhat(n);
      for (size_t o = 0; o < outer; o++) {
        const scalar_t *row_xhat = xhat->data() + (o * n);
        const scalar_t *row_incoming = incoming.data() + (o * n);
        scalar_t row_rstd = (*rstd)[o];

        scalar_t mean_dxhat_xhat = 0.0f;
        for (size_t i = 0; i < n; i++) {
          scalar_t dxhat_i = row_incoming[i] * backward_gain_values[i];
          dxhat[i] = dxhat_i;
          mean_dxhat_xhat += dxhat_i * row_xhat[i];

          gain_grad[i] += row_incoming[i] * row_xhat[i];
        }
        mean_dxhat_xhat /= static_cast<scalar_t>(n);

        for (size_t i = 0; i < n; i++) {
          input_grad[(o * n) + i] +=
              row_rstd * (dxhat[i] - (row_xhat[i] * mean_dxhat_xhat));
        }
      }
    };
  }

  return result;
}

}  // namespace micrograd
