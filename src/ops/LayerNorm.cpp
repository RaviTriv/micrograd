#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"

namespace micrograd {

std::shared_ptr<Tensor> Tensor::layer_norm(
    const std::vector<size_t> &normalized_shape,
    const std::shared_ptr<Tensor> &gain, const std::shared_ptr<Tensor> &bias,
    scalar_t eps) {
  if (normalized_shape.size() > shape_.size()) {
    throw std::invalid_argument(
        "layer_norm normalized_shape has more dimensions than the input");
  }

  size_t leading = shape_.size() - normalized_shape.size();
  for (size_t i = 0; i < normalized_shape.size(); i++) {
    if (shape_[leading + i] != normalized_shape[i]) {
      throw std::invalid_argument(
          "layer_norm input shape does not end with normalized_shape");
    }
  }

  size_t n = 1;
  for (auto dim : normalized_shape) {
    n *= dim;
  }
  if (gain->size() != n || bias->size() != n) {
    throw std::invalid_argument(
        "layer_norm gain and bias must match normalized_shape");
  }
  size_t outer = size() / n;

  auto result = std::make_shared<Tensor>(shape_);
  auto xhat = std::make_shared<std::vector<scalar_t>>(size());
  auto rstd = std::make_shared<std::vector<scalar_t>>(outer);

  const auto *source = static_cast<const scalar_t *>(data_.host_pointer());
  const auto *gain_values =
      static_cast<const scalar_t *>(gain->data_storage().host_pointer());
  const auto *bias_values =
      static_cast<const scalar_t *>(bias->data_storage().host_pointer());
  std::span<scalar_t> out_values = result->data();

  for (size_t o = 0; o < outer; o++) {
    const scalar_t *row = source + (o * n);

    scalar_t mean = 0.0f;
    for (size_t i = 0; i < n; i++) {
      mean += row[i];
    }
    mean /= static_cast<scalar_t>(n);

    scalar_t variance = 0.0f;
    for (size_t i = 0; i < n; i++) {
      scalar_t centered = row[i] - mean;
      variance += centered * centered;
    }
    variance /= static_cast<scalar_t>(n);

    scalar_t row_rstd = 1.0f / std::sqrt(variance + eps);
    (*rstd)[o] = row_rstd;

    for (size_t i = 0; i < n; i++) {
      scalar_t normalized = (row[i] - mean) * row_rstd;
      (*xhat)[(o * n) + i] = normalized;
      out_values[(o * n) + i] = (normalized * gain_values[i]) + bias_values[i];
    }
  }
  result->to(backend());

  result->requires_grad_ =
      GradEnabled() &&
      (requires_grad_ || gain->requires_grad() || bias->requires_grad());
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr, gain, bias};
    result->backward_fn_ = [self_ptr, gain, bias, out = result.get(), xhat,
                            rstd, n, outer]() {
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
      std::span<scalar_t> bias_grad(
          static_cast<scalar_t *>(bias->grad_storage().host_pointer()),
          bias->size());

      std::vector<scalar_t> dxhat(n);
      for (size_t o = 0; o < outer; o++) {
        const scalar_t *row_xhat = xhat->data() + (o * n);
        const scalar_t *row_incoming = incoming.data() + (o * n);
        scalar_t row_rstd = (*rstd)[o];

        scalar_t mean_dxhat = 0.0f;
        scalar_t mean_dxhat_xhat = 0.0f;
        for (size_t i = 0; i < n; i++) {
          scalar_t dxhat_i = row_incoming[i] * backward_gain_values[i];
          dxhat[i] = dxhat_i;
          mean_dxhat += dxhat_i;
          mean_dxhat_xhat += dxhat_i * row_xhat[i];

          gain_grad[i] += row_incoming[i] * row_xhat[i];
          bias_grad[i] += row_incoming[i];
        }
        mean_dxhat /= static_cast<scalar_t>(n);
        mean_dxhat_xhat /= static_cast<scalar_t>(n);

        for (size_t i = 0; i < n; i++) {
          input_grad[(o * n) + i] +=
              row_rstd *
              (dxhat[i] - mean_dxhat - (row_xhat[i] * mean_dxhat_xhat));
        }
      }
    };
  }

  return result;
}

}  // namespace micrograd
