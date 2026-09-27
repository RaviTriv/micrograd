#include <cmath>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"
#include "micrograd/ops/cpu/Ops.h"

namespace micrograd::ops::cpu {
namespace {

void LayerNorm(const OpArgs &args) {
  size_t n = args.rhs->size();
  size_t outer = args.lhs->size() / n;
  scalar_t eps = args.scalar;

  auto input = args.lhs->data();
  auto gain = args.rhs->data();
  auto bias = args.extra->data();
  auto out = args.out->data();

  for (size_t o = 0; o < outer; o++) {
    const scalar_t *row = input.data() + (o * n);

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
    scalar_t rstd = 1.0f / std::sqrt(variance + eps);

    for (size_t i = 0; i < n; i++) {
      scalar_t normalized = (row[i] - mean) * rstd;
      out[(o * n) + i] = (normalized * gain[i]) + bias[i];
    }
  }
}

std::function<void()> LayerNormBackward(const GradArgs &args) {
  return [lhs = args.lhs, gain = args.rhs, bias = args.extra, out = args.out,
          eps = args.scalar]() {
    size_t n = gain->size();
    size_t outer = lhs->size() / n;

    auto input = lhs->data();
    auto gain_values = gain->data();
    auto incoming = out->grad();
    auto input_grad = lhs->grad();
    auto gain_grad = gain->grad();
    auto bias_grad = bias->grad();

    std::vector<scalar_t> xhat(n);
    std::vector<scalar_t> dxhat(n);
    for (size_t o = 0; o < outer; o++) {
      const scalar_t *row = input.data() + (o * n);
      const scalar_t *row_incoming = incoming.data() + (o * n);

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
      scalar_t rstd = 1.0f / std::sqrt(variance + eps);

      scalar_t mean_dxhat = 0.0f;
      scalar_t mean_dxhat_xhat = 0.0f;
      for (size_t i = 0; i < n; i++) {
        scalar_t xhat_i = (row[i] - mean) * rstd;
        scalar_t dxhat_i = row_incoming[i] * gain_values[i];
        xhat[i] = xhat_i;
        dxhat[i] = dxhat_i;
        mean_dxhat += dxhat_i;
        mean_dxhat_xhat += dxhat_i * xhat_i;

        gain_grad[i] += row_incoming[i] * xhat_i;
        bias_grad[i] += row_incoming[i];
      }
      mean_dxhat /= static_cast<scalar_t>(n);
      mean_dxhat_xhat /= static_cast<scalar_t>(n);

      for (size_t i = 0; i < n; i++) {
        input_grad[(o * n) + i] +=
            rstd * (dxhat[i] - mean_dxhat - (xhat[i] * mean_dxhat_xhat));
      }
    }
  };
}

}  // namespace

void RegisterLayerNormOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kLayerNorm, Device::CPU, LayerNorm);
  registry.RegisterBackward(OpId::kLayerNorm, Device::CPU, LayerNormBackward);
}

}  // namespace micrograd::ops::cpu
