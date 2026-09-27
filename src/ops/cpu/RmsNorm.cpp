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

void RmsNorm(const OpArgs &args) {
  size_t n = args.rhs->size();
  size_t outer = args.lhs->size() / n;
  scalar_t eps = args.scalar;

  auto input = args.lhs->data();
  auto gain = args.rhs->data();
  auto out = args.out->data();

  for (size_t o = 0; o < outer; o++) {
    const scalar_t *row = input.data() + (o * n);

    scalar_t mean_square = 0.0f;
    for (size_t i = 0; i < n; i++) {
      mean_square += row[i] * row[i];
    }
    mean_square /= static_cast<scalar_t>(n);
    scalar_t rstd = 1.0f / std::sqrt(mean_square + eps);

    for (size_t i = 0; i < n; i++) {
      out[(o * n) + i] = row[i] * rstd * gain[i];
    }
  }
}

std::function<void()> RmsNormBackward(const GradArgs &args) {
  return
      [lhs = args.lhs, gain = args.rhs, out = args.out, eps = args.scalar]() {
        size_t n = gain->size();
        size_t outer = lhs->size() / n;

        auto input = lhs->data();
        auto gain_values = gain->data();
        auto incoming = out->grad();
        auto input_grad = lhs->grad();
        auto gain_grad = gain->grad();

        std::vector<scalar_t> xhat(n);
        std::vector<scalar_t> dxhat(n);
        for (size_t o = 0; o < outer; o++) {
          const scalar_t *row = input.data() + (o * n);
          const scalar_t *row_incoming = incoming.data() + (o * n);

          scalar_t mean_square = 0.0f;
          for (size_t i = 0; i < n; i++) {
            mean_square += row[i] * row[i];
          }
          mean_square /= static_cast<scalar_t>(n);
          scalar_t rstd = 1.0f / std::sqrt(mean_square + eps);

          scalar_t mean_dxhat_xhat = 0.0f;
          for (size_t i = 0; i < n; i++) {
            scalar_t xhat_i = row[i] * rstd;
            scalar_t dxhat_i = row_incoming[i] * gain_values[i];
            xhat[i] = xhat_i;
            dxhat[i] = dxhat_i;
            mean_dxhat_xhat += dxhat_i * xhat_i;

            gain_grad[i] += row_incoming[i] * xhat_i;
          }
          mean_dxhat_xhat /= static_cast<scalar_t>(n);

          for (size_t i = 0; i < n; i++) {
            input_grad[(o * n) + i] +=
                rstd * (dxhat[i] - (xhat[i] * mean_dxhat_xhat));
          }
        }
      };
}

}  // namespace

void RegisterRmsNormOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kRmsNorm, Device::CPU, RmsNorm);
  registry.RegisterBackward(OpId::kRmsNorm, Device::CPU, RmsNormBackward);
}

}  // namespace micrograd::ops::cpu
