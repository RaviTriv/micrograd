#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "micrograd/Autograd.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Dispatch.h"

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

  auto cos_table = std::make_shared<Tensor>(std::vector<size_t>{seq_len, half});
  auto sin_table = std::make_shared<Tensor>(std::vector<size_t>{seq_len, half});
  auto cos_values = cos_table->data();
  auto sin_values = sin_table->data();
  for (size_t t = 0; t < seq_len; t++) {
    for (size_t j = 0; j < half; j++) {
      scalar_t exponent =
          static_cast<scalar_t>(2 * j) / static_cast<scalar_t>(head_dim);
      scalar_t freq = 1.0f / std::pow(base, exponent);
      scalar_t angle = static_cast<scalar_t>(t) * freq;
      cos_values[(t * half) + j] = std::cos(angle);
      sin_values[(t * half) + j] = std::sin(angle);
    }
  }
  cos_table->to(backend());
  sin_table->to(backend());

  auto result = std::make_shared<Tensor>(shape_);
  DispatchOp(OpId::kRotaryEmbedding, backend(),
             {.lhs = this,
              .rhs = cos_table.get(),
              .extra = sin_table.get(),
              .out = result.get()});

  result->requires_grad_ = GradEnabled() && requires_grad_;
  if (result->requires_grad_) {
    auto self_ptr = shared_from_this();
    result->children_ = {self_ptr};
    result->backward_fn_ = MakeBackward(OpId::kRotaryEmbedding, backend(),
                                        {.lhs = self_ptr,
                                         .rhs = cos_table,
                                         .extra = sin_table,
                                         .out = result.get()});
  }

  return result;
}

}  // namespace micrograd
