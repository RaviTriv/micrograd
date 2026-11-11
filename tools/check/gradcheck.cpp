#include "micrograd/GradCheck.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpt/Attention.h"
#include "gpt/GRPO.h"
#include "micrograd/Device.h"
#include "micrograd/NN.h"
#include "micrograd/Random.h"
#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"

#ifdef MICROGRAD_CUDA_ENABLED
#include "micrograd/backends/cuda/CudaContext.h"
#endif

namespace {

using micrograd::Device;
using micrograd::DType;
using micrograd::GradCheck;
using micrograd::GradCheckFunction;
using micrograd::GradCheckOptions;
using micrograd::scalar_t;
using micrograd::Tensor;
using micrograd::internal::MakeGradCheckLeaves;
using micrograd::internal::ProjectionWeights;

using TensorPtr = std::shared_ptr<Tensor>;
using TensorList = std::vector<TensorPtr>;

struct Case {
  std::string name;
  GradCheckFunction function;
  TensorList inputs;
  GradCheckOptions options;
  bool device_only = false;
  bool tf32 = false;
  bool bf16 = false;
  bool slot_scaled_atol = false;
  size_t samples = 0;
};

TensorPtr Ramp(std::vector<size_t> shape, scalar_t start, scalar_t step) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= dim;
  }

  std::vector<scalar_t> values(count);
  for (size_t i = 0; i < count; i++) {
    values[i] = start + (step * static_cast<scalar_t>(i));
  }
  return Tensor::from_values(std::move(shape), std::move(values));
}

TensorPtr Low(const TensorPtr &t) {
  return t->backend() == Device::CUDA ? t->cast(DType::kBFloat16) : t;
}

TensorPtr High(const TensorPtr &t) { return t->cast(DType::kFloat32); }

TensorPtr Wave(std::vector<size_t> shape, scalar_t phase) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= dim;
  }

  std::vector<scalar_t> values(count);
  for (size_t i = 0; i < count; i++) {
    values[i] = std::sin((0.37f * static_cast<scalar_t>(i)) + phase);
  }
  return Tensor::from_values(std::move(shape), std::move(values));
}

TensorPtr Noise(std::vector<size_t> shape, uint32_t seed) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= dim;
  }

  std::vector<scalar_t> values(count);
  uint32_t state = seed;
  for (size_t i = 0; i < count; i++) {
    state = (state * 1664525u) + 1013904223u;
    values[i] = (static_cast<scalar_t>(state >> 8) / 8388608.0f) - 1.0f;
  }
  return Tensor::from_values(std::move(shape), std::move(values));
}

std::vector<Case> AllCases() {
  std::vector<Case> cases;
  auto add = [&cases](std::string name, GradCheckFunction function,
                      TensorList inputs, GradCheckOptions options = {}) {
    cases.push_back({.name = std::move(name),
                     .function = std::move(function),
                     .inputs = std::move(inputs),
                     .options = options});
  };
  auto add_device_only = [&cases](std::string name, GradCheckFunction function,
                                  TensorList inputs,
                                  GradCheckOptions options = {},
                                  bool tf32 = false, bool bf16 = false) {
    cases.push_back({.name = std::move(name),
                     .function = std::move(function),
                     .inputs = std::move(inputs),
                     .options = options,
                     .device_only = true,
                     .tf32 = tf32,
                     .bf16 = bf16});
  };
  auto add_low_precision = [&cases](std::string name,
                                    GradCheckFunction function,
                                    TensorList inputs, GradCheckOptions options,
                                    bool tf32 = false, bool bf16 = false) {
    cases.push_back({.name = std::move(name),
                     .function = std::move(function),
                     .inputs = std::move(inputs),
                     .options = options,
                     .device_only = true,
                     .tf32 = tf32,
                     .bf16 = bf16,
                     .slot_scaled_atol = true});
  };
  auto add_sampled = [&cases](std::string name, GradCheckFunction function,
                              TensorList inputs, GradCheckOptions options,
                              size_t samples) {
    cases.push_back({.name = std::move(name),
                     .function = std::move(function),
                     .inputs = std::move(inputs),
                     .options = options,
                     .samples = samples});
  };

  const auto signed_matrix = [] { return Ramp({2, 3}, -1.25f, 0.5f); };
  const auto positive_matrix = [] { return Ramp({2, 3}, 0.5f, 0.3f); };
  const auto signed_cube = [] { return Ramp({2, 3, 3}, -1.1f, 0.15f); };
  const auto row_vector = [] { return Ramp({3}, 0.4f, 0.7f); };
  const auto positive_row = [] { return Ramp({3}, 0.6f, 0.5f); };
  const auto column = [] { return Ramp({2, 1}, -0.8f, 1.3f); };

  add("add", [](const TensorList &in) { return in[0]->add(in[1]); },
      {signed_matrix(), positive_matrix()});
  add("sub", [](const TensorList &in) { return in[0]->sub(in[1]); },
      {signed_matrix(), positive_matrix()});
  add("mul", [](const TensorList &in) { return in[0]->mul(in[1]); },
      {signed_matrix(), positive_matrix()});
  add("div", [](const TensorList &in) { return in[0]->div(in[1]); },
      {signed_matrix(), positive_matrix()});

  add("add_broadcast", [](const TensorList &in) { return in[0]->add(in[1]); },
      {signed_matrix(), row_vector()});
  add("add_bias", [](const TensorList &in) { return in[0]->add_bias(in[1]); },
      {signed_matrix(), row_vector()});
  add("add_bias_many_rows_and_cols",
      [](const TensorList &in) { return in[0]->add_bias(in[1]); },
      {Wave({2, 150, 40}, 0.2f), Wave({40}, 1.3f)}, {.eps = 0.5f});
  add("sub_broadcast", [](const TensorList &in) { return in[0]->sub(in[1]); },
      {signed_matrix(), row_vector()});
  add("mul_broadcast", [](const TensorList &in) { return in[0]->mul(in[1]); },
      {signed_matrix(), row_vector()});
  add("div_broadcast", [](const TensorList &in) { return in[0]->div(in[1]); },
      {signed_matrix(), positive_row()});
  add("add_broadcast_lhs_smaller",
      [](const TensorList &in) { return in[0]->add(in[1]); },
      {row_vector(), signed_matrix()});
  add("sub_broadcast_lhs_smaller",
      [](const TensorList &in) { return in[0]->sub(in[1]); },
      {row_vector(), signed_matrix()});
  add("mul_broadcast_lhs_smaller",
      [](const TensorList &in) { return in[0]->mul(in[1]); },
      {row_vector(), signed_matrix()});
  add("div_broadcast_lhs_smaller",
      [](const TensorList &in) { return in[0]->div(in[1]); },
      {row_vector(), positive_matrix()});
  add("add_broadcast_two_sided",
      [](const TensorList &in) { return in[0]->add(in[1]); },
      {column(), Ramp({1, 3}, 0.4f, 0.7f)});
  add("sub_broadcast_two_sided",
      [](const TensorList &in) { return in[0]->sub(in[1]); },
      {column(), Ramp({1, 3}, 0.4f, 0.7f)});
  add("mul_broadcast_two_sided",
      [](const TensorList &in) { return in[0]->mul(in[1]); },
      {column(), Ramp({1, 3}, 0.4f, 0.7f)});
  add("div_broadcast_two_sided",
      [](const TensorList &in) { return in[0]->div(in[1]); },
      {column(), Ramp({1, 3}, 0.4f, 0.7f)});
  add("add_broadcast_3d",
      [](const TensorList &in) { return in[0]->add(in[1]); },
      {signed_cube(), Ramp({3, 1}, -0.6f, 0.5f)});
  add("sub_broadcast_3d",
      [](const TensorList &in) { return in[0]->sub(in[1]); },
      {signed_cube(), Ramp({3, 1}, -0.6f, 0.5f)});
  add("mul_broadcast_3d",
      [](const TensorList &in) { return in[0]->mul(in[1]); },
      {signed_cube(), Ramp({3, 1}, -0.6f, 0.5f)});
  add("div_broadcast_3d",
      [](const TensorList &in) { return in[0]->div(in[1]); },
      {signed_cube(), Ramp({1, 3}, 0.5f, 0.4f)});

  add("add_scalar", [](const TensorList &in) { return in[0]->add(1.5f); },
      {signed_matrix()});
  add("sub_scalar", [](const TensorList &in) { return in[0]->sub(0.75f); },
      {signed_matrix()});
  add("mul_scalar", [](const TensorList &in) { return in[0]->mul(-2.5f); },
      {signed_matrix()});
  add("div_scalar", [](const TensorList &in) { return in[0]->div(4.0f); },
      {signed_matrix()});

  add("pow_two", [](const TensorList &in) { return in[0]->pow(2.0f); },
      {signed_matrix()});
  add("pow_three", [](const TensorList &in) { return in[0]->pow(3.0f); },
      {signed_matrix()});
  add("pow_half", [](const TensorList &in) { return in[0]->pow(0.5f); },
      {positive_matrix()});

  add("reshape", [](const TensorList &in) { return in[0]->reshape({3, 2}); },
      {signed_matrix()});
  add("view", [](const TensorList &in) { return in[0]->view({-1}); },
      {signed_matrix()});
  add("transpose", [](const TensorList &in) { return in[0]->transpose(0, 1); },
      {signed_matrix()});
  add("permute", [](const TensorList &in) { return in[0]->permute({1, 0}); },
      {signed_matrix()});
  add("contiguous",
      [](const TensorList &in) { return in[0]->transpose(0, 1)->contiguous(); },
      {signed_matrix()});
  add("permute_3d_201",
      [](const TensorList &in) { return in[0]->permute({2, 0, 1}); },
      {signed_cube()});
  add("permute_3d_120",
      [](const TensorList &in) { return in[0]->permute({1, 2, 0}); },
      {signed_cube()});
  add("transpose_3d_02",
      [](const TensorList &in) { return in[0]->transpose(0, 2); },
      {signed_cube()});
  add("permute_reshape",
      [](const TensorList &in) {
        return in[0]->permute({2, 0, 1})->reshape({9, 2});
      },
      {signed_cube()});

  add("sum", [](const TensorList &in) { return in[0]->sum(); },
      {signed_matrix()});
  add("sum_dim", [](const TensorList &in) { return in[0]->sum(1); },
      {signed_matrix()});
  add("sum_dim_keepdim",
      [](const TensorList &in) { return in[0]->sum(0, true); },
      {signed_matrix()});
  add("mean_dim", [](const TensorList &in) { return in[0]->mean(1); },
      {signed_matrix()});
  add("mean_dim_keepdim",
      [](const TensorList &in) { return in[0]->mean(0, true); },
      {signed_matrix()});
  add("max_dim", [](const TensorList &in) { return in[0]->max(1); },
      {signed_matrix()});
  add("max_dim_keepdim",
      [](const TensorList &in) { return in[0]->max(0, true); },
      {signed_matrix()});
  add("sum_middle_dim", [](const TensorList &in) { return in[0]->sum(1); },
      {signed_cube()});
  add("sum_middle_dim_keepdim",
      [](const TensorList &in) { return in[0]->sum(1, true); },
      {signed_cube()});
  add("mean_middle_dim", [](const TensorList &in) { return in[0]->mean(1); },
      {signed_cube()});
  add("max_middle_dim", [](const TensorList &in) { return in[0]->max(1); },
      {signed_cube()});
  add("sum_negative_dim", [](const TensorList &in) { return in[0]->sum(-2); },
      {signed_cube()});
  add("mean_negative_dim", [](const TensorList &in) { return in[0]->mean(-1); },
      {signed_matrix()});
  add("max_negative_dim", [](const TensorList &in) { return in[0]->max(-3); },
      {signed_cube()});

  add("matmul", [](const TensorList &in) { return in[0]->matmul(in[1]); },
      {Ramp({2, 3}, -0.9f, 0.4f), Ramp({3, 4}, -1.1f, 0.2f)});
  add("matmul_fold_rows",
      [](const TensorList &in) { return in[0]->matmul(in[1]); },
      {Ramp({2, 3, 4}, -0.9f, 0.1f), Ramp({4, 5}, -1.1f, 0.15f)});
  add("matmul_fold_rows_multi_block",
      [](const TensorList &in) { return in[0]->matmul(in[1]); },
      {Ramp({2, 150, 4}, -0.9f, 0.0015f), Ramp({4, 5}, -1.1f, 0.15f)},
      {.eps = 0.5f});
  add("matmul_past_one_tile",
      [](const TensorList &in) { return in[0]->matmul(in[1]); },
      {Wave({33, 34}, 0.1f), Wave({34, 35}, 0.7f)},
      {.eps = 0.5f, .atol = 1e-4f});
  add("matmul_fold_rows_past_one_tile",
      [](const TensorList &in) { return in[0]->matmul(in[1]); },
      {Wave({2, 3, 33}, 0.2f), Wave({33, 34}, 0.9f)}, {.eps = 0.5f});
  add("matmul_batched_past_one_tile",
      [](const TensorList &in) { return in[0]->matmul(in[1]); },
      {Wave({2, 5, 33}, 0.3f), Wave({2, 33, 6}, 1.1f)}, {.eps = 0.5f});

  add("matmul_nt",
      [](const TensorList &in) { return in[0]->matmul_nt(in[1]); },
      {Wave({2, 3, 5}, 0.3f), Wave({4, 5}, 1.1f)});
  add("matmul_nt_chain",
      [](const TensorList &in) {
        return in[0]->mul(2.0f)->matmul_nt(in[1])->mul(0.5f);
      },
      {Wave({37, 20}, 0.3f), Wave({33, 20}, 1.1f)},
      {.eps = 0.5f, .atol = 2e-4f});
  const GradCheckOptions bf16_tolerance = {.rtol = 3e-2f, .atol = 3e-2f};
  const GradCheckOptions bf16_attention = {.rtol = 3e-2f, .atol = 1e-1f};
  const GradCheckOptions bf16_attention_activations = {.rtol = 3e-2f,
                                                       .atol = 1.5e-1f};
  const GradCheckOptions bf16_softmax = {.rtol = 3e-2f, .atol = 3e-3f};
  add_device_only("matmul_bf16",
                  [](const TensorList &in) { return in[0]->matmul(in[1]); },
                  {Wave({2, 37, 20}, 0.3f), Wave({20, 33}, 1.1f)},
                  bf16_tolerance, false, true);
  add_device_only("matmul_batched_bf16",
                  [](const TensorList &in) {
                    return in[0]->mul(1.0f)->matmul(in[1]->mul(1.0f));
                  },
                  {Wave({3, 17, 20}, 0.3f), Wave({3, 20, 9}, 1.1f)},
                  bf16_tolerance, false, true);
  add_device_only("matmul_nt_bf16",
                  [](const TensorList &in) {
                    return in[0]->mul(1.0f)->matmul_nt(in[1]);
                  },
                  {Wave({2, 37, 20}, 0.3f), Wave({33, 20}, 1.1f)},
                  bf16_tolerance, false, true);
  add_low_precision("causal_self_attention_bf16",
                    [](const TensorList &in) {
                      return in[0]->causal_self_attention(in[1], in[2], 2,
                                                          0.125f);
                    },
                    {Wave({2, 70, 128}, 0.4f), Wave({2, 70, 128}, 1.3f),
                     Wave({2, 70, 128}, 2.2f)},
                    bf16_attention, false, true);
  const GradCheckOptions bf16_activations = {.rtol = 3e-2f, .atol = 2e-2f};
  const GradCheckOptions bf16_row_sums = {.rtol = 3e-2f, .atol = 6e-2f};
  add_device_only("cast_bf16_roundtrip",
                  [](const TensorList &in) {
                    return High(Low(in[0]))->mul(in[1]);
                  },
                  {Wave({3, 37}, 0.2f), Wave({3, 37}, 1.1f)},
                  bf16_activations);
  add_device_only("add_bf16",
                  [](const TensorList &in) {
                    return High(Low(in[0])->add(Low(in[1])));
                  },
                  {Wave({4, 37}, 0.2f), Wave({4, 37}, 1.4f)},
                  bf16_activations);
  add_device_only("add_bf16_shared_input",
                  [](const TensorList &in) {
                    const TensorPtr x = Low(in[0]);
                    return High(x->add(x->gelu()));
                  },
                  {Wave({3, 40}, 0.6f)}, bf16_activations);
  add_device_only("add_bias_bf16",
                  [](const TensorList &in) {
                    return High(Low(in[0])->add_bias(in[1]));
                  },
                  {Wave({2, 150, 40}, 0.2f), Wave({40}, 1.3f)},
                  bf16_row_sums);
  add_device_only("add_bias_bf16_odd_cols",
                  [](const TensorList &in) {
                    return High(Low(in[0])->add_bias(in[1]));
                  },
                  {Wave({5, 37}, 0.2f), Wave({37}, 1.3f)}, bf16_activations);
  add_device_only("linear_bf16",
                  [](const TensorList &in) {
                    if (in[0]->backend() != Device::CUDA) {
                      return in[0]->matmul(in[1])->add_bias(in[2]);
                    }
                    return High(Low(in[0])->linear(in[1], in[2]));
                  },
                  {Wave({2, 37, 24}, 0.2f), Wave({24, 40}, 0.9f),
                   Wave({40}, 1.3f)},
                  bf16_row_sums);
  add_device_only("linear_bf16_odd_shape",
                  [](const TensorList &in) {
                    if (in[0]->backend() != Device::CUDA) {
                      return in[0]->matmul(in[1])->add_bias(in[2]);
                    }
                    return High(Low(in[0])->linear(Low(in[1]), in[2]));
                  },
                  {Wave({5, 19}, 0.2f), Wave({19, 7}, 0.9f), Wave({7}, 1.3f)},
                  bf16_activations);
  add_device_only("gelu_bf16",
                  [](const TensorList &in) {
                    return High(Low(in[0]->mul(2.0f))->gelu());
                  },
                  {Wave({3, 45}, 0.4f)}, bf16_activations);
  add_device_only("layer_norm_bf16",
                  [](const TensorList &in) {
                    return High(
                        Low(in[0])->layer_norm({64}, in[1], in[2], 1e-5f));
                  },
                  {Wave({3, 5, 64}, 0.4f), Wave({64}, 1.0f), Wave({64}, 2.0f)},
                  bf16_activations);
  add_device_only("layer_norm_bf16_gpt2_width",
                  [](const TensorList &in) {
                    return High(
                        Low(in[0])->layer_norm({768}, in[1], in[2], 1e-5f));
                  },
                  {Wave({3, 5, 768}, 0.4f), Wave({768}, 1.0f),
                   Wave({768}, 2.0f)},
                  bf16_row_sums);
  add_device_only("layer_norm_bf16_odd_width",
                  [](const TensorList &in) {
                    return High(
                        Low(in[0])->layer_norm({37}, in[1], in[2], 1e-5f));
                  },
                  {Wave({4, 37}, 0.4f), Wave({37}, 1.0f), Wave({37}, 2.0f)},
                  bf16_activations);
  add_device_only("matmul_bf16_activations",
                  [](const TensorList &in) {
                    return High(Low(in[0])->matmul(Low(in[1])));
                  },
                  {Wave({2, 37, 20}, 0.3f), Wave({20, 33}, 1.1f)},
                  bf16_activations);
  add_device_only("matmul_batched_bf16_activations",
                  [](const TensorList &in) {
                    return High(Low(in[0])->matmul(Low(in[1])));
                  },
                  {Wave({3, 17, 20}, 0.3f), Wave({3, 20, 9}, 1.1f)},
                  bf16_activations);
  add_device_only("matmul_nt_bf16_activations",
                  [](const TensorList &in) {
                    return High(Low(in[0])->matmul_nt(Low(in[1])));
                  },
                  {Wave({2, 37, 20}, 0.3f), Wave({33, 20}, 1.1f)},
                  bf16_activations);
  add_device_only("matmul_nt_bf16_master_weight",
                  [](const TensorList &in) {
                    return High(Low(in[0])->matmul_nt(in[1]));
                  },
                  {Wave({2, 37, 20}, 0.3f), Wave({33, 20}, 1.1f)},
                  bf16_activations);
  add_low_precision("cross_entropy_bf16",
                    [](const TensorList &in) {
                      return micrograd::cross_entropy(
                          Low(in[0]->mul(3.0f)), {0, 5, 36, 12, 1, 20});
                    },
                    {Wave({6, 37}, 0.3f)}, bf16_softmax);
  add_low_precision("cross_entropy_bf16_padded",
                    [](const TensorList &in) {
                      return micrograd::cross_entropy(
                          Low(in[0]->mul(3.0f)), {0, 5, 36, 12, 1, 20}, 37);
                    },
                    {Wave({6, 40}, 0.3f)}, bf16_softmax);
  add_low_precision("causal_self_attention_bf16_activations",
                    [](const TensorList &in) {
                      return High(Low(in[0])->causal_self_attention(
                          Low(in[1]), Low(in[2]), 2, 0.125f));
                    },
                    {Wave({2, 70, 128}, 0.4f), Wave({2, 70, 128}, 1.3f),
                     Wave({2, 70, 128}, 2.2f)},
                    bf16_attention_activations);
  add_low_precision("causal_self_attention_bf16_three_tiles",
                    [](const TensorList &in) {
                      return High(Low(in[0]->mul(2.0f))
                                      ->causal_self_attention(
                                          Low(in[1]->mul(2.0f)), Low(in[2]), 3,
                                          0.125f));
                    },
                    {Noise({1, 150, 192}, 1), Noise({1, 150, 192}, 2),
                     Noise({1, 150, 192}, 3)},
                    bf16_attention);
  add_low_precision(
      "gpt_block_bf16",
      [](const TensorList &in) {
        const TensorPtr x = Low(in[0]);
        const TensorPtr h = x->layer_norm({64}, in[1], in[2], 1e-5f)
                                ->matmul(Low(in[3]))
                                ->add_bias(in[4])
                                ->gelu()
                                ->matmul(Low(in[5]));
        return High(x->add(h));
      },
      {Noise({2, 9, 64}, 2), Wave({64}, 1.0f), Wave({64}, 2.0f),
       Wave({64, 96}, 0.7f)->mul(0.2f), Wave({96}, 1.5f),
       Wave({96, 64}, 2.5f)->mul(0.2f)},
      bf16_attention);
  add_device_only(
      "rms_norm_bf16_head_width",
      [](const TensorList &in) {
        return High(Low(in[0])->rms_norm({64}, in[1], 1e-5f));
      },
      {Wave({3, 5, 64}, 0.4f), Wave({64}, 1.0f)}, bf16_activations);
  add_device_only(
      "rms_norm_bf16_gpt2_width",
      [](const TensorList &in) {
        return High(Low(in[0])->rms_norm({768}, in[1], 1e-5f));
      },
      {Wave({3, 5, 768}, 0.4f), Wave({768}, 1.0f)}, bf16_row_sums);
  add_device_only(
      "rms_norm_bf16_odd_width",
      [](const TensorList &in) {
        return High(Low(in[0])->rms_norm({37}, in[1], 1e-5f));
      },
      {Wave({4, 37}, 0.4f), Wave({37}, 1.0f)}, bf16_activations);
  add_device_only(
      "rotary_bf16_seq_dim",
      [](const TensorList &in) {
        return High(Low(in[0])->rotary_embedding(10000.0f, 1));
      },
      {Wave({2, 7, 3, 16}, 0.4f)}, bf16_activations);
  add_device_only(
      "rotary_bf16_narrow_head",
      [](const TensorList &in) { return High(Low(in[0])->rotary_embedding()); },
      {Wave({2, 3, 7, 6}, 0.4f)}, bf16_activations);
  add_device_only(
      "relu_squared_bf16",
      [](const TensorList &in) {
        return High(Low(in[0]->mul(2.0f))->relu_squared());
      },
      {Wave({3, 45}, 0.4f)}, bf16_activations);
  add_device_only(
      "head_norm_rotary_bf16",
      [](const TensorList &in) {
        return High(Low(in[0])->head_norm_rotary(4, in[1], 1e-5f, true));
      },
      {Wave({2, 9, 256}, 0.4f), Wave({64}, 1.0f)}, bf16_activations);
  add_device_only(
      "head_norm_rotary_bf16_wide_head",
      [](const TensorList &in) {
        return High(Low(in[0])->head_norm_rotary(2, in[1], 1e-5f, true));
      },
      {Wave({2, 5, 200}, 0.4f), Wave({100}, 1.0f)}, bf16_activations);
  add_device_only(
      "head_norm_rotary_bf16_rotate_only",
      [](const TensorList &in) {
        return High(Low(in[0])->head_norm_rotary(4, nullptr, 0.0f, true));
      },
      {Wave({2, 9, 256}, 0.4f)}, bf16_activations);
  add_device_only("head_norm_rotary_fp32_cuda",
                  [](const TensorList &in) {
                    return in[0]->head_norm_rotary(4, in[1], 1e-5f, true);
                  },
                  {Wave({2, 9, 256}, 0.4f), Wave({64}, 1.0f)},
                  {.rtol = 1e-4f, .atol = 1e-5f});
  add_device_only(
      "qk_norm_rotary_bf16",
      [](const TensorList &in) {
        return High(Low(in[0])
                        ->rms_norm({16}, in[1], 1e-5f)
                        ->rotary_embedding(10000.0f, 1)
                        ->reshape({2, 9, 64}));
      },
      {Wave({2, 9, 4, 16}, 0.4f), Wave({16}, 1.0f)}, bf16_activations);
  add("layer_norm",
      [](const TensorList &in) {
        return in[0]->layer_norm({3}, in[1], in[2], 1e-5f);
      },
      {signed_matrix(), positive_row(), row_vector()});
  add("layer_norm_varied_rows",
      [](const TensorList &in) {
        return in[0]->layer_norm({5}, in[1], in[2], 1e-5f);
      },
      {Wave({4, 5}, 0.3f), Wave({5}, 1.1f), Wave({5}, 2.0f)}, {.eps = 2e-3f});

  const auto token_ids = [](std::vector<size_t> shape,
                            std::vector<scalar_t> ids, Device device) {
    auto tensor = Tensor::from_values(std::move(shape), std::move(ids));
    tensor->to(device);
    return tensor;
  };
  add("embedding_lookup_repeated_ids",
      [token_ids](const TensorList &in) {
        return in[0]->embedding_lookup(
            token_ids({3}, {0.0f, 0.0f, 2.0f}, in[0]->backend()));
      },
      {Wave({4, 3}, 0.5f)});
  add("embedding_tied_lm_head",
      [token_ids](const TensorList &in) {
        const TensorPtr hidden = in[0]->embedding_lookup(token_ids(
            {1, 4}, {3.0f, 1.0f, 3.0f, 0.0f}, in[0]->backend()));
        return micrograd::cross_entropy(hidden->tanh()->matmul_nt(in[0]),
                                        {1, 3, 0, 2});
      },
      {Wave({5, 3}, 0.5f)});

  add("rms_norm",
      [](const TensorList &in) { return in[0]->rms_norm({3}, in[1], 1e-5f); },
      {signed_matrix(), positive_row()});

  add("rotary_embedding",
      [](const TensorList &in) { return in[0]->rotary_embedding(); },
      {Ramp({2, 4}, -1.0f, 0.2f)});
  add("head_norm_rotary",
      [](const TensorList &in) {
        return in[0]->head_norm_rotary(2, in[1], 1e-5f, true);
      },
      {Ramp({2, 3, 8}, -1.0f, 0.09f), Ramp({4}, 0.6f, 0.3f)}, {.atol = 5e-5f});
  add("head_norm_rotary_norm_only",
      [](const TensorList &in) {
        return in[0]->head_norm_rotary(2, in[1], 1e-5f, false);
      },
      {Ramp({2, 3, 8}, -1.0f, 0.09f), Ramp({4}, 0.6f, 0.3f)}, {.atol = 5e-5f});
  add("head_norm_rotary_rotate_only",
      [](const TensorList &in) {
        return in[0]->head_norm_rotary(2, nullptr, 0.0f, true);
      },
      {Ramp({2, 3, 8}, -1.0f, 0.09f)}, {.atol = 5e-5f});
  add("rotary_embedding_seq_dim",
      [](const TensorList &in) { return in[0]->rotary_embedding(10000.0f, 1); },
      {Ramp({2, 3, 2, 4}, -1.0f, 0.1f)}, {.atol = 5e-5f});

  add("flash_attention",
      [](const TensorList &in) {
        return in[0]->flash_attention(in[1], in[2], 0.5f);
      },
      {Ramp({2, 3, 2}, -0.9f, 0.3f), Ramp({2, 3, 2}, -0.5f, 0.25f),
       Ramp({2, 3, 2}, 0.2f, 0.15f)});
  add("causal_attention",
      [](const TensorList &in) {
        return micrograd::gpt::causal_attention(
            in[0], in[1], in[2], 0.5f,
            micrograd::gpt::causal_mask(3, in[0]->backend()));
      },
      {Ramp({2, 3, 2}, -0.9f, 0.3f), Ramp({2, 3, 2}, -0.5f, 0.25f),
       Ramp({2, 3, 2}, 0.2f, 0.15f)});
  add("causal_attention_past_one_tile",
      [](const TensorList &in) {
        return micrograd::gpt::causal_attention(
            in[0], in[1], in[2], 0.35f,
            micrograd::gpt::causal_mask(33, in[0]->backend()));
      },
      {Wave({1, 33, 4}, 0.1f), Wave({1, 33, 4}, 0.8f), Wave({1, 33, 4}, 1.7f)},
      {.atol = 1e-4f});

  add("causal_attention_probs",
      [](const TensorList &in) {
        return in[0]->causal_attention_probs(in[1], 2, 0.5f);
      },
      {Wave({2, 3, 4}, 0.3f), Wave({2, 3, 4}, 1.2f)});
  add("attention_combine",
      [](const TensorList &in) { return in[0]->attention_combine(in[1]); },
      {Wave({2, 2, 3, 5}, 0.4f), Wave({2, 5, 4}, 0.9f)});
  add("fused_causal_attention",
      [](const TensorList &in) {
        auto q = in[0]->mul(1.5f);
        auto k = in[1]->add(0.25f);
        auto v = in[2]->mul(0.75f);
        return q->causal_attention_probs(k, 2, 0.4f)->attention_combine(v);
      },
      {Wave({2, 4, 6}, 0.2f), Wave({2, 4, 6}, 1.0f), Wave({2, 4, 6}, 2.1f)});
  add("fused_causal_attention_past_one_tile",
      [](const TensorList &in) {
        auto q = in[0]->add(0.0f);
        return q->causal_attention_probs(q, 3, 0.35f)->attention_combine(in[1]);
      },
      {Wave({2, 37, 12}, 0.1f), Wave({2, 37, 12}, 1.7f)}, {.atol = 1e-4f});
  add("causal_self_attention",
      [](const TensorList &in) {
        return in[0]->mul(1.2f)->causal_self_attention(in[1], in[2], 2, 0.4f);
      },
      {Wave({2, 5, 8}, 0.2f), Wave({2, 5, 8}, 1.0f), Wave({2, 5, 8}, 2.1f)},
      {.eps = 2e-3f, .atol = 5e-4f});
  add("causal_self_attention_head_dim_64",
      [](const TensorList &in) {
        return in[0]->mul(0.5f)->causal_self_attention(in[1], in[2], 1, 0.125f);
      },
      {Wave({1, 20, 64}, 0.2f), Wave({1, 20, 64}, 1.0f),
       Wave({1, 20, 64}, 2.1f)},
      {.atol = 1e-3f});
  add_device_only("causal_self_attention_shared_qk",
      [](const TensorList &in) {
        auto q = in[0]->add(0.0f);
        return q->causal_self_attention(q, in[1], 2, 0.125f);
      },
      {Wave({2, 70, 128}, 0.4f), Wave({2, 70, 128}, 1.3f)},
      {.atol = 1e-4f});
  const auto flash_case = [](const TensorList &in) {
    return in[0]->mul(0.8f)->causal_self_attention(in[1]->add(0.1f), in[2], 2,
                                                   0.125f);
  };
  const TensorList flash_inputs = {Wave({2, 70, 128}, 0.4f),
                                   Wave({2, 70, 128}, 1.3f),
                                   Wave({2, 70, 128}, 2.2f)};
  add_device_only("causal_self_attention_past_one_tile", flash_case,
                  flash_inputs, {.atol = 1e-4f});
  add_device_only("causal_self_attention_tf32_kernels", flash_case,
                  flash_inputs, {.rtol = 1e-2f, .atol = 2e-3f}, true);
  add_device_only("causal_self_attention_tf32_long",
                  [](const TensorList &in) {
                    return in[0]->causal_self_attention(in[1], in[2], 3,
                                                        0.125f);
                  },
                  {Noise({1, 200, 192}, 1), Noise({1, 200, 192}, 2),
                   Noise({1, 200, 192}, 3)},
                  {.rtol = 1e-2f, .atol = 2e-3f}, true);
  add_sampled("causal_self_attention_70x128_sampled", flash_case,
              {Noise({2, 70, 128}, 4), Noise({2, 70, 128}, 5),
               Noise({2, 70, 128}, 6)},
              {.rtol = 1e-2f, .atol = 1e-4f}, 32);
  add_sampled("causal_self_attention_200x192_sampled",
              [](const TensorList &in) {
                return in[0]->causal_self_attention(in[1], in[2], 3, 0.125f);
              },
              {Noise({1, 200, 192}, 7), Noise({1, 200, 192}, 8),
               Noise({1, 200, 192}, 9)},
              {.rtol = 1e-2f, .atol = 1e-4f}, 24);
  add("residual_block_chain",
      [](const TensorList &in) {
        auto x = in[0]->add(0.0f);
        auto h = x->layer_norm({4}, in[1], in[2], 1e-5f);
        auto y = h->matmul(in[3])->add_bias(in[4])->gelu();
        auto r = x->add(y->reshape({2, 3, 4}));
        return r->add(r->mul(0.5f));
      },
      {Wave({2, 3, 4}, 0.3f), Wave({4}, 1.1f), Wave({4}, 0.6f),
       Wave({4, 4}, 1.9f), Wave({4}, 0.8f)},
      {.rtol = 5e-3f});

  add("relu", [](const TensorList &in) { return in[0]->relu(); },
      {signed_matrix()});
  add("sigmoid", [](const TensorList &in) { return in[0]->sigmoid(); },
      {signed_matrix()});
  add("tanh", [](const TensorList &in) { return in[0]->tanh(); },
      {signed_matrix()});
  add("gelu_wide", [](const TensorList &in) { return in[0]->gelu(); },
      {Wave({7, 9}, 0.3f)}, {.atol = 5e-5f});
  add("exp", [](const TensorList &in) { return in[0]->exp(); },
      {signed_matrix()});
  add("log", [](const TensorList &in) { return in[0]->log(); },
      {positive_matrix()});
  add("sqrt", [](const TensorList &in) { return in[0]->sqrt(); },
      {positive_matrix()});
  add("neg", [](const TensorList &in) { return in[0]->neg(); },
      {signed_matrix()});
  add("softmax", [](const TensorList &in) { return in[0]->softmax(1); },
      {signed_matrix()});
  add("log_softmax", [](const TensorList &in) { return in[0]->log_softmax(1); },
      {signed_matrix()});
  add("softmax_dim0", [](const TensorList &in) { return in[0]->softmax(0); },
      {signed_cube()});
  add("log_softmax_dim0",
      [](const TensorList &in) { return in[0]->log_softmax(0); },
      {signed_cube()});
  add("softmax_negative_dim",
      [](const TensorList &in) { return in[0]->softmax(-2); }, {signed_cube()});

  add("relu_squared",
      [](const TensorList &in) { return micrograd::relu_squared(in[0]); },
      {signed_matrix()});
  add("gelu", [](const TensorList &in) { return micrograd::gelu(in[0]); },
      {Ramp({2, 3}, -1.4f, 0.5f)});
  add("qk_norm", [](const TensorList &in) { return micrograd::qk_norm(in[0]); },
      {signed_matrix()});

  add("clamp",
      [](const TensorList &in) { return in[0]->clamp(-0.5f, 0.5f); },
      {signed_matrix()});
  add("mse_loss",
      [](const TensorList &in) { return micrograd::mse_loss(in[0], in[1]); },
      {signed_matrix(), positive_matrix()});
  add("cross_entropy",
      [](const TensorList &in) {
        return micrograd::cross_entropy(in[0], {0, 2});
      },
      {signed_matrix()});
  add("cross_entropy_rank3",
      [](const TensorList &in) {
        return micrograd::cross_entropy(in[0], {0, 2, 1, 2, 0, 1});
      },
      {signed_cube()});
  add("cross_entropy_padded_classes",
      [](const TensorList &in) {
        return micrograd::cross_entropy(in[0], {0, 2, 1, 2, 0, 1}, 3);
      },
      {Ramp({2, 3, 5}, -1.1f, 0.15f)});
  add("binary_cross_entropy",
      [](const TensorList &in) {
        return micrograd::binary_cross_entropy(in[0], in[1]);
      },
      {Ramp({2, 3}, 0.3f, 0.08f), Ramp({2, 3}, 0.2f, 0.12f)});
  add("grpo_loss",
      [](const TensorList &in) {
        return micrograd::gpt::grpo_loss(in[0], {0, 2}, {1.0f, -0.5f},
                                         {-0.2f, -0.3f}, 0.1f);
      },
      {signed_matrix()});

  add("linear_relu_cross_entropy",
      [](const TensorList &in) {
        auto hidden = in[0]->matmul(in[1])->add(in[2])->relu();
        return micrograd::cross_entropy(hidden->matmul(in[3]), {1, 0});
      },
      {Ramp({2, 3}, -0.7f, 0.3f), Ramp({3, 4}, -0.5f, 0.2f),
       Ramp({1, 4}, -0.2f, 0.3f), Ramp({4, 3}, -0.6f, 0.15f)});

  add("checkpoint_dropout",
      [](const TensorList &in) {
        micrograd::manual_seed(7);
        auto dropout = std::make_shared<micrograd::Dropout>(0.5f);
        const TensorPtr &weight = in[1];
        auto block = [dropout, weight](const TensorPtr &x) {
          return dropout->forward(x->matmul(weight));
        };
        return Tensor::checkpoint(in[0], block);
      },
      {Ramp({2, 3}, -0.9f, 0.3f), Ramp({3, 4}, -0.6f, 0.2f)});

  return cases;
}

Device ParseDevice(const std::string &value) {
  if (value == "cpu") {
    return Device::CPU;
  }
  if (value == "cuda") {
    return Device::CUDA;
  }
  throw std::invalid_argument("Unknown MICROGRAD_DEVICE: " + value);
}

Device DeviceFromEnv() {
  const char *value = std::getenv("MICROGRAD_DEVICE");
  if (value == nullptr) {
    return Device::CPU;
  }
  return ParseDevice(value);
}

const char *DeviceName(Device device) {
  switch (device) {
    case Device::CPU:
      return "cpu";
    case Device::CUDA:
      return "cuda";
  }
  throw std::invalid_argument("Unknown device");
}

constexpr size_t kOutputSlot = std::numeric_limits<size_t>::max();

struct DeviceMismatch {
  size_t input;
  size_t index;
  scalar_t cpu;
  scalar_t device;
};

struct SlotStats {
  size_t slot;
  size_t count;
  double cpu_rms;
  double max_difference;
  double worst_ratio;
  double slope;
  double shrink;
};

struct DeviceCheck {
  std::vector<DeviceMismatch> mismatches;
  std::vector<SlotStats> stats;
};

struct DeviceRun {
  std::vector<scalar_t> output;
  std::vector<std::vector<scalar_t>> gradients;
};

DeviceRun RunForwardBackward(const GradCheckFunction &function,
                             const std::vector<std::vector<size_t>> &shapes,
                             const std::vector<std::vector<scalar_t>> &values,
                             Device device) {
  auto leaves = MakeGradCheckLeaves(shapes, values);
  for (auto &leaf : leaves) {
    leaf->to(device);
  }

  auto output = function(leaves);
  const std::vector<scalar_t> weights = ProjectionWeights(output->size());
  const auto seed = Tensor::from_values(output->shape(), weights);
  output->backward(*seed);

  DeviceRun run;
  output->to(Device::CPU);
  std::span<const scalar_t> output_span = output->data();
  run.output.assign(output_span.begin(), output_span.end());

  run.gradients.reserve(leaves.size());
  for (auto &leaf : leaves) {
    leaf->to(Device::CPU);
    std::span<const scalar_t> span = leaf->grad();
    run.gradients.emplace_back(span.begin(), span.end());
  }
  return run;
}

void CompareValues(const std::vector<scalar_t> &cpu,
                   const std::vector<scalar_t> &device, size_t slot,
                   const GradCheckOptions &options,
                   std::vector<DeviceMismatch> &mismatches) {
  for (size_t j = 0; j < cpu.size(); j++) {
    const scalar_t difference = std::abs(cpu[j] - device[j]);
    const scalar_t scale = std::max(std::abs(cpu[j]), std::abs(device[j]));
    if (difference > options.atol + (options.rtol * scale)) {
      mismatches.push_back(
          {.input = slot, .index = j, .cpu = cpu[j], .device = device[j]});
    }
  }
}

SlotStats ComputeSlotStats(const std::vector<scalar_t> &cpu,
                           const std::vector<scalar_t> &device, size_t slot,
                           const GradCheckOptions &options) {
  SlotStats stats{.slot = slot,
                  .count = cpu.size(),
                  .cpu_rms = 0.0,
                  .max_difference = 0.0,
                  .worst_ratio = 0.0,
                  .slope = 1.0,
                  .shrink = 0.0};
  if (cpu.empty()) {
    return stats;
  }

  double cpu_squares = 0.0;
  double cross = 0.0;
  double toward_zero = 0.0;
  for (size_t j = 0; j < cpu.size(); j++) {
    const auto reference = static_cast<double>(cpu[j]);
    const auto measured = static_cast<double>(device[j]);
    const double difference = std::abs(reference - measured);
    const double scale = std::max(std::abs(reference), std::abs(measured));
    const double allowed = static_cast<double>(options.atol) +
                           (static_cast<double>(options.rtol) * scale);
    cpu_squares += reference * reference;
    cross += reference * measured;
    toward_zero += std::abs(reference) - std::abs(measured);
    stats.max_difference = std::max(stats.max_difference, difference);
    if (allowed > 0.0) {
      stats.worst_ratio = std::max(stats.worst_ratio, difference / allowed);
    }
  }

  const auto count = static_cast<double>(cpu.size());
  stats.cpu_rms = std::sqrt(cpu_squares / count);
  if (cpu_squares > 0.0) {
    stats.slope = cross / cpu_squares;
    stats.shrink = (toward_zero / count) / stats.cpu_rms;
  }
  return stats;
}

GradCheckOptions SlotOptions(const Case &test,
                             const std::vector<scalar_t> &cpu) {
  GradCheckOptions options = test.options;
  if (test.slot_scaled_atol && !cpu.empty()) {
    double sum_squares = 0.0;
    for (scalar_t value : cpu) {
      sum_squares += static_cast<double>(value) * static_cast<double>(value);
    }
    options.atol *= static_cast<scalar_t>(
        std::sqrt(sum_squares / static_cast<double>(cpu.size())));
  }
  return options;
}

void Snapshot(const TensorList &inputs,
              std::vector<std::vector<size_t>> &shapes,
              std::vector<std::vector<scalar_t>> &values) {
  for (const auto &input : inputs) {
    std::span<const scalar_t> span = input->data();
    shapes.push_back(input->shape());
    values.emplace_back(span.begin(), span.end());
  }
}

DeviceRun RunCase(const Case &test, Device device) {
  std::vector<std::vector<size_t>> shapes;
  std::vector<std::vector<scalar_t>> values;
  Snapshot(test.inputs, shapes, values);
  return RunForwardBackward(test.function, shapes, values, device);
}

DeviceCheck CheckOnDevice(const Case &test, Device device) {
  const DeviceRun cpu = RunCase(test, Device::CPU);
  const DeviceRun other = RunCase(test, device);

  DeviceCheck check;
  const GradCheckOptions output_options = SlotOptions(test, cpu.output);
  CompareValues(cpu.output, other.output, kOutputSlot, output_options,
                check.mismatches);
  check.stats.push_back(
      ComputeSlotStats(cpu.output, other.output, kOutputSlot, output_options));
  for (size_t i = 0; i < cpu.gradients.size(); i++) {
    const GradCheckOptions options = SlotOptions(test, cpu.gradients[i]);
    CompareValues(cpu.gradients[i], other.gradients[i], i, options,
                  check.mismatches);
    check.stats.push_back(
        ComputeSlotStats(cpu.gradients[i], other.gradients[i], i, options));
  }
  return check;
}

std::string SlotName(size_t slot) {
  return slot == kOutputSlot ? std::string("output")
                             : "input " + std::to_string(slot);
}

void PrintSlotStats(const std::vector<SlotStats> &stats) {
  for (const auto &slot : stats) {
    std::printf(
        "      %s: n %zu rms %.3g max diff %.3g worst %.2fx allowed slope %.4f "
        "shrink %.2f%% of rms\n",
        SlotName(slot.slot).c_str(), slot.count, slot.cpu_rms,
        slot.max_difference, slot.worst_ratio, slot.slope,
        100.0 * slot.shrink);
  }
}

std::vector<micrograd::GradCheckMismatch> SampledGradCheck(const Case &test) {
  std::vector<std::vector<size_t>> shapes;
  std::vector<std::vector<scalar_t>> values;
  Snapshot(test.inputs, shapes, values);
  const DeviceRun analytic =
      RunForwardBackward(test.function, shapes, values, Device::CPU);
  const std::vector<scalar_t> weights =
      ProjectionWeights(analytic.output.size());
  const GradCheckOptions &options = test.options;

  std::vector<micrograd::GradCheckMismatch> mismatches;
  for (size_t i = 0; i < values.size(); i++) {
    const size_t count = values[i].size();
    for (size_t s = 0; s < test.samples; s++) {
      const size_t j = ((2 * s + 1) * count) / (2 * test.samples);
      const scalar_t original = values[i][j];
      values[i][j] = original + options.eps;
      const double up = micrograd::internal::Project(
          test.function(MakeGradCheckLeaves(shapes, values)), weights);
      values[i][j] = original - options.eps;
      const double down = micrograd::internal::Project(
          test.function(MakeGradCheckLeaves(shapes, values)), weights);
      values[i][j] = original;

      const auto numeric = static_cast<scalar_t>(
          (up - down) / (2.0 * static_cast<double>(options.eps)));
      const scalar_t expected = analytic.gradients[i][j];
      const scalar_t scale = std::max(std::abs(expected), std::abs(numeric));
      const scalar_t allowed = options.atol + (options.rtol * scale);
      if (std::abs(expected - numeric) > allowed) {
        mismatches.push_back({.input = i,
                              .index = j,
                              .analytic = expected,
                              .numeric = numeric});
      }
    }
  }
  return mismatches;
}

constexpr double kMaxZeroPassShare = 0.3;

double ZeroPassShare(const Case &test, const std::vector<scalar_t> &cpu) {
  const size_t nonzero = static_cast<size_t>(
      std::ranges::count_if(cpu, [](scalar_t value) { return value != 0.0f; }));
  if (nonzero == 0) {
    return 0.0;
  }
  std::vector<DeviceMismatch> mismatches;
  CompareValues(cpu, std::vector<scalar_t>(cpu.size(), 0.0f), 0,
                SlotOptions(test, cpu), mismatches);
  return static_cast<double>(nonzero - mismatches.size()) /
         static_cast<double>(nonzero);
}

size_t RunZeroGradientSelfCheck(const std::vector<Case> &cases) {
  size_t failed = 0;
  size_t checked = 0;
  for (const auto &test : cases) {
    if (!test.device_only) {
      continue;
    }
    checked++;
    const DeviceRun cpu = RunCase(test, Device::CPU);
    std::vector<std::pair<std::string, double>> shares;
    shares.emplace_back("output", ZeroPassShare(test, cpu.output));
    for (size_t i = 0; i < cpu.gradients.size(); i++) {
      shares.emplace_back("input " + std::to_string(i),
                          ZeroPassShare(test, cpu.gradients[i]));
    }

    const bool ok = std::ranges::all_of(shares, [](const auto &entry) {
      return entry.second <= kMaxZeroPassShare;
    });
    std::printf("%s  zero %s:", ok ? "ok  " : "FAIL", test.name.c_str());
    for (const auto &[slot, share] : shares) {
      std::printf(" %s %.0f%%", slot.c_str(), 100.0 * share);
    }
    std::printf("\n");
    if (!ok) {
      failed++;
    }
  }
  std::printf("%zu of %zu zero-result self-checks passed\n", checked - failed,
              checked);
  return failed;
}

void ProbeDevice(Device device) {
  auto probe = Tensor::zeros({1});
  probe->to(device);
}

size_t RunAllCases(Device device) {
  const std::vector<Case> cases = AllCases();
  if (device != Device::CPU) {
    ProbeDevice(device);
  }

  const bool print_stats = std::getenv("MICROGRAD_GRADCHECK_STATS") != nullptr;
  size_t failed = 0;
  size_t skipped = 0;
  for (const auto &test : cases) {
    if (device == Device::CPU && test.device_only) {
      skipped++;
      continue;
    }
    if (device == Device::CPU) {
      const auto mismatches =
          test.samples > 0
              ? SampledGradCheck(test)
              : GradCheck(test.function, test.inputs, test.options);
      if (mismatches.empty()) {
        std::printf("ok    %s\n", test.name.c_str());
        continue;
      }

      failed++;
      std::printf("FAIL  %s\n", test.name.c_str());
      for (const auto &mismatch : mismatches) {
        std::printf("      input %zu index %zu analytic %g numeric %g\n",
                    mismatch.input, mismatch.index,
                    static_cast<double>(mismatch.analytic),
                    static_cast<double>(mismatch.numeric));
      }
      continue;
    }

    DeviceCheck check;
    try {
#ifdef MICROGRAD_CUDA_ENABLED
      micrograd::CudaContext::instance().setAllowTf32(test.tf32);
      micrograd::CudaContext::instance().setAllowBf16Matmul(test.bf16);
#endif
      check = CheckOnDevice(test, device);
#ifdef MICROGRAD_CUDA_ENABLED
      micrograd::CudaContext::instance().setAllowTf32(false);
      micrograd::CudaContext::instance().setAllowBf16Matmul(false);
#endif
    } catch (const std::exception &error) {
      failed++;
      std::printf("FAIL  %s: %s\n", test.name.c_str(), error.what());
      continue;
    }

    if (check.mismatches.empty()) {
      std::printf("ok    %s\n", test.name.c_str());
      if (print_stats) {
        PrintSlotStats(check.stats);
      }
      continue;
    }

    failed++;
    std::printf("FAIL  %s\n", test.name.c_str());
    PrintSlotStats(check.stats);
    for (const auto &mismatch : check.mismatches) {
      std::printf("      %s index %zu cpu %g %s %g\n",
                  SlotName(mismatch.input).c_str(), mismatch.index,
                  static_cast<double>(mismatch.cpu), DeviceName(device),
                  static_cast<double>(mismatch.device));
    }
  }

  std::printf("%zu of %zu gradient checks passed on %s\n",
              cases.size() - skipped - failed, cases.size() - skipped,
              DeviceName(device));
  return failed + RunZeroGradientSelfCheck(cases);
}

}  // namespace

int main() {
  try {
    const Device device = DeviceFromEnv();
    return RunAllCases(device) == 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::printf("gradcheck raised: %s\n", error.what());
    return 1;
  }
}
