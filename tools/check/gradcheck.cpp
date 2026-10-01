#include "micrograd/GradCheck.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpt/GRPO.h"
#include "micrograd/Device.h"
#include "micrograd/NN.h"
#include "micrograd/Random.h"
#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"

namespace {

using micrograd::Device;
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
  return std::make_shared<Tensor>(std::move(shape), std::move(values));
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
      {Ramp({2, 150, 4}, -0.9f, 0.0015f), Ramp({4, 5}, -1.1f, 0.15f)});

  add("layer_norm",
      [](const TensorList &in) {
        return in[0]->layer_norm({3}, in[1], in[2], 1e-5f);
      },
      {signed_matrix(), positive_row(), row_vector()});

  add("rms_norm",
      [](const TensorList &in) { return in[0]->rms_norm({3}, in[1], 1e-5f); },
      {signed_matrix(), positive_row()});

  add("rotary_embedding",
      [](const TensorList &in) { return in[0]->rotary_embedding(); },
      {Ramp({2, 4}, -1.0f, 0.2f)});

  add("flash_attention",
      [](const TensorList &in) {
        return in[0]->flash_attention(in[1], in[2], 0.5f);
      },
      {Ramp({2, 3, 2}, -0.9f, 0.3f), Ramp({2, 3, 2}, -0.5f, 0.25f),
       Ramp({2, 3, 2}, 0.2f, 0.15f)});

  add("relu", [](const TensorList &in) { return in[0]->relu(); },
      {signed_matrix()});
  add("sigmoid", [](const TensorList &in) { return in[0]->sigmoid(); },
      {signed_matrix()});
  add("tanh", [](const TensorList &in) { return in[0]->tanh(); },
      {signed_matrix()});
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
  add("qk_norm", [](const TensorList &in) { return micrograd::qk_norm(in[0]); },
      {signed_matrix()});

  add("mse_loss",
      [](const TensorList &in) { return micrograd::mse_loss(in[0], in[1]); },
      {signed_matrix(), positive_matrix()});
  add("cross_entropy",
      [](const TensorList &in) {
        return micrograd::cross_entropy(in[0], {0, 2});
      },
      {signed_matrix()});
  add("masked_cross_entropy",
      [](const TensorList &in) {
        return micrograd::masked_cross_entropy(in[0], {0, 2}, {true, false});
      },
      {signed_matrix()});
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
  if (value == "metal") {
    return Device::Metal;
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
    case Device::Metal:
      return "metal";
    case Device::CUDA:
      return "cuda";
  }
  throw std::invalid_argument("Unknown device");
}

struct DeviceMismatch {
  size_t input;
  size_t index;
  scalar_t cpu;
  scalar_t device;
};

std::vector<std::vector<scalar_t>> AnalyticGradients(
    const GradCheckFunction &function,
    const std::vector<std::vector<size_t>> &shapes,
    const std::vector<std::vector<scalar_t>> &values, Device device) {
  auto leaves = MakeGradCheckLeaves(shapes, values);
  for (auto &leaf : leaves) {
    leaf->to(device);
  }

  auto output = function(leaves);
  const std::vector<scalar_t> weights = ProjectionWeights(output->size());
  const Tensor seed(output->shape(), weights);
  output->backward(seed);

  std::vector<std::vector<scalar_t>> gradients;
  gradients.reserve(leaves.size());
  for (auto &leaf : leaves) {
    leaf->to(Device::CPU);
    std::span<const scalar_t> span = leaf->grad();
    gradients.emplace_back(span.begin(), span.end());
  }
  return gradients;
}

std::vector<DeviceMismatch> CheckOnDevice(const GradCheckFunction &function,
                                          const TensorList &inputs,
                                          const GradCheckOptions &options,
                                          Device device) {
  std::vector<std::vector<size_t>> shapes;
  std::vector<std::vector<scalar_t>> values;
  for (const auto &input : inputs) {
    std::span<const scalar_t> span = input->data();
    shapes.push_back(input->shape());
    values.emplace_back(span.begin(), span.end());
  }

  const auto cpu_gradients =
      AnalyticGradients(function, shapes, values, Device::CPU);
  const auto device_gradients =
      AnalyticGradients(function, shapes, values, device);

  std::vector<DeviceMismatch> mismatches;
  for (size_t i = 0; i < cpu_gradients.size(); i++) {
    for (size_t j = 0; j < cpu_gradients[i].size(); j++) {
      const scalar_t cpu_value = cpu_gradients[i][j];
      const scalar_t device_value = device_gradients[i][j];
      const scalar_t difference = std::abs(cpu_value - device_value);
      const scalar_t scale =
          std::max(std::abs(cpu_value), std::abs(device_value));
      if (difference > options.atol + (options.rtol * scale)) {
        mismatches.push_back(
            {.input = i, .index = j, .cpu = cpu_value, .device = device_value});
      }
    }
  }
  return mismatches;
}

void ProbeDevice(Device device) {
  Tensor probe(std::vector<size_t>{1});
  probe.to(device);
}

size_t RunAllCases(Device device) {
  const std::vector<Case> cases = AllCases();
  if (device != Device::CPU) {
    ProbeDevice(device);
  }

  size_t failed = 0;
  for (const auto &test : cases) {
    if (device == Device::CPU) {
      const auto mismatches =
          GradCheck(test.function, test.inputs, test.options);
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

    std::vector<DeviceMismatch> mismatches;
    try {
      mismatches =
          CheckOnDevice(test.function, test.inputs, test.options, device);
    } catch (const std::exception &error) {
      failed++;
      std::printf("FAIL  %s: %s\n", test.name.c_str(), error.what());
      continue;
    }

    if (mismatches.empty()) {
      std::printf("ok    %s\n", test.name.c_str());
      continue;
    }

    failed++;
    std::printf("FAIL  %s\n", test.name.c_str());
    for (const auto &mismatch : mismatches) {
      std::printf("      input %zu index %zu cpu %g %s %g\n", mismatch.input,
                  mismatch.index, static_cast<double>(mismatch.cpu),
                  DeviceName(device), static_cast<double>(mismatch.device));
    }
  }

  std::printf("%zu of %zu gradient checks passed on %s\n",
              cases.size() - failed, cases.size(), DeviceName(device));
  return failed;
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
