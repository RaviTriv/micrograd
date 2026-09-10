#include "micrograd/GradCheck.h"

#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "micrograd/NN.h"
#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"

namespace {

using micrograd::GradCheck;
using micrograd::GradCheckFunction;
using micrograd::GradCheckOptions;
using micrograd::scalar_t;
using micrograd::Tensor;

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

  add("add", [](const TensorList &in) { return in[0]->add(in[1]); },
      {signed_matrix(), positive_matrix()});
  add("add_broadcast", [](const TensorList &in) { return in[0]->add(in[1]); },
      {signed_matrix(), Ramp({3}, 0.4f, 0.7f)});
  add("sub", [](const TensorList &in) { return in[0]->sub(in[1]); },
      {signed_matrix(), positive_matrix()});
  add("mul", [](const TensorList &in) { return in[0]->mul(in[1]); },
      {signed_matrix(), positive_matrix()});
  add("div", [](const TensorList &in) { return in[0]->div(in[1]); },
      {signed_matrix(), positive_matrix()});

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

  add("matmul", [](const TensorList &in) { return in[0]->matmul(in[1]); },
      {Ramp({2, 3}, -0.9f, 0.4f), Ramp({3, 4}, -1.1f, 0.2f)});

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

  add("mse_loss",
      [](const TensorList &in) { return micrograd::mse_loss(in[0], in[1]); },
      {signed_matrix(), positive_matrix()});
  add("cross_entropy",
      [](const TensorList &in) {
        return micrograd::cross_entropy(in[0], {0, 2});
      },
      {signed_matrix()});

  add("linear_relu_cross_entropy",
      [](const TensorList &in) {
        auto hidden = in[0]->matmul(in[1])->add(in[2])->relu();
        return micrograd::cross_entropy(hidden->matmul(in[3]), {1, 0});
      },
      {Ramp({2, 3}, -0.7f, 0.3f), Ramp({3, 4}, -0.5f, 0.2f),
       Ramp({1, 4}, -0.2f, 0.3f), Ramp({4, 3}, -0.6f, 0.15f)});

  return cases;
}

size_t RunAllCases() {
  const std::vector<Case> cases = AllCases();

  size_t failed = 0;
  for (const auto &test : cases) {
    const auto mismatches = GradCheck(test.function, test.inputs, test.options);
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
  }

  std::printf("%zu of %zu gradient checks passed\n", cases.size() - failed,
              cases.size());
  return failed;
}

}  // namespace

int main() {
  try {
    return RunAllCases() == 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::printf("gradcheck raised: %s\n", error.what());
    return 1;
  }
}
