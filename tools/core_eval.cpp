#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "examples/gpt/CoreEval.h"

namespace {

using micrograd::gpt::ContinuationLogLikelihood;
using micrograd::gpt::core_score;
using micrograd::gpt::CoreExample;
using micrograd::gpt::CoreMetric;
using micrograd::gpt::CoreTask;
using micrograd::gpt::CoreTaskResult;
using micrograd::gpt::evaluate_core_task;

bool Close(double a, double b) { return std::fabs(a - b) < 1e-9; }

double PerTokenScore(int32_t token) { return token == 9 ? 3.0 : 2.0; }

ContinuationLogLikelihood SumPerTokenScore() {
  return [](const std::vector<int32_t> &,
            const std::vector<int32_t> &continuation) {
    double total = 0.0;
    for (int32_t token : continuation) {
      total += PerTokenScore(token);
    }
    return total;
  };
}

size_t CheckAccuracy(const std::string &name, CoreMetric metric,
                     double random_baseline, double expected_accuracy) {
  CoreTask task{
      name, metric, random_baseline, {CoreExample{{}, {{9}, {5, 5, 5}}, 0}}};
  CoreTaskResult result = evaluate_core_task(task, SumPerTokenScore());

  size_t failed = 0;
  if (!Close(result.accuracy, expected_accuracy)) {
    std::printf("FAIL  %s accuracy %.3f, want %.3f\n", name.c_str(),
                result.accuracy, expected_accuracy);
    failed++;
  }
  double expected_centered =
      (expected_accuracy - random_baseline) / (1.0 - random_baseline);
  if (!Close(result.centered_accuracy, expected_centered)) {
    std::printf("FAIL  %s centered accuracy %.3f, want %.3f\n", name.c_str(),
                result.centered_accuracy, expected_centered);
    failed++;
  }
  if (failed == 0) {
    std::printf("ok    %s\n", name.c_str());
  }
  return failed;
}

size_t CheckAggregation() {
  std::vector<CoreTaskResult> results = {
      {"a", 1.0, 0.5, 1},
      {"b", 0.0, -0.25, 1},
      {"c", 0.75, 0.5, 1},
  };
  double expected = (0.5 - 0.25 + 0.5) / 3.0;
  double actual = core_score(results);
  if (!Close(actual, expected)) {
    std::printf("FAIL  core_score %.3f, want %.3f\n", actual, expected);
    return 1;
  }
  std::printf("ok    core_score\n");
  return 0;
}

size_t CheckEmptyTaskThrows() {
  CoreTask task{"empty", CoreMetric::kAcc, 0.5, {}};
  try {
    evaluate_core_task(task, SumPerTokenScore());
  } catch (const std::exception &) {
    std::printf("ok    empty task throws\n");
    return 0;
  }
  std::printf("FAIL  empty task did not throw\n");
  return 1;
}

}  // namespace

int main() {
  try {
    size_t failed = 0;
    failed += CheckAccuracy("acc_prefers_raw_sum", CoreMetric::kAcc, 0.5, 0.0);
    failed += CheckAccuracy("acc_norm_prefers_per_token", CoreMetric::kAccNorm,
                            0.5, 1.0);
    failed += CheckAggregation();
    failed += CheckEmptyTaskThrows();

    std::printf("%zu check(s) failed\n", failed);
    return failed == 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::printf("core_eval raised: %s\n", error.what());
    return 1;
  }
}
