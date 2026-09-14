#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace micrograd::gpt {

enum class CoreMetric { kAcc, kAccNorm };

struct CoreExample {
  std::vector<int32_t> context;
  std::vector<std::vector<int32_t>> continuations;
  size_t gold_index;
};

struct CoreTask {
  std::string name;
  CoreMetric metric;
  double random_baseline;
  std::vector<CoreExample> examples;
};

struct CoreTaskResult {
  std::string name;
  double accuracy;
  double centered_accuracy;
  size_t example_count;
};

using ContinuationLogLikelihood =
    std::function<double(const std::vector<int32_t> &context,
                         const std::vector<int32_t> &continuation)>;

CoreTaskResult evaluate_core_task(const CoreTask &task,
                                  const ContinuationLogLikelihood &score);

double core_score(const std::vector<CoreTaskResult> &results);

}  // namespace micrograd::gpt
