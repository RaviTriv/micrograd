#include "examples/gpt/CoreEval.h"

#include <limits>
#include <stdexcept>

namespace micrograd::gpt {

namespace {

size_t predict(CoreMetric metric, const CoreExample &example,
               const ContinuationLogLikelihood &score) {
  size_t best = 0;
  double best_value = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < example.continuations.size(); i++) {
    const std::vector<int32_t> &continuation = example.continuations[i];
    double log_likelihood = score(example.context, continuation);
    double value =
        metric == CoreMetric::kAccNorm
            ? log_likelihood / static_cast<double>(continuation.size())
            : log_likelihood;
    if (value > best_value) {
      best_value = value;
      best = i;
    }
  }
  return best;
}

}  // namespace

CoreTaskResult evaluate_core_task(const CoreTask &task,
                                  const ContinuationLogLikelihood &score) {
  if (task.examples.empty()) {
    throw std::invalid_argument("evaluate_core_task: task has no examples");
  }

  size_t correct = 0;
  for (const CoreExample &example : task.examples) {
    if (example.continuations.empty()) {
      throw std::invalid_argument(
          "evaluate_core_task: example has no continuations");
    }
    if (example.gold_index >= example.continuations.size()) {
      throw std::invalid_argument(
          "evaluate_core_task: gold_index out of range");
    }
    if (predict(task.metric, example, score) == example.gold_index) {
      correct++;
    }
  }

  double accuracy =
      static_cast<double>(correct) / static_cast<double>(task.examples.size());
  double centered =
      (accuracy - task.random_baseline) / (1.0 - task.random_baseline);
  return {task.name, accuracy, centered, task.examples.size()};
}

double core_score(const std::vector<CoreTaskResult> &results) {
  if (results.empty()) {
    throw std::invalid_argument("core_score: no task results given");
  }
  double sum = 0.0;
  for (const CoreTaskResult &result : results) {
    sum += result.centered_accuracy;
  }
  return sum / static_cast<double>(results.size());
}

}  // namespace micrograd::gpt
