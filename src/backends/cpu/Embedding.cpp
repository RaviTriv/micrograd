#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <span>

#include "micrograd/Tensor.h"
#include "micrograd/backends/cpu/Ops.h"
#include "micrograd/ops/Dispatch.h"

namespace micrograd::ops::cpu {
namespace {

void EmbeddingLookup(const OpArgs &args) {
  size_t dim = args.lhs->shape()[1];
  size_t count = args.rhs->size();
  auto weight = args.lhs->data();
  auto index_values = args.rhs->data();
  auto out_values = args.out->data();

  for (size_t p = 0; p < count; p++) {
    auto index = static_cast<size_t>(std::lround(index_values[p]));
    std::copy_n(&weight[index * dim], dim, &out_values[p * dim]);
  }
}

std::function<void()> EmbeddingLookupBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, indices = args.rhs]() {
    size_t dim = lhs->shape()[1];
    size_t count = indices->size();
    auto gradient = lhs->grad();
    auto incoming = out->grad();
    auto index_values = indices->data();

    for (size_t p = 0; p < count; p++) {
      auto index = static_cast<size_t>(std::lround(index_values[p]));
      for (size_t d = 0; d < dim; d++) {
        gradient[(index * dim) + d] += incoming[(p * dim) + d];
      }
    }
  };
}

void GatherPerRow(const OpArgs &args) {
  size_t rows = args.lhs->shape()[0];
  size_t cols = args.lhs->shape()[1];
  auto values = args.lhs->data();
  auto index_values = args.rhs->data();
  auto out_values = args.out->data();

  for (size_t i = 0; i < rows; i++) {
    auto index = static_cast<size_t>(std::lround(index_values[i]));
    out_values[i] = values[(i * cols) + index];
  }
}

std::function<void()> GatherPerRowBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, indices = args.rhs]() {
    size_t rows = lhs->shape()[0];
    size_t cols = lhs->shape()[1];
    auto gradient = lhs->grad();
    auto incoming = out->grad();
    auto index_values = indices->data();

    for (size_t i = 0; i < rows; i++) {
      auto index = static_cast<size_t>(std::lround(index_values[i]));
      gradient[(i * cols) + index] += incoming[i];
    }
  };
}

double RowLogSumExp(std::span<const scalar_t> row) {
  scalar_t max_value = row[0];
  for (scalar_t v : row) {
    max_value = std::max(max_value, v);
  }
  double sum = 0.0;
  for (scalar_t v : row) {
    sum += std::exp(static_cast<double>(v - max_value));
  }
  return static_cast<double>(max_value) + std::log(sum);
}

void CrossEntropy(const OpArgs &args) {
  size_t cols = args.lhs->shape().back();
  size_t rows = args.lhs->size() / cols;
  auto values = args.lhs->data();
  auto targets = args.rhs->data();
  auto out_values = args.out->data();

  for (size_t i = 0; i < rows; i++) {
    auto row = values.subspan(i * cols, cols);
    auto target = static_cast<size_t>(std::lround(targets[i]));
    out_values[i] =
        static_cast<scalar_t>(RowLogSumExp(row) - static_cast<double>(row[target]));
  }
}

std::function<void()> CrossEntropyBackward(const GradArgs &args) {
  return [out = args.out, lhs = args.lhs, targets = args.rhs]() {
    size_t cols = lhs->shape().back();
    size_t rows = lhs->size() / cols;
    auto values = lhs->data();
    auto gradient = lhs->grad();
    auto incoming = out->grad();
    auto target_values = targets->data();

    for (size_t i = 0; i < rows; i++) {
      auto row = values.subspan(i * cols, cols);
      const double lse = RowLogSumExp(row);
      auto target = static_cast<size_t>(std::lround(target_values[i]));
      for (size_t j = 0; j < cols; j++) {
        const double prob = std::exp(static_cast<double>(row[j]) - lse);
        const double onehot = j == target ? 1.0 : 0.0;
        gradient[(i * cols) + j] +=
            static_cast<scalar_t>((prob - onehot) * incoming[i]);
      }
    }
  };
}

}  // namespace

void RegisterEmbeddingOps() {
  OpRegistry &registry = OpRegistry::Instance();
  registry.Register(OpId::kEmbeddingLookup, Device::CPU, EmbeddingLookup);
  registry.RegisterBackward(OpId::kEmbeddingLookup, Device::CPU,
                            EmbeddingLookupBackward);
  registry.Register(OpId::kGatherPerRow, Device::CPU, GatherPerRow);
  registry.RegisterBackward(OpId::kGatherPerRow, Device::CPU,
                            GatherPerRowBackward);
  registry.Register(OpId::kCrossEntropy, Device::CPU, CrossEntropy);
  registry.RegisterBackward(OpId::kCrossEntropy, Device::CPU,
                            CrossEntropyBackward);
}

}  // namespace micrograd::ops::cpu
