#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "micrograd/Device.h"
#include "micrograd/NN.h"
#include "micrograd/Scalar.h"
#include "micrograd/Storage.h"
#include "micrograd/Tensor.h"

using micrograd::AdamW;
using micrograd::Device;
using micrograd::Linear;
using micrograd::ReLU;
using micrograd::scalar_t;
using micrograd::Sequential;
using micrograd::Tensor;

namespace {

constexpr size_t kFeatures = 196;
constexpr size_t kClasses = 10;
constexpr size_t kHidden = 100;
constexpr size_t kActivationRow = 1024;
constexpr int kRepeats = 5;
constexpr double kBytesPerElementForward = 2.0 * sizeof(scalar_t);

struct MatmulCase {
  std::string name;
  std::vector<size_t> lhs_shape;
  std::vector<size_t> rhs_shape;
  bool transpose_rhs;
  int iterations;
};

struct ActivationSize {
  size_t elements;
  int iterations;
};

using Activation =
    std::function<std::shared_ptr<Tensor>(const std::shared_ptr<Tensor> &)>;

struct ActivationCase {
  std::string name;
  Activation apply;
};

struct Row {
  std::string device;
  std::string benchmark;
  std::string detail;
  double seconds_per_iter;
  double metric;
  std::string metric_name;
};

std::vector<Device> CompiledDevices() {
  std::vector<Device> devices{Device::CPU};
#ifdef MICROGRAD_CUDA_ENABLED
  devices.push_back(Device::CUDA);
#endif
  return devices;
}

std::string DeviceName(Device device) {
  switch (device) {
    case Device::CPU:
      return "cpu";
    case Device::CUDA:
      return "cuda";
  }
  return "unknown";
}

std::shared_ptr<Tensor> Ramp(std::vector<size_t> shape, scalar_t start,
                             scalar_t step) {
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

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

std::vector<size_t> RampTargets(size_t count) {
  std::vector<size_t> targets(count);
  for (size_t i = 0; i < count; i++) {
    targets[i] = i % kClasses;
  }
  return targets;
}

std::shared_ptr<Sequential> BuildClassifier(Device device) {
  auto l1 = std::make_shared<Linear>(kFeatures, kHidden);
  auto l2 = std::make_shared<Linear>(kHidden, kClasses);
  auto net = std::make_shared<Sequential>(
      std::vector<std::shared_ptr<micrograd::nn::Module>>{
          l1, std::make_shared<ReLU>(), l2});
  for (auto &parameter : net->parameters()) {
    parameter->to(device);
  }
  return net;
}

double MedianSecondsPerIter(Device device, int iterations,
                            const std::function<void()> &run) {
  run();
  micrograd::synchronize(device);

  std::vector<double> samples;
  samples.reserve(kRepeats);
  for (int repeat = 0; repeat < kRepeats; repeat++) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
      run();
    }
    micrograd::synchronize(device);
    samples.push_back(SecondsSince(start) / iterations);
  }
  std::ranges::sort(samples);
  return samples[samples.size() / 2];
}

double MatmulFlops(const MatmulCase &matmul_case) {
  const auto &lhs = matmul_case.lhs_shape;
  const auto &rhs = matmul_case.rhs_shape;
  double flops = 2.0;
  for (auto dim : lhs) {
    flops *= static_cast<double>(dim);
  }
  return flops * static_cast<double>(matmul_case.transpose_rhs ? rhs[0]
                                                               : rhs.back());
}

std::shared_ptr<Tensor> RunMatmul(const MatmulCase &matmul_case,
                                  const std::shared_ptr<Tensor> &lhs,
                                  const std::shared_ptr<Tensor> &rhs) {
  return matmul_case.transpose_rhs ? lhs->matmul_nt(rhs) : lhs->matmul(rhs);
}

Row BenchMatmul(Device device, const MatmulCase &matmul_case, bool backward) {
  auto lhs = Ramp(matmul_case.lhs_shape, -1.0f, 0.0003f);
  auto rhs = Ramp(matmul_case.rhs_shape, 0.5f, -0.0002f);
  lhs->to(device);
  rhs->to(device);
  lhs->set_requires_grad(backward);
  rhs->set_requires_grad(backward);

  const auto probe = RunMatmul(matmul_case, lhs, rhs);
  const auto seed = Tensor::from_values(
      probe->shape(), std::vector<scalar_t>(probe->size(), 1.0f));
  seed->to(device);

  const double seconds_per_iter =
      MedianSecondsPerIter(device, matmul_case.iterations, [&]() {
        auto out = RunMatmul(matmul_case, lhs, rhs);
        if (backward) {
          out->backward(*seed);
          lhs->zero_grad();
          rhs->zero_grad();
        }
      });

  const double flops = MatmulFlops(matmul_case) * (backward ? 3.0 : 1.0);
  const double gflops = flops / seconds_per_iter / 1e9;

  return {DeviceName(device),
          "matmul",
          matmul_case.name + (backward ? " fwd+bwd" : " fwd"),
          seconds_per_iter,
          gflops,
          "gflops"};
}

Row BenchTrainingStep(Device device, size_t batch_size, int iterations) {
  auto net = BuildClassifier(device);
  AdamW optimizer(net->parameters(), 1e-3f);

  auto input = Ramp({batch_size, kFeatures}, -1.0f, 0.0001f);
  input->to(device);
  auto targets = RampTargets(batch_size);

  auto step_once = [&]() {
    auto out = net->forward(input);
    auto loss = micrograd::cross_entropy(out, targets);
    optimizer.zero_grad();
    loss->backward();
    optimizer.step();
  };

  const double seconds_per_iter =
      MedianSecondsPerIter(device, iterations, step_once);
  const double samples_per_second =
      static_cast<double>(batch_size) / seconds_per_iter;

  return {DeviceName(device),
          "train_step",
          "batch=" + std::to_string(batch_size),
          seconds_per_iter,
          samples_per_second,
          "samples/s"};
}

Row BenchEpoch(Device device, size_t batch_size, size_t total_samples) {
  auto net = BuildClassifier(device);
  AdamW optimizer(net->parameters(), 1e-3f);

  auto input = Ramp({batch_size, kFeatures}, -1.0f, 0.0001f);
  input->to(device);
  auto targets = RampTargets(batch_size);

  auto step_once = [&]() {
    auto out = net->forward(input);
    auto loss = micrograd::cross_entropy(out, targets);
    optimizer.zero_grad();
    loss->backward();
    optimizer.step();
  };

  const size_t batches = total_samples / batch_size;
  const double seconds =
      MedianSecondsPerIter(device, static_cast<int>(batches), step_once) *
      static_cast<double>(batches);
  const double samples_per_second =
      static_cast<double>(batches * batch_size) / seconds;

  return {DeviceName(device),
          "epoch",
          "samples=" + std::to_string(batches * batch_size),
          seconds,
          samples_per_second,
          "samples/s"};
}

std::vector<ActivationCase> ActivationCases() {
  return {
      {"relu", [](const auto &x) { return x->relu(); }},
      {"sigmoid", [](const auto &x) { return x->sigmoid(); }},
      {"tanh", [](const auto &x) { return x->tanh(); }},
      {"gelu", [](const auto &x) { return micrograd::gelu(x); }},
      {"softmax", [](const auto &x) { return x->softmax(-1); }},
  };
}

std::shared_ptr<Tensor> StandardNormal(size_t elements) {
  std::mt19937 rng(7);
  std::normal_distribution<scalar_t> normal;
  std::vector<scalar_t> values(elements);
  for (auto &value : values) {
    value = normal(rng);
  }
  return Tensor::from_values({elements / kActivationRow, kActivationRow},
                             std::move(values));
}

std::string ActivationDetail(const ActivationCase &activation,
                             const std::string &pass, size_t elements) {
  return activation.name + " " + pass + " n=" + std::to_string(elements);
}

Row BenchActivationForward(Device device, const ActivationCase &activation,
                           const ActivationSize &size) {
  auto input = StandardNormal(size.elements);
  input->to(device);

  const double seconds_per_iter = MedianSecondsPerIter(
      device, size.iterations, [&]() { activation.apply(input); });
  const double gb_per_second = kBytesPerElementForward *
                               static_cast<double>(size.elements) /
                               seconds_per_iter / 1e9;

  return {DeviceName(device),
          "activation",
          ActivationDetail(activation, "fwd", size.elements),
          seconds_per_iter,
          gb_per_second,
          "gb/s"};
}

Row BenchActivationBackward(Device device, const ActivationCase &activation,
                            const ActivationSize &size) {
  auto input = StandardNormal(size.elements);
  input->to(device);
  input->set_requires_grad(true);
  const auto seed = Tensor::from_values(
      input->shape(), std::vector<scalar_t>(size.elements, 1.0f));

  const double seconds_per_iter =
      MedianSecondsPerIter(device, size.iterations, [&]() {
        activation.apply(input)->backward(*seed);
        input->zero_grad();
      });
  const double melem_per_second =
      static_cast<double>(size.elements) / seconds_per_iter / 1e6;

  return {DeviceName(device),
          "activation",
          ActivationDetail(activation, "fwd+bwd", size.elements),
          seconds_per_iter,
          melem_per_second,
          "melem/s"};
}

void PrintMarkdownTable(const std::vector<Row> &rows) {
  std::cout << "| device | benchmark | detail | seconds | metric |\n";
  std::cout << "|---|---|---|---|---|\n";
  std::cout << std::fixed;
  for (const auto &row : rows) {
    std::cout << "| " << row.device << " | " << row.benchmark << " | "
              << row.detail << " | " << std::setprecision(6)
              << row.seconds_per_iter << " | " << std::setprecision(3)
              << row.metric << " " << row.metric_name << " |\n";
  }
}

}  // namespace

int main() {
  const std::vector<MatmulCase> matmul_cases{
      {"8x8", {8, 8}, {8, 8}, false, 20000},
      {"n=128", {128, 128}, {128, 128}, false, 20},
      {"n=256", {256, 256}, {256, 256}, false, 10},
      {"n=512", {512, 512}, {512, 512}, false, 5},
      {"n=1024", {1024, 1024}, {1024, 1024}, false, 2},
      {"batch=64 64x64x64", {64, 64, 64}, {64, 64, 64}, false, 50},
      {"nt 256x768x2048", {256, 768}, {2048, 768}, true, 3},
  };

  std::vector<Row> rows;
  for (auto device : CompiledDevices()) {
    for (const auto &matmul_case : matmul_cases) {
      rows.push_back(BenchMatmul(device, matmul_case, false));
      rows.push_back(BenchMatmul(device, matmul_case, true));
    }
    rows.push_back(BenchTrainingStep(device, 64, 20));
    rows.push_back(BenchEpoch(device, 64, 6400));
  }

  const std::vector<ActivationSize> activation_sizes{{1 << 16, 200},
                                                     {1 << 22, 10}};
  for (auto device : CompiledDevices()) {
    for (const auto &size : activation_sizes) {
      for (const auto &activation : ActivationCases()) {
        rows.push_back(BenchActivationForward(device, activation, size));
        rows.push_back(BenchActivationBackward(device, activation, size));
      }
    }
  }

  PrintMarkdownTable(rows);
  return 0;
}
