#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "micrograd/Device.h"
#include "micrograd/NN.h"
#include "micrograd/Scalar.h"
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

struct MatmulCase {
  size_t n;
  int iterations;
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
#ifdef MICROGRAD_METAL_ENABLED
  devices.push_back(Device::Metal);
#endif
#ifdef MICROGRAD_CUDA_ENABLED
  devices.push_back(Device::CUDA);
#endif
  return devices;
}

std::string DeviceName(Device device) {
  switch (device) {
    case Device::CPU:
      return "cpu";
    case Device::Metal:
      return "metal";
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
  return std::make_shared<Tensor>(std::move(shape), std::move(values));
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

Sequential BuildClassifier(Device device) {
  auto l1 = std::make_shared<Linear>(kFeatures, kHidden);
  auto l2 = std::make_shared<Linear>(kHidden, kClasses);
  Sequential net({l1, std::make_shared<ReLU>(), l2});
  for (auto &parameter : net.parameters()) {
    parameter->to(device);
  }
  return net;
}

Row BenchMatmul(Device device, const MatmulCase &matmul_case) {
  auto lhs = Ramp({matmul_case.n, matmul_case.n}, -1.0f, 0.0003f);
  auto rhs = Ramp({matmul_case.n, matmul_case.n}, 0.5f, -0.0002f);
  lhs->to(device);
  rhs->to(device);

  lhs->matmul(rhs);

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < matmul_case.iterations; i++) {
    lhs->matmul(rhs);
  }
  const double seconds_per_iter = SecondsSince(start) / matmul_case.iterations;

  const double flops = 2.0 * static_cast<double>(matmul_case.n) *
                       static_cast<double>(matmul_case.n) *
                       static_cast<double>(matmul_case.n);
  const double gflops = flops / seconds_per_iter / 1e9;

  return {DeviceName(device), "matmul", "n=" + std::to_string(matmul_case.n),
          seconds_per_iter,   gflops,   "gflops"};
}

Row BenchTrainingStep(Device device, size_t batch_size, int iterations) {
  auto net = BuildClassifier(device);
  AdamW optimizer(net.parameters(), 1e-3f);

  auto input = Ramp({batch_size, kFeatures}, -1.0f, 0.0001f);
  input->to(device);
  auto targets = RampTargets(batch_size);

  auto step_once = [&]() {
    auto out = net.forward(input);
    auto loss = micrograd::cross_entropy(out, targets);
    optimizer.zero_grad();
    loss->backward();
    optimizer.step();
  };

  step_once();

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    step_once();
  }
  const double seconds_per_iter = SecondsSince(start) / iterations;
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
  AdamW optimizer(net.parameters(), 1e-3f);

  auto input = Ramp({batch_size, kFeatures}, -1.0f, 0.0001f);
  input->to(device);
  auto targets = RampTargets(batch_size);

  auto step_once = [&]() {
    auto out = net.forward(input);
    auto loss = micrograd::cross_entropy(out, targets);
    optimizer.zero_grad();
    loss->backward();
    optimizer.step();
  };

  step_once();

  const size_t batches = total_samples / batch_size;
  const auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < batches; i++) {
    step_once();
  }
  const double seconds = SecondsSince(start);
  const double samples_per_second =
      static_cast<double>(batches * batch_size) / seconds;

  return {DeviceName(device),
          "epoch",
          "samples=" + std::to_string(batches * batch_size),
          seconds,
          samples_per_second,
          "samples/s"};
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
      {128, 20}, {256, 10}, {512, 5}, {1024, 3}};

  std::vector<Row> rows;
  for (auto device : CompiledDevices()) {
    for (const auto &matmul_case : matmul_cases) {
      rows.push_back(BenchMatmul(device, matmul_case));
    }
    rows.push_back(BenchTrainingStep(device, 64, 20));
    rows.push_back(BenchEpoch(device, 64, 6400));
  }

  PrintMarkdownTable(rows);
  return 0;
}
