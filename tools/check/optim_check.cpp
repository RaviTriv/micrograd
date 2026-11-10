#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <random>
#include <vector>

#include "micrograd/Tensor.h"
#include "micrograd/ops/Optimizer.h"

#ifdef MICROGRAD_CUDA_ENABLED
#include "micrograd/backends/cuda/CudaContext.h"
#endif

using namespace micrograd;

namespace {

struct Case {
  size_t rows;
  size_t cols;
};

std::vector<scalar_t> RandomValues(size_t n, uint64_t seed, scalar_t scale) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<scalar_t> dist(0.0f, scale);
  std::vector<scalar_t> values(n);
  for (auto &v : values) {
    v = dist(rng);
  }
  return values;
}

std::shared_ptr<Tensor> MakeParam(const Case &c, Device device) {
  const size_t n = c.rows * c.cols;
  auto p = Tensor::from_values({c.rows, c.cols}, RandomValues(n, 1, 0.02f));
  p->set_requires_grad(true);
  auto grad = RandomValues(n, 2, 1.0f);
  std::ranges::copy(grad, p->grad().begin());
  p->to(device);
  return p;
}

std::vector<scalar_t> RunSteps(const Case &c, Device device, int steps) {
  auto p = MakeParam(c, device);
  auto momentum = Tensor::zeros({c.rows, c.cols}, device);
  for (int i = 0; i < steps; i++) {
    ops::MuonStep(*p, *momentum, 0.02f, 0.95f, 0.01f, true, 5, 1.0f);
  }
  synchronize(device);
  p->to(Device::CPU);
  auto data = p->data();
  return {data.begin(), data.end()};
}

scalar_t UpdateRms(const Case &c, const std::vector<scalar_t> &after) {
  auto before = RandomValues(c.rows * c.cols, 1, 0.02f);
  double sum = 0.0;
  for (size_t i = 0; i < after.size(); i++) {
    const double d = (before[i] * (1.0f - (0.02f * 0.01f))) - after[i];
    sum += d * d;
  }
  return static_cast<scalar_t>(std::sqrt(sum / static_cast<double>(after.size())) /
                               0.02);
}

}  // namespace

int main() {
  try {
    const std::vector<Case> cases = {
        {64, 64}, {48, 200}, {200, 48}, {256, 1024}, {1024, 256}};
    int failures = 0;
    for (const Case &c : cases) {
      const auto cpu = RunSteps(c, Device::CPU, 1);
      const scalar_t rms = UpdateRms(c, cpu);
      const scalar_t expected =
          1.0f / std::sqrt(static_cast<scalar_t>(std::max(c.rows, c.cols)));
      const bool rms_ok = rms > 0.5f * expected && rms < 1.5f * expected;
      std::printf("muon %zux%zu cpu: update rms %.5f (orthogonal ~%.5f) %s\n",
                  c.rows, c.cols, rms, expected, rms_ok ? "ok" : "WRONG");
      failures += rms_ok ? 0 : 1;
#ifdef MICROGRAD_CUDA_ENABLED
      if (CudaContext::instance().initialize()) {
        const auto cpu3 = RunSteps(c, Device::CPU, 3);
        const auto gpu3 = RunSteps(c, Device::CUDA, 3);
        scalar_t worst = 0.0f;
        for (size_t i = 0; i < cpu3.size(); i++) {
          worst = std::max(worst, std::abs(cpu3[i] - gpu3[i]));
        }
        const bool ok = worst < 2e-5f;
        std::printf("muon %zux%zu cuda vs cpu, 3 steps: max abs diff %.3g %s\n",
                    c.rows, c.cols, worst, ok ? "ok" : "WRONG");
        failures += ok ? 0 : 1;
      }
#endif
    }
    std::printf("%s\n", failures == 0 ? "optim_check passed" : "optim_check FAILED");
    return failures == 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::printf("optim_check raised: %s\n", error.what());
    return 1;
  }
}
