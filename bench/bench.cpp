#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"

using micrograd::scalar_t;
using micrograd::Tensor;

namespace {

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

void BenchMatmul(size_t n, int iterations) {
  auto lhs = Ramp({n, n}, -1.0f, 0.0003f);
  auto rhs = Ramp({n, n}, 0.5f, -0.0002f);

  lhs->matmul(rhs);

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    lhs->matmul(rhs);
  }
  const auto end = std::chrono::steady_clock::now();

  const std::chrono::duration<double> elapsed = end - start;
  const double seconds_per_iter = elapsed.count() / iterations;
  const double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n) *
                       static_cast<double>(n);
  const double gflops = flops / seconds_per_iter / 1e9;

  std::cout << "matmul n=" << n << " iterations=" << iterations
            << " seconds_per_iter=" << seconds_per_iter << " gflops=" << gflops
            << "\n";
}

}  // namespace

int main() {
  BenchMatmul(512, 3);
  return 0;
}
