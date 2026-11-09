#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "micrograd/Distributed.h"
#include "micrograd/Storage.h"
#include "micrograd/Tensor.h"

using namespace micrograd;

namespace {

Device DeviceFromEnv() {
  const char *value = std::getenv("MICROGRAD_DEVICE");
  return value != nullptr && std::string(value) == "cuda" ? Device::CUDA
                                                          : Device::CPU;
}

size_t CheckSum(Device device, int rank, int world) {
  constexpr size_t kCount = size_t{1} << 20;
  std::vector<scalar_t> values(kCount);
  for (size_t i = 0; i < kCount; i++) {
    values[i] = static_cast<scalar_t>(rank + 1) + static_cast<scalar_t>(i % 7);
  }
  auto tensor = Tensor::from_values({kCount}, std::move(values));
  tensor->to(device);
  distributed::AllReduceSum(tensor->data_storage());
  tensor->to(Device::CPU);

  const auto data = tensor->data();
  const auto rank_sum = static_cast<scalar_t>(world * (world + 1) / 2);
  size_t wrong = 0;
  for (size_t i = 0; i < kCount; i++) {
    const scalar_t expected =
        rank_sum + static_cast<scalar_t>(world) * static_cast<scalar_t>(i % 7);
    if (data[i] != expected) {
      wrong++;
    }
  }
  return wrong;
}

double TimeAllReduceMs(Device device, size_t count, int iterations) {
  auto buffer = Tensor::zeros({count}, device);
  distributed::AllReduceSum(buffer->data_storage());
  synchronize(device);
  distributed::Barrier();

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    distributed::AllReduceSum(buffer->data_storage());
  }
  synchronize(device);
  const std::chrono::duration<double, std::milli> elapsed =
      std::chrono::steady_clock::now() - start;
  return elapsed.count() / iterations;
}

}  // namespace

int main() {
  try {
    distributed::Initialize(distributed::FromEnvironment());
    const Device device = DeviceFromEnv();
    const int rank = distributed::Rank();
    const int world = distributed::WorldSize();
    std::printf("rank %d/%d local %d on %s\n", rank, world,
                distributed::LocalRank(),
                distributed::DeviceDescription(device).c_str());

    const size_t wrong = CheckSum(device, rank, world);
    distributed::Barrier();
    std::printf("rank %d: all-reduce sum %s\n", rank,
                wrong == 0 ? "ok" : "WRONG");

    constexpr size_t kTimedCount = size_t{16} << 20;
    const double ms = TimeAllReduceMs(device, kTimedCount, 10);
    if (distributed::IsMain()) {
      std::printf("all-reduce of %zu MB: %.2f ms\n",
                  (kTimedCount * sizeof(scalar_t)) >> 20, ms);
    }

    distributed::Barrier();
    distributed::Shutdown();
    return wrong == 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::printf("dist_check raised: %s\n", error.what());
    return 1;
  }
}
