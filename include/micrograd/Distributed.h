#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "micrograd/Device.h"
#include "micrograd/Scalar.h"

namespace micrograd {
class Storage;
class Tensor;
}  // namespace micrograd

namespace micrograd::distributed {

struct Info {
  int rank = 0;
  int world_size = 1;
  int local_rank = 0;
  std::string master_addr = "127.0.0.1";
  int master_port = 29500;
};

Info FromEnvironment();

void Initialize(const Info &info);
void Shutdown();

bool Active();
int Rank();
int WorldSize();
int LocalRank();
bool IsMain();

std::string DeviceDescription(Device device);

void AllReduceSum(Storage &storage);
void AllReduceSum(scalar_t *data, size_t count, Device device);
void AllReduceSum(std::span<Storage *const> storages);
void Broadcast(Storage &storage);
double AllReduceSum(double value);
void Barrier();

class GradientReducer {
 public:
  static constexpr size_t kDefaultBucketBytes = size_t{25} << 20;

  GradientReducer(std::vector<std::shared_ptr<Tensor>> parameters,
                  size_t bucket_bytes = kDefaultBucketBytes,
                  bool bf16 = false);
  ~GradientReducer();

  GradientReducer(const GradientReducer &) = delete;
  GradientReducer &operator=(const GradientReducer &) = delete;

  void prepare(bool reduce_this_backward);
  void finish();

  bool overlaps() const;
  size_t bucket_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace micrograd::distributed
