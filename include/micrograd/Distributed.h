#pragma once

#include <cstddef>
#include <string>

#include "micrograd/Device.h"
#include "micrograd/Scalar.h"

namespace micrograd {
class Storage;
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
void Barrier();

}  // namespace micrograd::distributed
