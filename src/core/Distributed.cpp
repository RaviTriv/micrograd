#include "micrograd/Distributed.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "micrograd/Storage.h"

#ifdef MICROGRAD_CUDA_ENABLED
#include <cuda_runtime.h>

#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"
#endif

#ifdef MICROGRAD_NCCL_ENABLED
#include <nccl.h>
#endif

namespace micrograd::distributed {
namespace {

constexpr auto kConnectTimeout = std::chrono::seconds(60);

struct State {
  bool active = false;
  Info info;
  int listen_fd = -1;
  std::vector<int> peer_fds;
#ifdef MICROGRAD_NCCL_ENABLED
  ncclComm_t comm = nullptr;
#endif
};

State &GetState() {
  static State state;
  return state;
}

int EnvInt(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0 || parsed > 65535) {
    throw std::invalid_argument(std::string("Invalid value for ") + name +
                                ": " + value);
  }
  return static_cast<int>(parsed);
}

std::string EnvString(const char *name, const std::string &fallback) {
  const char *value = std::getenv(name);
  return value == nullptr || *value == '\0' ? fallback : std::string(value);
}

[[noreturn]] void ThrowErrno(const std::string &what) {
  throw std::runtime_error(what + ": " + std::strerror(errno));
}

void DisableSigpipe([[maybe_unused]] int fd) {
#ifdef SO_NOSIGPIPE
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
}

constexpr int kSendFlags =
#ifdef MSG_NOSIGNAL
    MSG_NOSIGNAL;
#else
    0;
#endif

void SendAll(int fd, const void *data, size_t bytes) {
  const auto *cursor = static_cast<const char *>(data);
  while (bytes > 0) {
    const ssize_t sent = send(fd, cursor, bytes, kSendFlags);
    if (sent < 0) {
      if (errno == EINTR) {
        continue;
      }
      ThrowErrno("distributed: send failed");
    }
    cursor += sent;
    bytes -= static_cast<size_t>(sent);
  }
}

void RecvAll(int fd, void *data, size_t bytes) {
  auto *cursor = static_cast<char *>(data);
  while (bytes > 0) {
    const ssize_t received = recv(fd, cursor, bytes, 0);
    if (received < 0) {
      if (errno == EINTR) {
        continue;
      }
      ThrowErrno("distributed: recv failed");
    }
    if (received == 0) {
      throw std::runtime_error("distributed: a peer closed the connection");
    }
    cursor += received;
    bytes -= static_cast<size_t>(received);
  }
}

void SetNoDelay(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  DisableSigpipe(fd);
}

int Listen(int port, int backlog) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    ThrowErrno("distributed: socket failed");
  }
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(static_cast<uint16_t>(port));
  if (bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
    close(fd);
    ThrowErrno("distributed: bind to port " + std::to_string(port) + " failed");
  }
  if (listen(fd, backlog) != 0) {
    close(fd);
    ThrowErrno("distributed: listen failed");
  }
  timeval timeout{};
  timeout.tv_sec =
      static_cast<decltype(timeout.tv_sec)>(kConnectTimeout.count());
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  return fd;
}

int Connect(const std::string &host, int port) {
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(port));
  if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
    throw std::invalid_argument(
        "distributed: MASTER_ADDR must be an IPv4 "
        "address, got " +
        host);
  }

  const auto deadline = std::chrono::steady_clock::now() + kConnectTimeout;
  while (true) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      ThrowErrno("distributed: socket failed");
    }
    if (connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) ==
        0) {
      return fd;
    }
    close(fd);
    if (std::chrono::steady_clock::now() > deadline) {
      ThrowErrno("distributed: could not reach rank 0 at " + host + ":" +
                 std::to_string(port));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void ConnectControlChannel(State &state) {
  const Info &info = state.info;
  if (info.rank == 0) {
    state.listen_fd = Listen(info.master_port, info.world_size);
    state.peer_fds.assign(static_cast<size_t>(info.world_size), -1);
    for (int i = 1; i < info.world_size; i++) {
      const int fd = accept(state.listen_fd, nullptr, nullptr);
      if (fd < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          throw std::runtime_error(
              "distributed: timed out waiting for all ranks to connect");
        }
        ThrowErrno("distributed: accept failed");
      }
      SetNoDelay(fd);
      int32_t peer_rank = -1;
      RecvAll(fd, &peer_rank, sizeof(peer_rank));
      if (peer_rank <= 0 || peer_rank >= info.world_size ||
          state.peer_fds[static_cast<size_t>(peer_rank)] != -1) {
        close(fd);
        throw std::runtime_error("distributed: unexpected rank " +
                                 std::to_string(peer_rank) + " connected");
      }
      state.peer_fds[static_cast<size_t>(peer_rank)] = fd;
    }
    return;
  }

  const int fd = Connect(info.master_addr, info.master_port);
  SetNoDelay(fd);
  const auto rank = static_cast<int32_t>(info.rank);
  SendAll(fd, &rank, sizeof(rank));
  state.peer_fds = {fd};
}

void HostAllReduceSum(State &state, scalar_t *data, size_t count) {
  const size_t bytes = count * sizeof(scalar_t);
  if (state.info.rank == 0) {
    std::vector<scalar_t> incoming(count);
    for (int peer = 1; peer < state.info.world_size; peer++) {
      RecvAll(state.peer_fds[static_cast<size_t>(peer)], incoming.data(),
              bytes);
      for (size_t i = 0; i < count; i++) {
        data[i] += incoming[i];
      }
    }
    for (int peer = 1; peer < state.info.world_size; peer++) {
      SendAll(state.peer_fds[static_cast<size_t>(peer)], data, bytes);
    }
    return;
  }
  SendAll(state.peer_fds[0], data, bytes);
  RecvAll(state.peer_fds[0], data, bytes);
}

[[maybe_unused]] void BroadcastBytes(State &state, void *data, size_t bytes) {
  if (state.info.rank == 0) {
    for (int peer = 1; peer < state.info.world_size; peer++) {
      SendAll(state.peer_fds[static_cast<size_t>(peer)], data, bytes);
    }
    return;
  }
  RecvAll(state.peer_fds[0], data, bytes);
}

#ifdef MICROGRAD_NCCL_ENABLED
void CheckNccl(ncclResult_t result, const char *what) {
  if (result != ncclSuccess) {
    throw std::runtime_error(std::string("NCCL error in ") + what + ": " +
                             ncclGetErrorString(result));
  }
}
#endif

int CudaDeviceCount() {
#ifdef MICROGRAD_CUDA_ENABLED
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return count;
#else
  return 0;
#endif
}

}  // namespace

Info FromEnvironment() {
  Info info;
  info.rank = EnvInt("RANK", 0);
  info.world_size = EnvInt("WORLD_SIZE", 1);
  info.local_rank = EnvInt("LOCAL_RANK", info.rank);
  info.master_addr = EnvString("MASTER_ADDR", info.master_addr);
  info.master_port = EnvInt("MASTER_PORT", info.master_port);
  if (info.world_size < 1 || info.rank >= info.world_size) {
    throw std::invalid_argument("distributed: RANK must be below WORLD_SIZE");
  }
  return info;
}

void Initialize(const Info &info) {
  State &state = GetState();
  if (state.active) {
    throw std::runtime_error("distributed: already initialized");
  }
  state.info = info;

  [[maybe_unused]] const int gpus = CudaDeviceCount();
#ifdef MICROGRAD_CUDA_ENABLED
  if (gpus > 0) {
    CudaContext::instance().setDevice(info.local_rank % gpus);
  }
#endif

  if (info.world_size > 1) {
    ConnectControlChannel(state);
#ifdef MICROGRAD_NCCL_ENABLED
    if (gpus > 0) {
      ncclUniqueId id{};
      if (info.rank == 0) {
        CheckNccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
      }
      BroadcastBytes(state, &id, sizeof(id));
      if (!CudaContext::instance().initialize()) {
        throw std::runtime_error("distributed: CUDA device is unavailable");
      }
      CheckNccl(ncclCommInitRank(&state.comm, info.world_size, id, info.rank),
                "ncclCommInitRank");
    }
#endif
  }
  state.active = true;
}

void Shutdown() {
  State &state = GetState();
#ifdef MICROGRAD_NCCL_ENABLED
  if (state.comm != nullptr) {
    ncclCommDestroy(state.comm);
    state.comm = nullptr;
  }
#endif
  for (int fd : state.peer_fds) {
    if (fd >= 0) {
      close(fd);
    }
  }
  state.peer_fds.clear();
  if (state.listen_fd >= 0) {
    close(state.listen_fd);
    state.listen_fd = -1;
  }
  state.active = false;
  state.info = Info{};
}

bool Active() { return GetState().active; }
int Rank() { return GetState().info.rank; }
int WorldSize() { return GetState().info.world_size; }
int LocalRank() { return GetState().info.local_rank; }
bool IsMain() { return Rank() == 0; }

std::string DeviceDescription(Device device) {
  if (device == Device::CPU) {
    return "cpu";
  }
#ifdef MICROGRAD_CUDA_ENABLED
  const int index = CudaContext::instance().device();
  cudaDeviceProp props{};
  MICROGRAD_CUDA_CHECK(cudaGetDeviceProperties(&props, index));
  char bus[32];
  std::snprintf(bus, sizeof(bus), "%04x:%02x:%02x", props.pciDomainID,
                props.pciBusID, props.pciDeviceID);
  return "cuda:" + std::to_string(index) + " " + props.name + " [" + bus + "]";
#else
  return "cuda (not compiled in)";
#endif
}

void AllReduceSum(scalar_t *data, size_t count, Device device) {
  State &state = GetState();
  if (!state.active || state.info.world_size == 1 || count == 0) {
    return;
  }
  if (device == Device::CUDA) {
#ifdef MICROGRAD_NCCL_ENABLED
    if (state.comm == nullptr) {
      throw std::runtime_error("distributed: no NCCL communicator");
    }
    CheckNccl(ncclAllReduce(data, data, count, ncclFloat, ncclSum, state.comm,
                            CudaContext::instance().stream()),
              "ncclAllReduce");
    return;
#else
    throw std::runtime_error(
        "distributed: CUDA all-reduce needs a build with -DENABLE_NCCL=ON");
#endif
  }
  HostAllReduceSum(state, data, count);
}

void AllReduceSum(Storage &storage) {
  if (storage.dtype() != DType::kFloat32) {
    throw std::invalid_argument(
        "distributed: all-reduce supports float32 only");
  }
  const size_t count = storage.bytes() / sizeof(scalar_t);
  auto *data = static_cast<scalar_t *>(storage.device() == Device::CUDA
                                          ? storage.device_pointer()
                                          : storage.host_pointer());
  AllReduceSum(data, count, storage.device());
}

void Barrier() {
  State &state = GetState();
  if (!state.active || state.info.world_size == 1) {
    return;
  }
  scalar_t token = 0.0f;
  HostAllReduceSum(state, &token, 1);
}

}  // namespace micrograd::distributed
