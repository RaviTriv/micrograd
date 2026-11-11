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
#include <utility>
#include <vector>

#include "micrograd/Storage.h"
#include "micrograd/Tensor.h"

#ifdef MICROGRAD_CUDA_ENABLED
#include <cuda_runtime.h>

#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"
#include "micrograd/backends/cuda/ops/Convert.h"
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

template <typename T>
void HostAllReduceSum(State &state, T *data, size_t count) {
  const size_t bytes = count * sizeof(T);
  if (state.info.rank == 0) {
    std::vector<T> incoming(count);
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

void BroadcastBytes(State &state, void *data, size_t bytes) {
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

namespace {

scalar_t *StoragePointer(Storage &storage) {
  if (storage.dtype() != DType::kFloat32) {
    throw std::invalid_argument(
        "distributed: collectives support float32 only");
  }
  return static_cast<scalar_t *>(storage.device() == Device::CUDA
                                     ? storage.device_pointer()
                                     : storage.host_pointer());
}

size_t StorageCount(const Storage &storage) {
  return storage.bytes() / sizeof(scalar_t);
}

}  // namespace

void AllReduceSum(Storage &storage) {
  AllReduceSum(StoragePointer(storage), StorageCount(storage),
               storage.device());
}

void AllReduceSum(std::span<Storage *const> storages) {
  State &state = GetState();
  if (!state.active || state.info.world_size == 1) {
    return;
  }
#ifdef MICROGRAD_NCCL_ENABLED
  bool all_cuda = !storages.empty();
  for (Storage *storage : storages) {
    all_cuda = all_cuda && storage->device() == Device::CUDA;
  }
  if (all_cuda && state.comm != nullptr) {
    CheckNccl(ncclGroupStart(), "ncclGroupStart");
    for (Storage *storage : storages) {
      const size_t count = StorageCount(*storage);
      if (count == 0) {
        continue;
      }
      scalar_t *data = StoragePointer(*storage);
      CheckNccl(ncclAllReduce(data, data, count, ncclFloat, ncclSum, state.comm,
                              CudaContext::instance().stream()),
                "ncclAllReduce");
    }
    CheckNccl(ncclGroupEnd(), "ncclGroupEnd");
    return;
  }
#endif
  for (Storage *storage : storages) {
    AllReduceSum(*storage);
  }
}

void Broadcast(Storage &storage) {
  State &state = GetState();
  if (!state.active || state.info.world_size == 1 || storage.bytes() == 0) {
    return;
  }
  scalar_t *data = StoragePointer(storage);
  if (storage.device() == Device::CUDA) {
#ifdef MICROGRAD_NCCL_ENABLED
    if (state.comm == nullptr) {
      throw std::runtime_error("distributed: no NCCL communicator");
    }
    CheckNccl(ncclBroadcast(data, data, StorageCount(storage), ncclFloat, 0,
                            state.comm, CudaContext::instance().stream()),
              "ncclBroadcast");
    return;
#else
    throw std::runtime_error(
        "distributed: CUDA broadcast needs a build with -DENABLE_NCCL=ON");
#endif
  }
  BroadcastBytes(state, data, storage.bytes());
}

double AllReduceSum(double value) {
  State &state = GetState();
  if (!state.active || state.info.world_size == 1) {
    return value;
  }
  HostAllReduceSum(state, &value, 1);
  return value;
}

void Barrier() {
  State &state = GetState();
  if (!state.active || state.info.world_size == 1) {
    return;
  }
  scalar_t token = 0.0f;
  HostAllReduceSum(state, &token, 1);
}

struct GradientReducer::Impl {
  struct Bucket {
    std::vector<size_t> params;
    std::vector<size_t> offsets;
    size_t pending = 0;
    Storage packed;
#ifdef MICROGRAD_NCCL_ENABLED
    cudaEvent_t ready = nullptr;
#endif
  };

  std::vector<std::shared_ptr<Tensor>> parameters;
  std::vector<size_t> bucket_of;
  std::vector<bool> ready;
  std::vector<Bucket> buckets;
  size_t next_bucket = 0;
  bool overlap = false;
  bool bf16 = false;
  bool armed = false;
#ifdef MICROGRAD_NCCL_ENABLED
  cudaStream_t stream = nullptr;
  cudaEvent_t done = nullptr;
#endif

  void mark_ready(size_t index) {
    if (!armed) {
      return;
    }
    if (ready[index]) {
      throw std::logic_error(
          "distributed: a parameter gradient became ready twice in one "
          "backward pass");
    }
    ready[index] = true;
    if (--buckets[bucket_of[index]].pending == 0) {
      while (next_bucket < buckets.size() &&
             buckets[next_bucket].pending == 0) {
        launch(buckets[next_bucket++]);
      }
    }
  }

  void launch([[maybe_unused]] Bucket &bucket) {
#ifdef MICROGRAD_NCCL_ENABLED
    State &state = GetState();
    std::vector<std::pair<scalar_t *, size_t>> grads;
    grads.reserve(bucket.params.size());
    for (size_t index : bucket.params) {
      Storage &grad = parameters[index]->grad_storage();
      grads.emplace_back(StoragePointer(grad), StorageCount(grad));
    }
    MICROGRAD_CUDA_CHECK(
        cudaEventRecord(bucket.ready, CudaContext::instance().stream()));
    MICROGRAD_CUDA_CHECK(cudaStreamWaitEvent(stream, bucket.ready, 0));
    if (bf16) {
      auto *packed = static_cast<char *>(bucket.packed.device_pointer());
      constexpr size_t kBf16Bytes = 2;
      for (size_t i = 0; i < grads.size(); i++) {
        cuda::ops::FloatToBf16(grads[i].first,
                               packed + (bucket.offsets[i] * kBf16Bytes),
                               grads[i].second, stream);
      }
      CheckNccl(ncclAllReduce(packed, packed, bucket.offsets.back(),
                              ncclBfloat16, ncclSum, state.comm, stream),
                "ncclAllReduce");
      for (size_t i = 0; i < grads.size(); i++) {
        cuda::ops::Bf16ToFloat(packed + (bucket.offsets[i] * kBf16Bytes),
                               grads[i].first, grads[i].second, stream);
      }
      return;
    }
    CheckNccl(ncclGroupStart(), "ncclGroupStart");
    for (const auto &[data, count] : grads) {
      CheckNccl(ncclAllReduce(data, data, count, ncclFloat, ncclSum, state.comm,
                              stream),
                "ncclAllReduce");
    }
    CheckNccl(ncclGroupEnd(), "ncclGroupEnd");
#endif
  }
};

GradientReducer::GradientReducer(
    std::vector<std::shared_ptr<Tensor>> parameters,
    [[maybe_unused]] size_t bucket_bytes, bool bf16)
    : impl_(std::make_unique<Impl>()) {
  Impl &impl = *impl_;
  impl.parameters = std::move(parameters);
  if (bf16 && bucket_bytes == 0) {
    throw std::invalid_argument(
        "distributed: bf16 gradient all-reduce needs gradient buckets");
  }
#ifdef MICROGRAD_NCCL_ENABLED
  State &state = GetState();
  bool overlap = bucket_bytes > 0 && state.active &&
                 state.info.world_size > 1 && state.comm != nullptr &&
                 !impl.parameters.empty();
  for (const auto &p : impl.parameters) {
    overlap = overlap && p->backend() == Device::CUDA &&
              p->dtype() == DType::kFloat32;
  }
  if (!overlap) {
    return;
  }
  impl.overlap = true;
  impl.bf16 = bf16;

  const size_t count = impl.parameters.size();
  impl.bucket_of.resize(count);
  impl.ready.assign(count, false);
  size_t bytes = 0;
  for (size_t i = count; i-- > 0;) {
    if (impl.buckets.empty() || bytes >= bucket_bytes) {
      impl.buckets.emplace_back();
      bytes = 0;
    }
    impl.buckets.back().params.push_back(i);
    impl.bucket_of[i] = impl.buckets.size() - 1;
    bytes += impl.parameters[i]->size() * sizeof(scalar_t);
  }

  int least = 0;
  int greatest = 0;
  MICROGRAD_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  MICROGRAD_CUDA_CHECK(cudaStreamCreateWithPriority(
      &impl.stream, cudaStreamNonBlocking, greatest));
  MICROGRAD_CUDA_CHECK(
      cudaEventCreateWithFlags(&impl.done, cudaEventDisableTiming));
  constexpr size_t kPackAlign = 8;
  for (Impl::Bucket &bucket : impl.buckets) {
    MICROGRAD_CUDA_CHECK(
        cudaEventCreateWithFlags(&bucket.ready, cudaEventDisableTiming));
    if (!impl.bf16) {
      continue;
    }
    size_t offset = 0;
    for (size_t index : bucket.params) {
      bucket.offsets.push_back(offset);
      offset += (impl.parameters[index]->size() + kPackAlign - 1) / kPackAlign *
                kPackAlign;
    }
    bucket.offsets.push_back(offset);
    bucket.packed = Storage(offset * 2, Device::CUDA, DType::kBFloat16);
    MICROGRAD_CUDA_CHECK(cudaMemsetAsync(bucket.packed.device_pointer(), 0,
                                         bucket.packed.bytes(),
                                         CudaContext::instance().stream()));
  }
  for (size_t i = 0; i < count; i++) {
    impl.parameters[i]->set_post_accumulate_grad_hook(
        [&impl, i](Tensor & /*param*/) { impl.mark_ready(i); });
  }
#endif
}

GradientReducer::~GradientReducer() {
  Impl &impl = *impl_;
  if (!impl.overlap) {
    return;
  }
  for (const auto &p : impl.parameters) {
    p->set_post_accumulate_grad_hook(nullptr);
  }
#ifdef MICROGRAD_NCCL_ENABLED
  cudaStreamSynchronize(impl.stream);
  for (Impl::Bucket &bucket : impl.buckets) {
    cudaEventDestroy(bucket.ready);
  }
  cudaEventDestroy(impl.done);
  cudaStreamDestroy(impl.stream);
#endif
}

void GradientReducer::prepare(bool reduce_this_backward) {
  Impl &impl = *impl_;
  impl.armed = impl.overlap && reduce_this_backward;
  if (!impl.armed) {
    return;
  }
  impl.ready.assign(impl.parameters.size(), false);
  for (Impl::Bucket &bucket : impl.buckets) {
    bucket.pending = bucket.params.size();
  }
  impl.next_bucket = 0;
}

void GradientReducer::finish() {
  Impl &impl = *impl_;
  State &state = GetState();
  if (!state.active || state.info.world_size == 1) {
    return;
  }
  if (!impl.overlap) {
    std::vector<Storage *> grads;
    grads.reserve(impl.parameters.size());
    for (const auto &p : impl.parameters) {
      grads.push_back(&p->grad_storage());
    }
    AllReduceSum(grads);
    return;
  }
  if (!impl.armed) {
    prepare(true);
  }
  while (impl.next_bucket < impl.buckets.size()) {
    impl.launch(impl.buckets[impl.next_bucket++]);
  }
  impl.armed = false;
#ifdef MICROGRAD_NCCL_ENABLED
  MICROGRAD_CUDA_CHECK(cudaEventRecord(impl.done, impl.stream));
  MICROGRAD_CUDA_CHECK(
      cudaStreamWaitEvent(CudaContext::instance().stream(), impl.done, 0));
#endif
}

bool GradientReducer::overlaps() const { return impl_->overlap; }

size_t GradientReducer::bucket_count() const { return impl_->buckets.size(); }

}  // namespace micrograd::distributed
