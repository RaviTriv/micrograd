#include "micrograd/nn/Serialize.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "micrograd/Storage.h"
#include "micrograd/Tensor.h"

namespace micrograd {

namespace {

constexpr uint32_t kStateDictMagic = 0x4d47534e;
constexpr uint32_t kStateDictVersion = 1;
constexpr uint32_t kMaxNameLength = 4096;
constexpr uint32_t kMaxRank = 8;

std::string format_shape(const std::vector<size_t> &shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i > 0) {
      result += ", ";
    }
    result += std::to_string(shape[i]);
  }
  result += "]";
  return result;
}

void write_u32(std::ofstream &file, uint32_t value) {
  file.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_u64(std::ofstream &file, uint64_t value) {
  file.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void read_exact(std::ifstream &file, void *destination, size_t bytes,
                const std::string &path) {
  file.read(static_cast<char *>(destination),
            static_cast<std::streamsize>(bytes));
  if (!file) {
    throw std::runtime_error("State dict file is truncated: " + path);
  }
}

uint32_t read_u32(std::ifstream &file, const std::string &path) {
  uint32_t value = 0;
  read_exact(file, &value, sizeof(value), path);
  return value;
}

uint64_t read_u64(std::ifstream &file, const std::string &path) {
  uint64_t value = 0;
  read_exact(file, &value, sizeof(value), path);
  return value;
}

size_t file_size(std::ifstream &file) {
  file.seekg(0, std::ios::end);
  const std::streamoff end = file.tellg();
  file.seekg(0, std::ios::beg);
  return static_cast<size_t>(end);
}

}  // namespace

void save(const std::string &path, nn::Module &module) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for saving: " + path);
  }

  auto named = module.named_parameters();

  write_u32(file, kStateDictMagic);
  write_u32(file, kStateDictVersion);
  write_u32(file, static_cast<uint32_t>(named.size()));

  for (auto &[name, tensor] : named) {
    write_u32(file, static_cast<uint32_t>(name.size()));
    file.write(name.data(), static_cast<std::streamsize>(name.size()));

    const auto &shape = tensor->shape();
    write_u32(file, static_cast<uint32_t>(shape.size()));
    for (auto dim : shape) {
      write_u64(file, static_cast<uint64_t>(dim));
    }

    Storage host_data = tensor->data_storage().copy_to(Device::CPU);
    file.write(static_cast<const char *>(host_data.data()),
               static_cast<std::streamsize>(host_data.bytes()));
  }

  if (!file) {
    throw std::runtime_error("Failed while writing state dict: " + path);
  }
}

void load(const std::string &path, nn::Module &module) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for loading: " + path);
  }

  const size_t max_elements = file_size(file) / sizeof(scalar_t);

  uint32_t magic = read_u32(file, path);
  if (magic != kStateDictMagic) {
    throw std::runtime_error("Not a micrograd state dict file: " + path);
  }

  uint32_t version = read_u32(file, path);
  if (version != kStateDictVersion) {
    throw std::runtime_error("Unsupported state dict version " +
                             std::to_string(version) + " in " + path);
  }

  uint32_t count = read_u32(file, path);

  std::unordered_map<std::string, std::pair<std::vector<size_t>, Storage>>
      stored;

  for (uint32_t i = 0; i < count; i++) {
    uint32_t name_len = read_u32(file, path);
    if (name_len > kMaxNameLength) {
      throw std::runtime_error("State dict name is too long in " + path);
    }
    std::string name(name_len, '\0');
    read_exact(file, name.data(), name_len, path);

    uint32_t ndim = read_u32(file, path);
    if (ndim > kMaxRank) {
      throw std::runtime_error("State dict rank is too large for " + name);
    }
    std::vector<size_t> shape(ndim);
    size_t total = 1;
    for (uint32_t d = 0; d < ndim; d++) {
      shape[d] = static_cast<size_t>(read_u64(file, path));
      const bool overflows = shape[d] != 0 && total > max_elements / shape[d];
      if (overflows) {
        throw std::runtime_error(
            "State dict shape is larger than the file for " + name);
      }
      total *= shape[d];
    }

    Storage host(total * sizeof(scalar_t), Device::CPU);
    read_exact(file, host.host_pointer(), host.bytes(), path);

    stored.emplace(std::move(name),
                   std::make_pair(std::move(shape), std::move(host)));
  }

  auto named = module.named_parameters();
  if (named.size() != stored.size()) {
    throw std::runtime_error("State dict parameter count mismatch: model has " +
                             std::to_string(named.size()) +
                             " parameters, file has " +
                             std::to_string(stored.size()));
  }

  for (auto &[name, tensor] : named) {
    auto it = stored.find(name);
    if (it == stored.end()) {
      throw std::runtime_error("State dict is missing parameter: " + name);
    }

    const auto &shape = it->second.first;
    const Storage &host = it->second.second;
    if (shape != tensor->shape()) {
      throw std::runtime_error(
          "State dict shape mismatch for parameter " + name + ": model has " +
          format_shape(tensor->shape()) + ", file has " + format_shape(shape));
    }
    if (host.bytes() != tensor->data_storage().bytes()) {
      throw std::runtime_error("State dict dtype mismatch for parameter " +
                               name);
    }

    tensor->data_storage() = host.copy_to(tensor->backend());
  }
}

}  // namespace micrograd
