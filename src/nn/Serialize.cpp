#include "micrograd/nn/Serialize.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "micrograd/Tensor.h"

namespace micrograd {

namespace {

constexpr uint32_t kStateDictMagic = 0x4d47534e;
constexpr uint32_t kStateDictVersion = 1;

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

uint32_t read_u32(std::ifstream &file) {
  uint32_t value = 0;
  file.read(reinterpret_cast<char *>(&value), sizeof(value));
  return value;
}

uint64_t read_u64(std::ifstream &file) {
  uint64_t value = 0;
  file.read(reinterpret_cast<char *>(&value), sizeof(value));
  return value;
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

  uint32_t magic = read_u32(file);
  if (magic != kStateDictMagic) {
    throw std::runtime_error("Not a micrograd state dict file: " + path);
  }

  uint32_t version = read_u32(file);
  if (version != kStateDictVersion) {
    throw std::runtime_error("Unsupported state dict version " +
                             std::to_string(version) + " in " + path);
  }

  uint32_t count = read_u32(file);

  std::unordered_map<std::string,
                     std::pair<std::vector<size_t>, std::vector<scalar_t>>>
      stored;
  stored.reserve(count);

  for (uint32_t i = 0; i < count; i++) {
    uint32_t name_len = read_u32(file);
    std::string name(name_len, '\0');
    file.read(name.data(), name_len);

    uint32_t ndim = read_u32(file);
    std::vector<size_t> shape(ndim);
    size_t total = 1;
    for (uint32_t d = 0; d < ndim; d++) {
      shape[d] = static_cast<size_t>(read_u64(file));
      total *= shape[d];
    }

    std::vector<scalar_t> values(total);
    file.read(reinterpret_cast<char *>(values.data()),
              static_cast<std::streamsize>(total * sizeof(scalar_t)));

    stored.emplace(std::move(name),
                   std::make_pair(std::move(shape), std::move(values)));
  }

  if (!file) {
    throw std::runtime_error("State dict file is truncated: " + path);
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
    const auto &values = it->second.second;
    if (shape != tensor->shape()) {
      throw std::runtime_error(
          "State dict shape mismatch for parameter " + name + ": model has " +
          format_shape(tensor->shape()) + ", file has " + format_shape(shape));
    }

    auto data = tensor->data();
    std::ranges::copy(values, data.begin());
  }
}

}  // namespace micrograd
