#include "micrograd/nn/Serialize.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "micrograd/Storage.h"
#include "micrograd/Tensor.h"

namespace micrograd {

namespace {

static_assert(std::endian::native == std::endian::little);

constexpr uint32_t kStateDictMagic = 0x4d47534e;
constexpr uint32_t kVersionFloat32Only = 1;
constexpr uint32_t kVersionWithDTypeAndMetadata = 2;
constexpr uint32_t kMaxNameLength = 4096;
constexpr uint32_t kMaxRank = 8;
constexpr uint32_t kFloat32Code = 0;
constexpr uint32_t kBFloat16Code = 1;

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

std::string dtype_name(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return "float32";
    case DType::kBFloat16:
      return "bfloat16";
  }
  throw std::runtime_error("Unknown dtype");
}

uint32_t dtype_code(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return kFloat32Code;
    case DType::kBFloat16:
      return kBFloat16Code;
  }
  throw std::runtime_error("Unknown dtype");
}

class Writer {
 public:
  explicit Writer(const std::string &path)
      : file_(path, std::ios::binary), path_(path) {
    if (!file_.is_open()) {
      throw std::runtime_error("Could not open file for saving: " + path);
    }
  }

  void u32(uint32_t value) { bytes(&value, sizeof(value)); }

  void u64(uint64_t value) { bytes(&value, sizeof(value)); }

  void string(const std::string &value, size_t max_length,
              const std::string &what) {
    if (value.size() > max_length) {
      throw std::runtime_error(what + " is too long to save: " + path_);
    }
    u32(static_cast<uint32_t>(value.size()));
    bytes(value.data(), value.size());
  }

  void bytes(const void *data, size_t count) {
    file_.write(static_cast<const char *>(data),
                static_cast<std::streamsize>(count));
  }

  void close() {
    file_.close();
    if (!file_) {
      throw std::runtime_error("Failed while writing state dict: " + path_);
    }
  }

 private:
  std::ofstream file_;
  std::string path_;
};

class Reader {
 public:
  explicit Reader(const std::string &path)
      : file_(path, std::ios::binary), path_(path) {
    if (!file_.is_open()) {
      throw std::runtime_error("Could not open file for loading: " + path);
    }
    file_.seekg(0, std::ios::end);
    size_ = static_cast<size_t>(file_.tellg());
    file_.seekg(0, std::ios::beg);
  }

  uint32_t u32() {
    uint32_t value = 0;
    bytes(&value, sizeof(value));
    return value;
  }

  uint64_t u64() {
    uint64_t value = 0;
    bytes(&value, sizeof(value));
    return value;
  }

  std::string string(size_t max_length, const std::string &what) {
    const uint32_t length = u32();
    if (length > max_length) {
      fail(what + " is too long");
    }
    std::string value(length, '\0');
    bytes(value.data(), length);
    return value;
  }

  void bytes(void *destination, size_t count) {
    file_.read(static_cast<char *>(destination),
               static_cast<std::streamsize>(count));
    if (!file_) {
      throw std::runtime_error("State dict file is truncated: " + path_);
    }
  }

  bool at_end() {
    return file_.peek() == std::ifstream::traits_type::eof();
  }

  size_t size() const { return size_; }

  [[noreturn]] void fail(const std::string &message) const {
    throw std::runtime_error(message + " in " + path_);
  }

 private:
  std::ifstream file_;
  std::string path_;
  size_t size_ = 0;
};

struct StoredTensor {
  DType dtype = DType::kFloat32;
  std::vector<size_t> shape;
  Storage data;
};

uint32_t read_header(Reader &in) {
  if (in.u32() != kStateDictMagic) {
    in.fail("Not a micrograd state dict file");
  }
  const uint32_t version = in.u32();
  if (version != kVersionFloat32Only &&
      version != kVersionWithDTypeAndMetadata) {
    in.fail("Unsupported state dict version " + std::to_string(version));
  }
  return version;
}

StateDictMetadata read_metadata(Reader &in, uint32_t version) {
  StateDictMetadata metadata;
  if (version == kVersionFloat32Only) {
    return metadata;
  }
  const uint32_t count = in.u32();
  for (uint32_t i = 0; i < count; i++) {
    std::string key = in.string(kMaxNameLength, "Metadata key");
    std::string value = in.string(in.size(), "Metadata value for " + key);
    if (metadata.contains(key)) {
      in.fail("Duplicate metadata key " + key);
    }
    metadata.emplace(std::move(key), std::move(value));
  }
  return metadata;
}

DType read_dtype(Reader &in, uint32_t version, const std::string &name) {
  if (version == kVersionFloat32Only) {
    return DType::kFloat32;
  }
  const uint32_t code = in.u32();
  switch (code) {
    case kFloat32Code:
      return DType::kFloat32;
    case kBFloat16Code:
      return DType::kBFloat16;
    default:
      in.fail("Unknown dtype code " + std::to_string(code) +
              " for parameter " + name);
  }
}

StoredTensor read_tensor(Reader &in, uint32_t version,
                         const std::string &name) {
  StoredTensor tensor;
  tensor.dtype = read_dtype(in, version, name);

  const uint32_t ndim = in.u32();
  if (ndim > kMaxRank) {
    in.fail("Rank " + std::to_string(ndim) + " is too large for parameter " +
            name);
  }
  const size_t element_bytes = dtype_size(tensor.dtype);
  const size_t max_elements = in.size() / element_bytes;
  tensor.shape.resize(ndim);
  size_t total = 1;
  for (uint32_t d = 0; d < ndim; d++) {
    tensor.shape[d] = static_cast<size_t>(in.u64());
    if (tensor.shape[d] != 0 && total > max_elements / tensor.shape[d]) {
      in.fail("Shape is larger than the file for parameter " + name);
    }
    total *= tensor.shape[d];
  }

  tensor.data = Storage(total * element_bytes, Device::CPU, tensor.dtype);
  in.bytes(tensor.data.host_pointer(), tensor.data.bytes());
  return tensor;
}

std::unordered_map<std::string, StoredTensor> read_tensors(Reader &in,
                                                           uint32_t version) {
  std::unordered_map<std::string, StoredTensor> stored;
  const uint32_t count = in.u32();
  for (uint32_t i = 0; i < count; i++) {
    std::string name = in.string(kMaxNameLength, "Parameter name");
    StoredTensor tensor = read_tensor(in, version, name);
    if (stored.contains(name)) {
      in.fail("Duplicate parameter " + name);
    }
    stored.emplace(std::move(name), std::move(tensor));
  }
  if (!in.at_end()) {
    in.fail("Unexpected data after the last parameter");
  }
  return stored;
}

std::vector<Storage> stage_parameters(
    const std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> &named,
    const std::unordered_map<std::string, StoredTensor> &stored,
    const Reader &in) {
  if (named.size() != stored.size()) {
    in.fail("State dict parameter count mismatch: model has " +
            std::to_string(named.size()) + " parameters, file has " +
            std::to_string(stored.size()));
  }

  std::vector<Storage> staged;
  staged.reserve(named.size());
  for (const auto &[name, tensor] : named) {
    auto it = stored.find(name);
    if (it == stored.end()) {
      in.fail("State dict is missing parameter " + name);
    }
    const StoredTensor &entry = it->second;
    if (entry.shape != tensor->shape()) {
      in.fail("State dict shape mismatch for parameter " + name +
              ": model has " + format_shape(tensor->shape()) +
              ", file has " + format_shape(entry.shape));
    }
    if (entry.dtype != tensor->dtype()) {
      in.fail("State dict dtype mismatch for parameter " + name +
              ": model has " + dtype_name(tensor->dtype()) + ", file has " +
              dtype_name(entry.dtype));
    }
    if (entry.data.bytes() != tensor->data_storage().bytes()) {
      in.fail("State dict size mismatch for parameter " + name +
              ": model has " + std::to_string(tensor->data_storage().bytes()) +
              " bytes, file has " + std::to_string(entry.data.bytes()));
    }
    staged.push_back(entry.data.copy_to(tensor->backend()));
  }
  return staged;
}

void write_state_dict(const std::string &path, nn::Module &module,
                      const StateDictMetadata &metadata) {
  constexpr size_t kMaxValueLength = std::numeric_limits<uint32_t>::max();
  Writer out(path);
  out.u32(kStateDictMagic);
  out.u32(kVersionWithDTypeAndMetadata);

  out.u32(static_cast<uint32_t>(metadata.size()));
  for (const auto &[key, value] : metadata) {
    out.string(key, kMaxNameLength, "Metadata key " + key);
    out.string(value, kMaxValueLength, "Metadata value for " + key);
  }

  auto named = module.named_parameters();
  out.u32(static_cast<uint32_t>(named.size()));
  for (const auto &[name, tensor] : named) {
    out.string(name, kMaxNameLength, "Parameter name " + name);
    out.u32(dtype_code(tensor->dtype()));
    const auto &shape = tensor->shape();
    out.u32(static_cast<uint32_t>(shape.size()));
    for (auto dim : shape) {
      out.u64(static_cast<uint64_t>(dim));
    }
    Storage host = tensor->data_storage().copy_to(Device::CPU);
    out.bytes(host.host_pointer(), host.bytes());
  }
  out.close();
}

}  // namespace

void save(const std::string &path, nn::Module &module) {
  save(path, module, StateDictMetadata{});
}

void save(const std::string &path, nn::Module &module,
          const StateDictMetadata &metadata) {
  const std::filesystem::path target(path);
  std::filesystem::path temporary = target;
  temporary += ".tmp";
  try {
    write_state_dict(temporary.string(), module, metadata);
    std::filesystem::rename(temporary, target);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

void load(const std::string &path, nn::Module &module) {
  Reader in(path);
  const uint32_t version = read_header(in);
  read_metadata(in, version);
  const auto stored = read_tensors(in, version);

  const auto named = module.named_parameters();
  std::vector<Storage> staged = stage_parameters(named, stored, in);
  for (size_t i = 0; i < named.size(); i++) {
    named[i].second->data_storage() = std::move(staged[i]);
  }
}

StateDictMetadata load_metadata(const std::string &path) {
  Reader in(path);
  const uint32_t version = read_header(in);
  return read_metadata(in, version);
}

}  // namespace micrograd
