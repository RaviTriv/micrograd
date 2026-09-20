#include "examples/gpt/Dataset.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace micrograd::gpt {

namespace {
constexpr int32_t kShardMagic = 20240520;
constexpr int32_t kShardVersion = 1;
constexpr size_t kShardHeaderInts = 256;
constexpr size_t kShardHeaderBytes = kShardHeaderInts * sizeof(int32_t);
}  // namespace

Dataset::Dataset(const std::string &path, double val_fraction) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Dataset: could not open " + path);
  }
  std::string text((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  if (text.empty()) {
    throw std::runtime_error("Dataset: " + path + " is empty");
  }
  if (val_fraction <= 0.0 || val_fraction >= 1.0) {
    throw std::invalid_argument("Dataset: val_fraction must be in (0, 1)");
  }

  std::array<bool, 256> seen{};
  for (char c : text) {
    seen[static_cast<unsigned char>(c)] = true;
  }

  byte_to_index_.fill(-1);
  for (size_t byte = 0; byte < 256; byte++) {
    if (seen[byte]) {
      byte_to_index_[byte] = static_cast<int64_t>(vocab_size_);
      itos_.push_back(static_cast<uint8_t>(byte));
      vocab_size_++;
    }
  }

  tokens_.resize(text.size());
  for (size_t i = 0; i < text.size(); i++) {
    tokens_[i] = static_cast<uint8_t>(
        byte_to_index_[static_cast<unsigned char>(text[i])]);
  }

  auto val_size =
      static_cast<size_t>(static_cast<double>(tokens_.size()) * val_fraction);
  train_size_ = tokens_.size() - val_size;
  if (train_size_ == 0 || val_size == 0) {
    throw std::runtime_error(
        "Dataset: corpus too small for the requested validation split");
  }
}

Dataset::Batch Dataset::sample(size_t batch_size, size_t block_size,
                               Split split, std::mt19937_64 &rng) const {
  size_t begin = split == Split::kTrain ? 0 : train_size_;
  size_t end = split == Split::kTrain ? train_size_ : tokens_.size();
  if (end - begin <= block_size) {
    throw std::runtime_error(
        "Dataset: not enough tokens for the requested block size");
  }
  size_t window_count = (end - begin) - block_size;

  Batch batch;
  batch.inputs.resize(batch_size * block_size);
  batch.targets.resize(batch_size * block_size);

  std::uniform_int_distribution<size_t> pick(0, window_count - 1);
  for (size_t row = 0; row < batch_size; row++) {
    size_t start = begin + pick(rng);
    for (size_t col = 0; col < block_size; col++) {
      size_t index = (row * block_size) + col;
      batch.inputs[index] = tokens_[start + col];
      batch.targets[index] = tokens_[start + col + 1];
    }
  }
  return batch;
}

std::vector<size_t> Dataset::encode(const std::string &text) const {
  std::vector<size_t> tokens;
  tokens.reserve(text.size());
  for (char c : text) {
    int64_t index = byte_to_index_[static_cast<unsigned char>(c)];
    if (index < 0) {
      throw std::invalid_argument("Dataset::encode: byte not in vocabulary");
    }
    tokens.push_back(static_cast<size_t>(index));
  }
  return tokens;
}

char Dataset::decode(size_t token) const {
  if (token >= itos_.size()) {
    throw std::invalid_argument("Dataset::decode: token out of range");
  }
  return static_cast<char>(itos_[token]);
}

ShardedDataset::Shard ShardedDataset::map_shard(const std::string &path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("ShardedDataset: could not open " + path);
  }

  struct stat info {};
  if (fstat(fd, &info) != 0) {
    close(fd);
    throw std::runtime_error("ShardedDataset: could not stat " + path);
  }
  auto file_size = static_cast<size_t>(info.st_size);
  if (file_size < kShardHeaderBytes) {
    close(fd);
    throw std::runtime_error("ShardedDataset: " + path +
                             " is smaller than the shard header");
  }

  void *mapping = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapping == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("ShardedDataset: mmap failed for " + path);
  }

  const auto *header = static_cast<const int32_t *>(mapping);
  if (header[0] != kShardMagic || header[1] != kShardVersion) {
    munmap(mapping, file_size);
    close(fd);
    throw std::runtime_error("ShardedDataset: " + path +
                             " has an unrecognised shard header");
  }
  auto shard_tokens = static_cast<size_t>(header[2]);
  size_t expected_size = kShardHeaderBytes + (shard_tokens * sizeof(uint16_t));
  if (file_size < expected_size) {
    munmap(mapping, file_size);
    close(fd);
    throw std::runtime_error("ShardedDataset: " + path + " is truncated");
  }

  Shard shard;
  shard.mapping = mapping;
  shard.mapping_size = file_size;
  shard.fd = fd;
  shard.token_count = shard_tokens;
  shard.tokens = reinterpret_cast<const uint16_t *>(
      static_cast<const uint8_t *>(mapping) + kShardHeaderBytes);
  return shard;
}

void ShardedDataset::unmap_shard(Shard &shard) {
  if (shard.mapping != nullptr) {
    munmap(shard.mapping, shard.mapping_size);
    shard.mapping = nullptr;
  }
  if (shard.fd >= 0) {
    close(shard.fd);
    shard.fd = -1;
  }
}

ShardedDataset::ShardedDataset(
    const std::vector<std::string> &train_shard_paths,
    const std::string &val_shard_path) {
  if (train_shard_paths.empty()) {
    throw std::invalid_argument(
        "ShardedDataset: at least one training shard is required");
  }
  train_shards_.reserve(train_shard_paths.size());
  for (const auto &path : train_shard_paths) {
    train_shards_.push_back(map_shard(path));
  }
  val_shard_ = map_shard(val_shard_path);
}

ShardedDataset::~ShardedDataset() {
  for (auto &shard : train_shards_) {
    unmap_shard(shard);
  }
  unmap_shard(val_shard_);
}

Dataset::Batch ShardedDataset::sample(size_t batch_size, size_t block_size,
                                      Dataset::Split split,
                                      std::mt19937_64 &rng) const {
  Dataset::Batch batch;
  batch.inputs.resize(batch_size * block_size);
  batch.targets.resize(batch_size * block_size);

  std::uniform_int_distribution<size_t> pick_shard(
      0, split == Dataset::Split::kTrain ? train_shards_.size() - 1 : 0);

  for (size_t row = 0; row < batch_size; row++) {
    const Shard &shard = split == Dataset::Split::kTrain
                             ? train_shards_[pick_shard(rng)]
                             : val_shard_;
    if (shard.token_count <= block_size) {
      throw std::runtime_error(
          "ShardedDataset: shard too small for the requested block size");
    }
    std::uniform_int_distribution<size_t> pick_start(
        0, shard.token_count - block_size - 1);
    size_t start = pick_start(rng);
    for (size_t col = 0; col < block_size; col++) {
      size_t index = (row * block_size) + col;
      batch.inputs[index] = shard.tokens[start + col];
      batch.targets[index] = shard.tokens[start + col + 1];
    }
  }
  return batch;
}

size_t ShardedDataset::token_count(Dataset::Split split) const {
  if (split == Dataset::Split::kTrain) {
    size_t total = 0;
    for (const auto &shard : train_shards_) {
      total += shard.token_count;
    }
    return total;
  }
  return val_shard_.token_count;
}

}  // namespace micrograd::gpt
