#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace micrograd::gpt {

class Dataset {
 public:
  enum class Split { kTrain, kVal };

  struct Batch {
    std::vector<size_t> inputs;
    std::vector<size_t> targets;
  };

  Dataset(const std::string &path, double val_fraction);

  Batch sample(size_t batch_size, size_t block_size, Split split,
               std::mt19937_64 &rng) const;

  std::vector<size_t> encode(const std::string &text) const;
  char decode(size_t token) const;

  size_t vocab_size() const { return vocab_size_; }
  size_t token_count() const { return tokens_.size(); }

 private:
  std::vector<uint8_t> tokens_;
  size_t train_size_ = 0;
  size_t vocab_size_ = 0;
  std::array<int64_t, 256> byte_to_index_{};
  std::vector<uint8_t> itos_;
};

class ShardedDataset {
 public:
  ShardedDataset(const std::vector<std::string> &train_shard_paths,
                 const std::string &val_shard_path);
  ~ShardedDataset();

  ShardedDataset(const ShardedDataset &) = delete;
  ShardedDataset &operator=(const ShardedDataset &) = delete;

  Dataset::Batch sample(size_t batch_size, size_t block_size,
                        Dataset::Split split, std::mt19937_64 &rng) const;

  size_t token_count(Dataset::Split split) const;

 private:
  struct Shard {
    const uint16_t *tokens = nullptr;
    size_t token_count = 0;
    void *mapping = nullptr;
    size_t mapping_size = 0;
    int fd = -1;
  };

  static Shard map_shard(const std::string &path);
  static void unmap_shard(Shard &shard);

  std::vector<Shard> train_shards_;
  Shard val_shard_;
};

}  // namespace micrograd::gpt
