#pragma once

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

  size_t vocab_size() const { return vocab_size_; }
  size_t token_count() const { return tokens_.size(); }

 private:
  std::vector<uint8_t> tokens_;
  size_t train_size_ = 0;
  size_t vocab_size_ = 0;
};

}  // namespace micrograd::gpt
