#include "examples/gpt/Dataset.h"

#include <array>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace micrograd::gpt {

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

  std::array<size_t, 256> byte_to_index{};
  for (size_t byte = 0; byte < 256; byte++) {
    if (seen[byte]) {
      byte_to_index[byte] = vocab_size_;
      vocab_size_++;
    }
  }

  tokens_.resize(text.size());
  for (size_t i = 0; i < text.size(); i++) {
    tokens_[i] = static_cast<uint8_t>(
        byte_to_index[static_cast<unsigned char>(text[i])]);
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

}  // namespace micrograd::gpt
