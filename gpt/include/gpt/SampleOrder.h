#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace micrograd::gpt {

class SampleOrder {
 public:
  struct Location {
    size_t span = 0;
    size_t offset = 0;
  };

  SampleOrder(const std::vector<size_t> &span_tokens, size_t block,
              uint64_t seed);

  Location next();

  size_t block_size() const { return block_size_; }
  size_t samples_per_epoch() const { return order_.size(); }
  size_t epoch() const { return epoch_; }
  size_t position() const { return position_; }

 private:
  void shuffle();
  Location locate(uint32_t sample) const;

  std::vector<size_t> span_first_sample_;
  size_t block_size_;
  uint64_t seed_;
  size_t epoch_ = 0;
  size_t position_ = 0;
  size_t shift_ = 0;
  std::vector<uint32_t> order_;
};

}  // namespace micrograd::gpt
