#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
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
  void seek(size_t samples);

  void write_state(std::ostream &out) const;
  void read_state(std::istream &in);

  size_t block_size() const { return block_size_; }
  size_t samples_per_epoch() const { return order_.size(); }
  size_t epoch() const { return epoch_; }
  size_t position() const { return position_; }
  size_t consumed() const { return (epoch_ * order_.size()) + position_; }

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
