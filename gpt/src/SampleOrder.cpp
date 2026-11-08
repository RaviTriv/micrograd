#include "gpt/SampleOrder.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace micrograd::gpt {

SampleOrder::SampleOrder(const std::vector<size_t> &span_tokens, size_t block,
                         uint64_t seed)
    : block_size_(block), seed_(seed) {
  if (block_size_ == 0) {
    throw std::invalid_argument("SampleOrder: block size must be positive");
  }

  size_t total = 0;
  span_first_sample_.reserve(span_tokens.size() + 1);
  for (size_t tokens : span_tokens) {
    span_first_sample_.push_back(total);
    if (tokens >= 2 * block_size_) {
      total += (tokens - block_size_) / block_size_;
    }
  }
  span_first_sample_.push_back(total);

  if (total == 0) {
    throw std::runtime_error(
        "SampleOrder: no span holds two blocks of tokens");
  }
  if (total > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("SampleOrder: too many samples for uint32 ids");
  }

  order_.resize(total);
  shuffle();
}

SampleOrder::Location SampleOrder::next() {
  if (position_ == order_.size()) {
    epoch_++;
    position_ = 0;
    shuffle();
  }
  return locate(order_[position_++]);
}

void SampleOrder::shuffle() {
  std::mt19937_64 rng(seed_ + static_cast<uint64_t>(epoch_));
  shift_ = static_cast<size_t>(rng() % block_size_);

  std::iota(order_.begin(), order_.end(), uint32_t{0});
  for (size_t i = order_.size() - 1; i > 0; i--) {
    auto j = static_cast<size_t>(rng() % (i + 1));
    std::swap(order_[i], order_[j]);
  }
}

SampleOrder::Location SampleOrder::locate(uint32_t sample) const {
  auto it = std::upper_bound(span_first_sample_.begin(),
                             span_first_sample_.end(), size_t{sample});
  auto span = static_cast<size_t>(
                  std::distance(span_first_sample_.begin(), it)) -
              1;
  size_t index = size_t{sample} - span_first_sample_[span];
  return {span, shift_ + (index * block_size_)};
}

}  // namespace micrograd::gpt
