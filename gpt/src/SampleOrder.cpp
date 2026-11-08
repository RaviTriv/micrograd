#include "gpt/SampleOrder.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace micrograd::gpt {

namespace {
constexpr uint32_t kStateMagic = 20261108;
constexpr uint32_t kStateVersion = 1;

template <typename T>
void write_value(std::ostream &out, T value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

template <typename T>
T read_value(std::istream &in) {
  T value{};
  in.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error("SampleOrder: data state is truncated");
  }
  return value;
}

void expect_match(const char *field, uint64_t saved, uint64_t current) {
  if (saved != current) {
    throw std::runtime_error(
        std::string("SampleOrder: data state was saved with ") + field + " " +
        std::to_string(saved) + ", this run has " + std::to_string(current));
  }
}
}  // namespace

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

void SampleOrder::seek(size_t samples) {
  epoch_ = samples / order_.size();
  position_ = samples % order_.size();
  shuffle();
}

void SampleOrder::write_state(std::ostream &out) const {
  write_value(out, kStateMagic);
  write_value(out, kStateVersion);
  write_value(out, seed_);
  write_value(out, static_cast<uint64_t>(block_size_));
  write_value(out, static_cast<uint64_t>(order_.size()));
  write_value(out, static_cast<uint64_t>(consumed()));
}

void SampleOrder::read_state(std::istream &in) {
  if (read_value<uint32_t>(in) != kStateMagic) {
    throw std::runtime_error("SampleOrder: not a data state file");
  }
  auto version = read_value<uint32_t>(in);
  if (version != kStateVersion) {
    throw std::runtime_error("SampleOrder: unsupported data state version " +
                             std::to_string(version));
  }
  expect_match("seed", read_value<uint64_t>(in), seed_);
  expect_match("block size", read_value<uint64_t>(in), block_size_);
  expect_match("samples per epoch", read_value<uint64_t>(in), order_.size());
  seek(static_cast<size_t>(read_value<uint64_t>(in)));
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
