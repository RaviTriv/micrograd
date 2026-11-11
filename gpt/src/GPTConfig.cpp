#include "gpt/GPTConfig.h"

#include <algorithm>
#include <charconv>
#include <stdexcept>
#include <string>
#include <system_error>

namespace micrograd::gpt {

namespace {

constexpr size_t kChannelsPerLayer = 64;
constexpr size_t kHeadDim = 128;
constexpr size_t kReferenceEmbd = 768;
constexpr scalar_t kMatrixLr = 0.02f;
constexpr scalar_t kEmbeddingLr = 0.2f;
constexpr scalar_t kUnembeddingLr = 0.004f;

constexpr size_t kGpt2Depth = 12;
constexpr size_t kGpt2NLayer = 12;
constexpr size_t kGpt2NEmbd = 768;
constexpr size_t kGpt2NHead = 12;
constexpr size_t kGpt2BlockSize = 1024;
constexpr size_t kGpt2VocabSize = 50257;

constexpr const char *kFormatKey = "format";
constexpr const char *kArchFormat = "gpt-arch-1";

const char *bool_text(bool value) { return value ? "true" : "false"; }

std::runtime_error bad_key(const std::string &key, const std::string &text,
                           const std::string &expected) {
  return std::runtime_error("checkpoint architecture key " + key + " is '" +
                            text + "', expected " + expected);
}

const std::string &require_key(const StateDictMetadata &metadata,
                               const std::string &key) {
  auto it = metadata.find(key);
  if (it == metadata.end()) {
    throw std::runtime_error("checkpoint architecture is missing key " + key);
  }
  return it->second;
}

size_t read_size(const StateDictMetadata &metadata, const std::string &key,
                 bool positive) {
  const std::string &text = require_key(metadata, key);
  const char *end = text.data() + text.size();
  size_t value = 0;
  auto [ptr, error] = std::from_chars(text.data(), end, value);
  if (error != std::errc() || ptr != end || (positive && value == 0)) {
    throw bad_key(key, text,
                  positive ? "a positive integer" : "a non-negative integer");
  }
  return value;
}

bool read_bool(const StateDictMetadata &metadata, const std::string &key) {
  const std::string &text = require_key(metadata, key);
  if (text != "true" && text != "false") {
    throw bad_key(key, text, "true or false");
  }
  return text == "true";
}

}  // namespace

GPTConfig config_for_depth(size_t depth) {
  if (depth == 0) {
    throw std::invalid_argument("depth must be positive");
  }

  GPTConfig config;
  config.depth = depth;
  config.n_layer = depth;
  config.n_embd = depth * kChannelsPerLayer;
  config.n_head = std::max<size_t>(1, config.n_embd / kHeadDim);
  config.n_kv_head = config.n_head;

  scalar_t width_scale = static_cast<scalar_t>(kReferenceEmbd) /
                         static_cast<scalar_t>(config.n_embd);
  config.matrix_lr = kMatrixLr;
  config.embedding_lr = kEmbeddingLr * width_scale;
  config.unembedding_lr = kUnembeddingLr * width_scale;

  return config;
}

GPTConfig gpt2_124m() {
  GPTConfig config;
  config.depth = kGpt2Depth;
  config.n_layer = kGpt2NLayer;
  config.n_embd = kGpt2NEmbd;
  config.n_head = kGpt2NHead;
  config.n_kv_head = kGpt2NHead;
  config.block_size = kGpt2BlockSize;
  config.vocab_size = kGpt2VocabSize;
  config.matrix_lr = kMatrixLr;
  config.embedding_lr = kEmbeddingLr;
  config.unembedding_lr = kUnembeddingLr;
  return config;
}

StateDictMetadata architecture_metadata(const GPTConfig &config) {
  return {
      {kFormatKey, kArchFormat},
      {"n_layer", std::to_string(config.n_layer)},
      {"n_embd", std::to_string(config.n_embd)},
      {"n_head", std::to_string(config.n_head)},
      {"block_size", std::to_string(config.block_size)},
      {"vocab_size", std::to_string(config.vocab_size)},
      {"padded_vocab_size", std::to_string(config.padded_vocab_size)},
      {"rotary", bool_text(config.arch.rotary)},
      {"qk_norm", bool_text(config.arch.qk_norm)},
      {"relu_squared", bool_text(config.arch.relu_squared)},
  };
}

GPTConfig config_from_metadata(const StateDictMetadata &metadata) {
  auto format = metadata.find(kFormatKey);
  if (format == metadata.end()) {
    throw std::runtime_error(
        "checkpoint predates architecture metadata; re-save it with "
        "train_gpt");
  }
  if (format->second != kArchFormat) {
    throw bad_key(kFormatKey, format->second, kArchFormat);
  }

  GPTConfig config;
  config.n_layer = read_size(metadata, "n_layer", true);
  config.n_embd = read_size(metadata, "n_embd", true);
  config.n_head = read_size(metadata, "n_head", true);
  config.block_size = read_size(metadata, "block_size", true);
  config.vocab_size = read_size(metadata, "vocab_size", true);
  config.padded_vocab_size = read_size(metadata, "padded_vocab_size", false);
  config.arch.rotary = read_bool(metadata, "rotary");
  config.arch.qk_norm = read_bool(metadata, "qk_norm");
  config.arch.relu_squared = read_bool(metadata, "relu_squared");

  if (config.n_embd % config.n_head != 0) {
    throw bad_key("n_head", metadata.at("n_head"),
                  "a divisor of n_embd " + metadata.at("n_embd"));
  }
  if (config.padded_vocab_size != 0 &&
      config.padded_vocab_size < config.vocab_size) {
    throw bad_key("padded_vocab_size", metadata.at("padded_vocab_size"),
                  "0 or at least vocab_size " + metadata.at("vocab_size"));
  }

  config.depth = config.n_layer;
  config.n_kv_head = config.n_head;
  config.matrix_lr = 0.0f;
  config.embedding_lr = 0.0f;
  config.unembedding_lr = 0.0f;
  return config;
}

void require_same_architecture(const GPTConfig &checkpoint,
                               const GPTConfig &flags) {
  const StateDictMetadata saved = architecture_metadata(checkpoint);
  const StateDictMetadata wanted = architecture_metadata(flags);
  std::string differences;
  for (const auto &[key, value] : saved) {
    if (wanted.at(key) != value) {
      differences += "\n  " + key;
      differences += ": checkpoint " + value;
      differences += ", flags " + wanted.at(key);
    }
  }
  if (!differences.empty()) {
    throw std::runtime_error(
        "checkpoint architecture does not match the flags:" + differences);
  }
}

}  // namespace micrograd::gpt
