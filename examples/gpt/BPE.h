#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace micrograd::gpt {

class BPE {
 public:
  static constexpr size_t kNanochatVocabSize = 65536;

  BPE(const std::string &vocab_path, const std::string &merges_path);

  static BPE train(const std::vector<std::string> &corpus,
                   size_t target_vocab_size = kNanochatVocabSize);

  std::vector<int32_t> encode(const std::string &text) const;
  std::string decode(const std::vector<int32_t> &ids) const;

  size_t vocab_size() const { return id_to_token_.size(); }

 private:
  BPE();
  void load_vocab(const std::string &vocab_path);
  void load_merges(const std::string &merges_path);
  std::string encode_bytes(const std::string &chunk) const;
  static size_t next_chunk_length(const std::string &text, size_t pos);
  std::vector<std::string> bpe_merge(std::vector<std::string> symbols) const;

  std::array<uint32_t, 256> byte_encoder_;
  std::unordered_map<uint32_t, uint8_t> byte_decoder_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int32_t> merge_ranks_;
};

}  // namespace micrograd::gpt
