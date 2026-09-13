#include "examples/gpt/BPE.h"

#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace micrograd::gpt {

namespace {

void append_utf8(std::string &out, uint32_t codepoint) {
  if (codepoint <= 0x7F) {
    out.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else if (codepoint <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  }
}

size_t utf8_char_length(unsigned char lead) {
  if ((lead & 0x80) == 0x00) {
    return 1;
  }
  if ((lead & 0xE0) == 0xC0) {
    return 2;
  }
  if ((lead & 0xF0) == 0xE0) {
    return 3;
  }
  if ((lead & 0xF8) == 0xF0) {
    return 4;
  }
  return 1;
}

uint32_t decode_utf8_codepoint(const std::string &data, size_t pos,
                               size_t len) {
  static constexpr std::array<unsigned char, 5> kLeadMask = {0x00, 0xFF, 0x1F,
                                                             0x0F, 0x07};
  uint32_t codepoint = static_cast<unsigned char>(data[pos]) & kLeadMask[len];
  for (size_t i = 1; i < len; ++i) {
    codepoint =
        (codepoint << 6) | (static_cast<unsigned char>(data[pos + i]) & 0x3F);
  }
  return codepoint;
}

bool is_letter(unsigned char b) {
  return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || b >= 0x80;
}

bool is_digit(unsigned char b) { return b >= '0' && b <= '9'; }

bool is_whitespace(unsigned char b) {
  return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\v' ||
         b == '\f';
}

bool is_other(unsigned char b) {
  return !is_letter(b) && !is_digit(b) && !is_whitespace(b);
}

std::array<uint32_t, 256> build_byte_encoder() {
  std::array<uint32_t, 256> table{};
  std::array<bool, 256> assigned{};
  auto mark = [&](int lo, int hi) {
    for (int b = lo; b <= hi; ++b) {
      table[static_cast<size_t>(b)] = static_cast<uint32_t>(b);
      assigned[static_cast<size_t>(b)] = true;
    }
  };
  mark('!', '~');
  mark(0xA1, 0xAC);
  mark(0xAE, 0xFF);
  uint32_t next = 256;
  for (int b = 0; b < 256; ++b) {
    if (!assigned[static_cast<size_t>(b)]) {
      table[static_cast<size_t>(b)] = next++;
    }
  }
  return table;
}

void skip_whitespace(const std::string &data, size_t &pos) {
  while (pos < data.size() &&
         std::isspace(static_cast<unsigned char>(data[pos]))) {
    ++pos;
  }
}

uint32_t parse_hex4(const std::string &data, size_t pos) {
  uint32_t value = 0;
  for (size_t i = 0; i < 4; ++i) {
    char c = data.at(pos + i);
    value <<= 4;
    if (c >= '0' && c <= '9') {
      value |= static_cast<uint32_t>(c - '0');
    } else if (c >= 'a' && c <= 'f') {
      value |= static_cast<uint32_t>(c - 'a' + 10);
    } else if (c >= 'A' && c <= 'F') {
      value |= static_cast<uint32_t>(c - 'A' + 10);
    } else {
      throw std::runtime_error("BPE: invalid unicode escape in vocab file");
    }
  }
  return value;
}

std::string parse_json_string(const std::string &data, size_t &pos) {
  if (data.at(pos) != '"') {
    throw std::runtime_error("BPE: expected string in vocab file");
  }
  ++pos;
  std::string result;
  while (true) {
    char c = data.at(pos);
    if (c == '"') {
      ++pos;
      return result;
    }
    if (c != '\\') {
      result.push_back(c);
      ++pos;
      continue;
    }
    char esc = data.at(pos + 1);
    switch (esc) {
      case '"':
        result.push_back('"');
        pos += 2;
        break;
      case '\\':
        result.push_back('\\');
        pos += 2;
        break;
      case '/':
        result.push_back('/');
        pos += 2;
        break;
      case 'n':
        result.push_back('\n');
        pos += 2;
        break;
      case 't':
        result.push_back('\t');
        pos += 2;
        break;
      case 'r':
        result.push_back('\r');
        pos += 2;
        break;
      case 'b':
        result.push_back('\b');
        pos += 2;
        break;
      case 'f':
        result.push_back('\f');
        pos += 2;
        break;
      case 'u': {
        uint32_t codepoint = parse_hex4(data, pos + 2);
        pos += 6;
        if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
          if (data.at(pos) != '\\' || data.at(pos + 1) != 'u') {
            throw std::runtime_error("BPE: unpaired surrogate in vocab file");
          }
          uint32_t low = parse_hex4(data, pos + 2);
          pos += 6;
          codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
        }
        append_utf8(result, codepoint);
        break;
      }
      default:
        throw std::runtime_error("BPE: unsupported escape in vocab file");
    }
  }
}

}  // namespace

BPE::BPE(const std::string &vocab_path, const std::string &merges_path)
    : byte_encoder_(build_byte_encoder()) {
  for (size_t b = 0; b < byte_encoder_.size(); ++b) {
    byte_decoder_[byte_encoder_[b]] = static_cast<uint8_t>(b);
  }
  load_vocab(vocab_path);
  load_merges(merges_path);
}

void BPE::load_vocab(const std::string &vocab_path) {
  std::ifstream file(vocab_path);
  if (!file) {
    throw std::runtime_error("BPE: could not open vocab file: " + vocab_path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  std::string data = buffer.str();

  size_t pos = 0;
  skip_whitespace(data, pos);
  if (data.at(pos) != '{') {
    throw std::runtime_error("BPE: malformed vocab file: " + vocab_path);
  }
  ++pos;
  skip_whitespace(data, pos);
  if (data.at(pos) == '}') {
    return;
  }

  while (true) {
    skip_whitespace(data, pos);
    std::string token = parse_json_string(data, pos);
    skip_whitespace(data, pos);
    if (data.at(pos) != ':') {
      throw std::runtime_error("BPE: malformed vocab file: " + vocab_path);
    }
    ++pos;
    skip_whitespace(data, pos);

    size_t start = pos;
    if (data.at(pos) == '-') {
      ++pos;
    }
    while (pos < data.size() &&
           std::isdigit(static_cast<unsigned char>(data[pos]))) {
      ++pos;
    }
    int32_t id = std::stoi(data.substr(start, pos - start));
    if (id < 0) {
      throw std::runtime_error("BPE: negative token id in vocab file");
    }

    if (static_cast<size_t>(id) >= id_to_token_.size()) {
      id_to_token_.resize(static_cast<size_t>(id) + 1);
    }
    id_to_token_[static_cast<size_t>(id)] = token;
    token_to_id_[token] = id;

    skip_whitespace(data, pos);
    char c = data.at(pos);
    ++pos;
    if (c == ',') {
      continue;
    }
    if (c == '}') {
      break;
    }
    throw std::runtime_error("BPE: malformed vocab file: " + vocab_path);
  }
}

void BPE::load_merges(const std::string &merges_path) {
  std::ifstream file(merges_path);
  if (!file) {
    throw std::runtime_error("BPE: could not open merges file: " + merges_path);
  }
  std::string line;
  int32_t rank = 0;
  bool first = true;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (first) {
      first = false;
      if (!line.empty() && line[0] == '#') {
        continue;
      }
    }
    if (line.empty()) {
      continue;
    }
    if (line.find(' ') == std::string::npos) {
      throw std::runtime_error("BPE: malformed merge line: " + line);
    }
    merge_ranks_[line] = rank++;
  }
}

std::string BPE::encode_bytes(const std::string &chunk) const {
  std::string out;
  out.reserve(chunk.size() * 2);
  for (char c : chunk) {
    append_utf8(out, byte_encoder_[static_cast<unsigned char>(c)]);
  }
  return out;
}

size_t BPE::next_chunk_length(const std::string &text, size_t pos) {
  static const std::array<std::string, 7> kContractions = {
      "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
  for (const auto &suffix : kContractions) {
    if (text.compare(pos, suffix.size(), suffix) == 0) {
      return suffix.size();
    }
  }

  const size_t n = text.size();
  auto byte_at = [&](size_t i) { return static_cast<unsigned char>(text[i]); };

  if (byte_at(pos) == ' ' && pos + 1 < n && is_letter(byte_at(pos + 1))) {
    size_t end = pos + 2;
    while (end < n && is_letter(byte_at(end))) {
      ++end;
    }
    return end - pos;
  }
  if (is_letter(byte_at(pos))) {
    size_t end = pos + 1;
    while (end < n && is_letter(byte_at(end))) {
      ++end;
    }
    return end - pos;
  }
  if (byte_at(pos) == ' ' && pos + 1 < n && is_digit(byte_at(pos + 1))) {
    size_t end = pos + 2;
    while (end < n && is_digit(byte_at(end))) {
      ++end;
    }
    return end - pos;
  }
  if (is_digit(byte_at(pos))) {
    size_t end = pos + 1;
    while (end < n && is_digit(byte_at(end))) {
      ++end;
    }
    return end - pos;
  }
  if (byte_at(pos) == ' ' && pos + 1 < n && is_other(byte_at(pos + 1))) {
    size_t end = pos + 2;
    while (end < n && is_other(byte_at(end))) {
      ++end;
    }
    return end - pos;
  }
  if (is_other(byte_at(pos))) {
    size_t end = pos + 1;
    while (end < n && is_other(byte_at(end))) {
      ++end;
    }
    return end - pos;
  }

  size_t end = pos + 1;
  while (end < n && is_whitespace(byte_at(end))) {
    ++end;
  }
  if (end == n) {
    return end - pos;
  }
  size_t run = end - pos;
  if (run >= 2) {
    return run - 1;
  }
  return run;
}

std::vector<std::string> BPE::bpe_merge(
    std::vector<std::string> symbols) const {
  while (symbols.size() > 1) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    size_t best_index = symbols.size();
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      auto it = merge_ranks_.find(symbols[i] + " " + symbols[i + 1]);
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_index = i;
      }
    }
    if (best_index == symbols.size()) {
      break;
    }
    symbols[best_index] += symbols[best_index + 1];
    symbols.erase(symbols.begin() + static_cast<ptrdiff_t>(best_index) + 1);
  }
  return symbols;
}

std::vector<int32_t> BPE::encode(const std::string &text) const {
  std::vector<int32_t> ids;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t chunk_len = next_chunk_length(text, pos);
    std::string byte_chars = encode_bytes(text.substr(pos, chunk_len));
    pos += chunk_len;

    std::vector<std::string> symbols;
    size_t p = 0;
    while (p < byte_chars.size()) {
      size_t len = utf8_char_length(static_cast<unsigned char>(byte_chars[p]));
      symbols.push_back(byte_chars.substr(p, len));
      p += len;
    }

    for (const auto &piece : bpe_merge(std::move(symbols))) {
      auto it = token_to_id_.find(piece);
      if (it == token_to_id_.end()) {
        throw std::runtime_error("BPE: token not found in vocab: " + piece);
      }
      ids.push_back(it->second);
    }
  }
  return ids;
}

std::string BPE::decode(const std::vector<int32_t> &ids) const {
  std::string byte_chars;
  for (int32_t id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
      throw std::runtime_error("BPE: token id out of range: " +
                               std::to_string(id));
    }
    byte_chars += id_to_token_[static_cast<size_t>(id)];
  }

  std::string bytes;
  size_t pos = 0;
  while (pos < byte_chars.size()) {
    size_t len = utf8_char_length(static_cast<unsigned char>(byte_chars[pos]));
    uint32_t codepoint = decode_utf8_codepoint(byte_chars, pos, len);
    auto it = byte_decoder_.find(codepoint);
    if (it == byte_decoder_.end()) {
      throw std::runtime_error("BPE: invalid byte-level token during decode");
    }
    bytes.push_back(static_cast<char>(it->second));
    pos += len;
  }
  return bytes;
}

}  // namespace micrograd::gpt
