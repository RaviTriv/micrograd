#include "gpt/JsonText.h"

#include <stdexcept>

namespace micrograd::gpt {

namespace {

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
      throw std::runtime_error("json: invalid unicode escape");
    }
  }
  return value;
}

}  // namespace

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

std::string parse_json_string(const std::string &data, size_t &pos) {
  if (data.at(pos) != '"') {
    throw std::runtime_error("json: expected a string");
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
            throw std::runtime_error("json: unpaired surrogate");
          }
          uint32_t low = parse_hex4(data, pos + 2);
          pos += 6;
          codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
        }
        append_utf8(result, codepoint);
        break;
      }
      default:
        throw std::runtime_error("json: unsupported escape");
    }
  }
}

}  // namespace micrograd::gpt
