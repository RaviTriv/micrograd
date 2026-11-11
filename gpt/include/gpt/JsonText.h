#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace micrograd::gpt {

void append_utf8(std::string &out, uint32_t codepoint);

std::string parse_json_string(const std::string &data, size_t &pos);

}  // namespace micrograd::gpt
