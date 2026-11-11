#include "gpt/JsonText.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

using micrograd::gpt::append_utf8;
using micrograd::gpt::parse_json_string;

namespace {

std::string Parse(const std::string &data) {
  size_t pos = 0;
  return parse_json_string(data, pos);
}

std::string Utf8(uint32_t codepoint) {
  std::string out;
  append_utf8(out, codepoint);
  return out;
}

TEST(JsonText, AppendUtf8EncodesEachLength) {
  EXPECT_EQ(Utf8(0x41), "A");
  EXPECT_EQ(Utf8(0x7F), "\x7F");
  EXPECT_EQ(Utf8(0x80), "\xC2\x80");
  EXPECT_EQ(Utf8(0xE9), "\xC3\xA9");
  EXPECT_EQ(Utf8(0x7FF), "\xDF\xBF");
  EXPECT_EQ(Utf8(0x800), "\xE0\xA0\x80");
  EXPECT_EQ(Utf8(0x20AC), "\xE2\x82\xAC");
  EXPECT_EQ(Utf8(0xFFFF), "\xEF\xBF\xBF");
  EXPECT_EQ(Utf8(0x10000), "\xF0\x90\x80\x80");
  EXPECT_EQ(Utf8(0x1F600), "\xF0\x9F\x98\x80");
  EXPECT_EQ(Utf8(0x10FFFF), "\xF4\x8F\xBF\xBF");
}

TEST(JsonText, AppendUtf8Appends) {
  std::string out = "x";
  append_utf8(out, 0xE9);
  EXPECT_EQ(out, "x\xC3\xA9");
}

TEST(JsonText, ParsesPlainStringAndAdvancesPastQuote) {
  const std::string data = R"(  "hello" : 1)";
  size_t pos = 2;
  EXPECT_EQ(parse_json_string(data, pos), "hello");
  EXPECT_EQ(pos, 9U);
}

TEST(JsonText, ParsesEmptyString) { EXPECT_EQ(Parse(R"("")"), ""); }

TEST(JsonText, ParsesSimpleEscapes) {
  EXPECT_EQ(Parse(R"("\"\\\/\n\t\r\b\f")"), "\"\\/\n\t\r\b\f");
}

TEST(JsonText, KeepsRawUtf8Bytes) {
  EXPECT_EQ(Parse("\"caf\xC3\xA9\""), "caf\xC3\xA9");
}

TEST(JsonText, ParsesUnicodeEscapes) {
  EXPECT_EQ(Parse(R"("A")"), "A");
  EXPECT_EQ(Parse(R"("é")"), "\xC3\xA9");
  EXPECT_EQ(Parse(R"("é")"), "\xC3\xA9");
  EXPECT_EQ(Parse(R"("€")"), "\xE2\x82\xAC");
  EXPECT_EQ(Parse(R"("Ġx")"), "\xC4\xA0x");
}

TEST(JsonText, ParsesNulEscape) {
  EXPECT_EQ(Parse(R"("a\u0000b")"), std::string("a\0b", 3));
}

TEST(JsonText, CombinesSurrogatePairs) {
  EXPECT_EQ(Parse(R"("😀")"), "\xF0\x9F\x98\x80");
  EXPECT_EQ(Parse(R"("𐀀")"), "\xF0\x90\x80\x80");
  EXPECT_EQ(Parse(R"("􏿿!")"), "\xF4\x8F\xBF\xBF!");
}

TEST(JsonText, RejectsMissingOpeningQuote) {
  EXPECT_THROW(Parse("abc"), std::runtime_error);
}

TEST(JsonText, RejectsUnterminatedString) {
  EXPECT_THROW(Parse(R"("abc)"), std::out_of_range);
  EXPECT_THROW(Parse(R"("abc\)"), std::out_of_range);
}

TEST(JsonText, RejectsUnsupportedEscape) {
  EXPECT_THROW(Parse(R"("\x41")"), std::runtime_error);
}

TEST(JsonText, RejectsInvalidHexDigits) {
  EXPECT_THROW(Parse(R"("\u00g1")"), std::runtime_error);
}

TEST(JsonText, RejectsTruncatedUnicodeEscape) {
  EXPECT_THROW(Parse(R"("\u00)"), std::out_of_range);
}

TEST(JsonText, RejectsUnpairedHighSurrogate) {
  EXPECT_THROW(Parse(R"("\ud83dx")"), std::runtime_error);
  EXPECT_THROW(Parse(R"("\ud83d\n")"), std::runtime_error);
}

}  // namespace
