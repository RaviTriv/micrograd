#include "gpt/RunLog.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <locale>
#include <optional>
#include <sstream>
#include <string>

namespace micrograd::gpt {

namespace {

constexpr int kValueDigits = 7;

std::ostringstream json_stream() {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  return out;
}

std::string number(double value) {
  if (!std::isfinite(value)) {
    return "null";
  }
  std::ostringstream out = json_stream();
  out << std::setprecision(kValueDigits) << value;
  return out.str();
}

std::string number(const std::optional<double> &value) {
  return value ? number(*value) : "null";
}

std::string timestamp(double seconds) {
  std::ostringstream out = json_stream();
  out << std::fixed << std::setprecision(3) << seconds;
  return out.str();
}

}  // namespace

std::string json_quote(const std::string &text) {
  std::ostringstream out = json_stream();
  out << '"';
  for (const char c : text) {
    const auto byte = static_cast<unsigned char>(c);
    if (c == '"' || c == '\\') {
      out << '\\' << c;
    } else if (byte < 0x20) {
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
          << static_cast<int>(byte) << std::dec;
    } else {
      out << c;
    }
  }
  out << '"';
  return out.str();
}

std::string to_json_line(const RunStart &record) {
  std::ostringstream out = json_stream();
  out << R"({"kind":"run","step":)" << record.step
      << R"(,"time":)" << timestamp(record.time) << R"(,"world":)"
      << record.world << R"(,"tokens_per_step":)" << record.tokens_per_step
      << R"(,"device":)" << json_quote(record.device) << R"(,"flags":)"
      << json_quote(record.flags) << "}";
  return out.str();
}

std::string to_json_line(const TrainRecord &record) {
  std::ostringstream out = json_stream();
  out << R"({"kind":"train","step":)" << record.step
      << R"(,"time":)" << timestamp(record.time) << R"(,"loss":)"
      << number(record.loss) << R"(,"lr":)" << number(record.lr)
      << R"(,"grad_norm":)" << number(record.grad_norm) << R"(,"ms":)"
      << number(record.ms) << "}";
  return out.str();
}

std::string to_json_line(const ValRecord &record) {
  std::ostringstream out = json_stream();
  out << R"({"kind":"val","step":)" << record.step
      << R"(,"time":)" << timestamp(record.time) << R"(,"train_loss":)"
      << number(record.train_loss) << R"(,"val_loss":)"
      << number(record.val_loss) << "}";
  return out.str();
}

double unix_seconds() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

}  // namespace micrograd::gpt
