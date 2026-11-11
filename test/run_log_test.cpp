#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <optional>
#include <string>

#include "gpt/JsonText.h"
#include "gpt/RunLog.h"

using micrograd::gpt::json_quote;
using micrograd::gpt::RunStart;
using micrograd::gpt::to_json_line;
using micrograd::gpt::TrainRecord;
using micrograd::gpt::ValRecord;

TEST(RunLog, TrainRecordIsOneJsonLine) {
  const TrainRecord record{.step = 12,
                           .time = 1700000000.5,
                           .loss = 3.25,
                           .lr = static_cast<double>(6e-4f),
                           .grad_norm = 0.5,
                           .ms = 108.25};
  EXPECT_EQ(to_json_line(record),
            R"({"kind":"train","step":12,"time":1700000000.500,"loss":3.25,)"
            R"("lr":0.0006,"grad_norm":0.5,"ms":108.25})");
}

TEST(RunLog, MissingAndNonFiniteValuesBecomeNull) {
  const TrainRecord record{.step = 1,
                           .time = 0.0,
                           .loss = std::numeric_limits<double>::quiet_NaN(),
                           .lr = std::numeric_limits<double>::infinity(),
                           .grad_norm = std::nullopt,
                           .ms = 1.0};
  const std::string line = to_json_line(record);
  EXPECT_NE(line.find(R"("loss":null)"), std::string::npos) << line;
  EXPECT_NE(line.find(R"("lr":null)"), std::string::npos) << line;
  EXPECT_NE(line.find(R"("grad_norm":null)"), std::string::npos) << line;
}

TEST(RunLog, ValRecordWithoutTrainLoss) {
  const ValRecord record{.step = 76000,
                         .time = 2.0,
                         .train_loss = std::nullopt,
                         .val_loss = 3.0625};
  EXPECT_EQ(to_json_line(record),
            R"({"kind":"val","step":76000,"time":2.000,"train_loss":null,)"
            R"("val_loss":3.0625})");
}

TEST(RunLog, RunStartQuotesFlags) {
  const RunStart record{.step = 1000,
                        .time = 3.0,
                        .world = 8,
                        .tokens_per_step = 131072,
                        .device = "NVIDIA A100",
                        .flags = "--log a\"b\\c"};
  EXPECT_EQ(to_json_line(record),
            R"({"kind":"run","step":1000,"time":3.000,"world":8,)"
            R"("tokens_per_step":131072,"device":"NVIDIA A100",)"
            R"("flags":"--log a\"b\\c"})");
}

TEST(RunLog, QuotedTextParsesBack) {
  const std::string text = "tab\there \"quoted\" back\\slash\nnew line \x01";
  const std::string quoted = json_quote(text);
  EXPECT_EQ(quoted.find('\n'), std::string::npos);
  size_t pos = 0;
  EXPECT_EQ(micrograd::gpt::parse_json_string(quoted, pos), text);
  EXPECT_EQ(pos, quoted.size());
}
