#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace micrograd::gpt {

struct RunStart {
  size_t step = 0;
  double time = 0.0;
  size_t world = 1;
  size_t tokens_per_step = 0;
  std::string device;
  std::string flags;
};

struct TrainRecord {
  size_t step = 0;
  double time = 0.0;
  double loss = 0.0;
  double lr = 0.0;
  std::optional<double> grad_norm;
  double ms = 0.0;
};

struct ValRecord {
  size_t step = 0;
  double time = 0.0;
  std::optional<double> train_loss;
  double val_loss = 0.0;
};

std::string json_quote(const std::string &text);

std::string to_json_line(const RunStart &record);
std::string to_json_line(const TrainRecord &record);
std::string to_json_line(const ValRecord &record);

double unix_seconds();

}  // namespace micrograd::gpt
