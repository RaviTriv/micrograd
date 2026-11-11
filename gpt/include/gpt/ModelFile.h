#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpt/GPTConfig.h"
#include "gpt/Model.h"
#include "micrograd/Device.h"

namespace micrograd::gpt {

void save_model(const std::string &path, Model &model, const GPTConfig &config);
std::unique_ptr<Model> load_model(const std::string &path, Device device);
GPTConfig read_model_config(const std::string &path);

struct ContinuationScore {
  double logprob = 0.0;
  bool greedy = true;
  size_t scored = 0;
};

ContinuationScore score_continuation(Model &model, const GPTConfig &config,
                                     Device device,
                                     const std::vector<int32_t> &context,
                                     const std::vector<int32_t> &continuation);

std::string score_request(Model &model, const GPTConfig &config, Device device,
                          const std::string &line);

}  // namespace micrograd::gpt
