#include "gpt/ModelFile.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <system_error>
#include <utility>

#include "gpt/JsonText.h"
#include "micrograd/DType.h"
#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"
#include "micrograd/nn/Serialize.h"

namespace micrograd::gpt {

namespace {

std::shared_ptr<Tensor> token_tensor(const std::vector<int32_t> &tokens,
                                     Device device) {
  std::vector<scalar_t> values(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++) {
    values[i] = static_cast<scalar_t>(tokens[i]);
  }
  auto tensor = Tensor::from_values(std::vector<size_t>{1, tokens.size()},
                                    std::move(values));
  tensor->to(device);
  return tensor;
}

void require_in_vocab(const std::vector<int32_t> &tokens, size_t vocab_size) {
  for (int32_t token : tokens) {
    if (token < 0 || std::cmp_greater_equal(token, vocab_size)) {
      throw std::invalid_argument("token id " + std::to_string(token) +
                                  " is outside the vocabulary of " +
                                  std::to_string(vocab_size));
    }
  }
}

char peek(const std::string &text, size_t pos) {
  return pos < text.size() ? text[pos] : '\0';
}

void skip_space(const std::string &text, size_t &pos) {
  while (pos < text.size() &&
         std::isspace(static_cast<unsigned char>(text[pos]))) {
    pos++;
  }
}

void expect(const std::string &text, size_t &pos, char c) {
  if (peek(text, pos) != c) {
    throw std::invalid_argument(std::string("expected '") + c + "' at column " +
                                std::to_string(pos + 1));
  }
  pos++;
}

std::vector<int32_t> parse_token_array(const std::string &text, size_t &pos) {
  std::vector<int32_t> tokens;
  expect(text, pos, '[');
  skip_space(text, pos);
  if (peek(text, pos) == ']') {
    pos++;
    return tokens;
  }
  while (true) {
    skip_space(text, pos);
    if (!std::isdigit(static_cast<unsigned char>(peek(text, pos)))) {
      throw std::invalid_argument(
          "expected a non-negative integer token id at column " +
          std::to_string(pos + 1));
    }
    const char *begin = text.data() + pos;
    const char *end = text.data() + text.size();
    int32_t token = 0;
    auto [ptr, error] = std::from_chars(begin, end, token);
    if (error != std::errc()) {
      throw std::invalid_argument("token id at column " +
                                  std::to_string(pos + 1) + " is too large");
    }
    tokens.push_back(token);
    pos += static_cast<size_t>(ptr - begin);
    skip_space(text, pos);
    if (peek(text, pos) == ']') {
      pos++;
      return tokens;
    }
    expect(text, pos, ',');
  }
}

std::pair<std::vector<int32_t>, std::vector<int32_t>> parse_request(
    const std::string &line) {
  std::vector<int32_t> context;
  std::vector<int32_t> continuation;
  bool has_context = false;
  bool has_continuation = false;

  size_t pos = 0;
  skip_space(line, pos);
  expect(line, pos, '{');
  skip_space(line, pos);
  if (peek(line, pos) == '}') {
    pos++;
  } else {
    while (true) {
      skip_space(line, pos);
      if (peek(line, pos) != '"') {
        throw std::invalid_argument("expected a key at column " +
                                    std::to_string(pos + 1));
      }
      std::string key = parse_json_string(line, pos);
      skip_space(line, pos);
      expect(line, pos, ':');
      skip_space(line, pos);
      if (key == "context" && !has_context) {
        context = parse_token_array(line, pos);
        has_context = true;
      } else if (key == "continuation" && !has_continuation) {
        continuation = parse_token_array(line, pos);
        has_continuation = true;
      } else {
        throw std::invalid_argument(
            "request keys must be context and continuation, each once");
      }
      skip_space(line, pos);
      if (peek(line, pos) == '}') {
        pos++;
        break;
      }
      expect(line, pos, ',');
    }
  }
  skip_space(line, pos);
  if (pos != line.size()) {
    throw std::invalid_argument("unexpected text after the request at column " +
                                std::to_string(pos + 1));
  }
  if (!has_context || !has_continuation) {
    throw std::invalid_argument("request needs context and continuation");
  }
  return {std::move(context), std::move(continuation)};
}

std::string format_double(double value) {
  std::array<char, 32> buffer{};
  auto [ptr, error] =
      std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  if (error != std::errc()) {
    throw std::runtime_error("could not format a log-probability");
  }
  return {buffer.data(), ptr};
}

}  // namespace

void save_model(const std::string &path, Model &model,
                const GPTConfig &config) {
  save(path, model, architecture_metadata(config));
}

GPTConfig read_model_config(const std::string &path) {
  return config_from_metadata(load_metadata(path));
}

std::unique_ptr<Model> load_model(const std::string &path, Device device) {
  GPTConfig config = read_model_config(path);
  auto model = std::make_unique<Model>(config.vocab_size, config);
  load(path, *model);
  for (auto &parameter : model->parameters()) {
    parameter->to(device);
  }
  return model;
}

ContinuationScore score_continuation(Model &model, const GPTConfig &config,
                                     Device device,
                                     const std::vector<int32_t> &context,
                                     const std::vector<int32_t> &continuation) {
  if (continuation.empty()) {
    throw std::invalid_argument("continuation has no tokens");
  }
  if (continuation.size() > config.block_size) {
    throw std::invalid_argument("continuation has " +
                                std::to_string(continuation.size()) +
                                " tokens, more than the block size " +
                                std::to_string(config.block_size));
  }
  require_in_vocab(context, config.vocab_size);
  require_in_vocab(continuation, config.vocab_size);

  std::vector<int32_t> sequence = context;
  sequence.insert(sequence.end(), continuation.begin(), continuation.end());
  const size_t window = std::min(sequence.size() - 1, config.block_size);
  if (window == 0) {
    throw std::invalid_argument(
        "an empty context needs at least two continuation tokens");
  }
  const size_t first = sequence.size() - 1 - window;
  const size_t start = std::max(context.size(), first + 1);

  std::vector<int32_t> input(
      sequence.begin() + static_cast<int64_t>(first),
      sequence.begin() + static_cast<int64_t>(first + window));
  auto logits =
      model.forward(token_tensor(input, device))->cast(DType::kFloat32);
  logits->to(Device::CPU);

  ContinuationScore score;
  for (size_t target = start; target < sequence.size(); target++) {
    const size_t row = target - 1 - first;
    size_t argmax = 0;
    double max_logit = static_cast<double>(logits->at({0, row, 0}));
    for (size_t v = 1; v < config.vocab_size; v++) {
      const auto logit = static_cast<double>(logits->at({0, row, v}));
      if (logit > max_logit) {
        max_logit = logit;
        argmax = v;
      }
    }
    double sum_exp = 0.0;
    for (size_t v = 0; v < config.vocab_size; v++) {
      sum_exp +=
          std::exp(static_cast<double>(logits->at({0, row, v})) - max_logit);
    }
    const auto token = static_cast<size_t>(sequence[target]);
    score.logprob += static_cast<double>(logits->at({0, row, token})) -
                     max_logit - std::log(sum_exp);
    score.greedy = score.greedy && argmax == token;
    score.scored++;
  }
  return score;
}

std::string score_request(Model &model, const GPTConfig &config, Device device,
                          const std::string &line) {
  try {
    auto [context, continuation] = parse_request(line);
    ContinuationScore score =
        score_continuation(model, config, device, context, continuation);
    std::string response = R"({"logprob": )" + format_double(score.logprob);
    response += R"(, "greedy": )";
    response += score.greedy ? "true" : "false";
    response += R"(, "scored": )" + std::to_string(score.scored) + "}";
    return response;
  } catch (const std::exception &error) {
    std::string response = R"({"error": ")";
    response += error.what();
    response += R"("})";
    return response;
  }
}

}  // namespace micrograd::gpt
