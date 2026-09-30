#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpt/BPE.h"
#include "gpt/CoreEval.h"
#include "gpt/GPTConfig.h"
#include "gpt/Model.h"
#include "micrograd/Autograd.h"
#include "micrograd/Device.h"
#include "micrograd/NN.h"
#include "micrograd/Scalar.h"
#include "micrograd/Tensor.h"

namespace {

using micrograd::Device;
using micrograd::load;
using micrograd::NoGradGuard;
using micrograd::scalar_t;
using micrograd::Tensor;
using micrograd::gpt::BPE;
using micrograd::gpt::ContinuationLogLikelihood;
using micrograd::gpt::CoreExample;
using micrograd::gpt::CoreMetric;
using micrograd::gpt::CoreTask;
using micrograd::gpt::CoreTaskResult;
using micrograd::gpt::evaluate_core_task;
using micrograd::gpt::gpt2_124m;
using micrograd::gpt::GPTConfig;
using micrograd::gpt::Model;

struct HellaswagRecord {
  std::string ctx;
  std::vector<std::string> endings;
  size_t label = 0;
};

struct Config {
  std::string data_path;
  std::string vocab_path;
  std::string merges_path;
  std::string checkpoint_path;
  size_t n_layer = gpt2_124m().n_layer;
  size_t n_head = gpt2_124m().n_head;
  size_t n_embd = gpt2_124m().n_embd;
  size_t block_size = gpt2_124m().block_size;
  Device device = Device::CPU;
};

Device parse_device(const std::string &value) {
  if (value == "cpu") {
    return Device::CPU;
  }
  if (value == "metal") {
    return Device::Metal;
  }
  if (value == "cuda") {
    return Device::CUDA;
  }
  throw std::invalid_argument("Unknown device: " + value);
}

Config parse_args(int argc, char **argv) {
  Config config;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto next_value = [&]() {
      if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + arg);
      }
      return std::string(argv[++i]);
    };

    if (arg == "--data") {
      config.data_path = next_value();
    } else if (arg == "--vocab") {
      config.vocab_path = next_value();
    } else if (arg == "--merges") {
      config.merges_path = next_value();
    } else if (arg == "--checkpoint") {
      config.checkpoint_path = next_value();
    } else if (arg == "--n-layer") {
      config.n_layer = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--n-head") {
      config.n_head = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--n-embd") {
      config.n_embd = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--block-size") {
      config.block_size = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--device") {
      config.device = parse_device(next_value());
    } else {
      throw std::invalid_argument("Unknown flag: " + arg);
    }
  }

  if (config.data_path.empty() || config.vocab_path.empty() ||
      config.merges_path.empty()) {
    throw std::invalid_argument(
        "hellaswag requires --data, --vocab, and --merges");
  }
  return config;
}

GPTConfig build_model_config(const Config &config) {
  GPTConfig model_config = gpt2_124m();
  model_config.n_layer = config.n_layer;
  model_config.n_head = config.n_head;
  model_config.n_kv_head = config.n_head;
  model_config.n_embd = config.n_embd;
  model_config.block_size = config.block_size;
  return model_config;
}

void skip_whitespace(const std::string &data, size_t &pos) {
  while (pos < data.size() &&
         std::isspace(static_cast<unsigned char>(data[pos]))) {
    pos++;
  }
}

uint32_t parse_hex4(const std::string &data, size_t pos) {
  uint32_t value = 0;
  for (size_t i = 0; i < 4; i++) {
    char c = data.at(pos + i);
    value <<= 4;
    if (c >= '0' && c <= '9') {
      value |= static_cast<uint32_t>(c - '0');
    } else if (c >= 'a' && c <= 'f') {
      value |= static_cast<uint32_t>(c - 'a' + 10);
    } else if (c >= 'A' && c <= 'F') {
      value |= static_cast<uint32_t>(c - 'A' + 10);
    } else {
      throw std::runtime_error("hellaswag: invalid unicode escape");
    }
  }
  return value;
}

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
    throw std::runtime_error("hellaswag: expected a json string");
  }
  pos++;
  std::string result;
  while (true) {
    char c = data.at(pos);
    if (c == '"') {
      pos++;
      return result;
    }
    if (c != '\\') {
      result.push_back(c);
      pos++;
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
            throw std::runtime_error("hellaswag: unpaired surrogate");
          }
          uint32_t low = parse_hex4(data, pos + 2);
          pos += 6;
          codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
        }
        append_utf8(result, codepoint);
        break;
      }
      default:
        throw std::runtime_error("hellaswag: unsupported json escape");
    }
  }
}

void skip_json_value(const std::string &data, size_t &pos) {
  skip_whitespace(data, pos);
  char c = data.at(pos);
  if (c == '"') {
    parse_json_string(data, pos);
    return;
  }
  if (c == '{' || c == '[') {
    char close = c == '{' ? '}' : ']';
    pos++;
    skip_whitespace(data, pos);
    if (data.at(pos) == close) {
      pos++;
      return;
    }
    while (true) {
      if (c == '{') {
        skip_whitespace(data, pos);
        parse_json_string(data, pos);
        skip_whitespace(data, pos);
        if (data.at(pos) != ':') {
          throw std::runtime_error("hellaswag: malformed json object");
        }
        pos++;
      }
      skip_json_value(data, pos);
      skip_whitespace(data, pos);
      char sep = data.at(pos);
      pos++;
      if (sep == close) {
        return;
      }
      if (sep != ',') {
        throw std::runtime_error("hellaswag: malformed json value");
      }
    }
  }

  size_t start = pos;
  while (pos < data.size() && data[pos] != ',' && data[pos] != '}' &&
         data[pos] != ']' &&
         !std::isspace(static_cast<unsigned char>(data[pos]))) {
    pos++;
  }
  if (pos == start) {
    throw std::runtime_error("hellaswag: malformed json value");
  }
}

std::vector<std::string> parse_string_array(const std::string &data,
                                            size_t &pos) {
  std::vector<std::string> values;
  skip_whitespace(data, pos);
  if (data.at(pos) != '[') {
    throw std::runtime_error("hellaswag: expected a json array");
  }
  pos++;
  skip_whitespace(data, pos);
  if (data.at(pos) == ']') {
    pos++;
    return values;
  }
  while (true) {
    skip_whitespace(data, pos);
    values.push_back(parse_json_string(data, pos));
    skip_whitespace(data, pos);
    char sep = data.at(pos);
    pos++;
    if (sep == ']') {
      break;
    }
    if (sep != ',') {
      throw std::runtime_error("hellaswag: malformed endings array");
    }
  }
  return values;
}

size_t parse_label(const std::string &data, size_t &pos) {
  skip_whitespace(data, pos);
  if (data.at(pos) == '"') {
    return static_cast<size_t>(std::stoul(parse_json_string(data, pos)));
  }
  size_t start = pos;
  while (pos < data.size() &&
         std::isdigit(static_cast<unsigned char>(data[pos]))) {
    pos++;
  }
  if (pos == start) {
    throw std::runtime_error("hellaswag: malformed label");
  }
  return static_cast<size_t>(std::stoul(data.substr(start, pos - start)));
}

HellaswagRecord parse_record(const std::string &line) {
  HellaswagRecord record;
  bool has_ctx = false;
  bool has_label = false;
  bool has_endings = false;

  size_t pos = 0;
  skip_whitespace(line, pos);
  if (line.at(pos) != '{') {
    throw std::runtime_error("hellaswag: malformed record");
  }
  pos++;
  skip_whitespace(line, pos);
  if (line.at(pos) != '}') {
    while (true) {
      skip_whitespace(line, pos);
      std::string key = parse_json_string(line, pos);
      skip_whitespace(line, pos);
      if (line.at(pos) != ':') {
        throw std::runtime_error("hellaswag: malformed record");
      }
      pos++;

      if (key == "ctx") {
        skip_whitespace(line, pos);
        record.ctx = parse_json_string(line, pos);
        has_ctx = true;
      } else if (key == "label") {
        record.label = parse_label(line, pos);
        has_label = true;
      } else if (key == "endings") {
        record.endings = parse_string_array(line, pos);
        has_endings = true;
      } else {
        skip_json_value(line, pos);
      }

      skip_whitespace(line, pos);
      char sep = line.at(pos);
      pos++;
      if (sep == '}') {
        break;
      }
      if (sep != ',') {
        throw std::runtime_error("hellaswag: malformed record");
      }
    }
  }

  if (!has_ctx || !has_label || !has_endings) {
    throw std::runtime_error(
        "hellaswag: record is missing ctx, label, or endings");
  }
  if (record.label >= record.endings.size()) {
    throw std::runtime_error("hellaswag: label is out of range");
  }
  return record;
}

std::vector<HellaswagRecord> load_records(const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("hellaswag: could not open " + path);
  }

  std::vector<HellaswagRecord> records;
  std::string line;
  while (std::getline(file, line)) {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }
    records.push_back(parse_record(line));
  }
  if (records.empty()) {
    throw std::runtime_error("hellaswag: " + path + " has no records");
  }
  return records;
}

CoreTask build_task(const std::vector<HellaswagRecord> &records,
                    const BPE &bpe) {
  CoreTask task;
  task.name = "hellaswag";
  task.metric = CoreMetric::kAccNorm;
  task.random_baseline = 0.25;
  task.examples.reserve(records.size());

  for (const HellaswagRecord &record : records) {
    CoreExample example;
    example.context = bpe.encode(record.ctx);
    example.gold_index = record.label;
    example.continuations.reserve(record.endings.size());
    for (const std::string &ending : record.endings) {
      example.continuations.push_back(bpe.encode(" " + ending));
    }
    task.examples.push_back(std::move(example));
  }
  return task;
}

std::shared_ptr<Tensor> token_tensor(const std::vector<int32_t> &tokens,
                                     Device device) {
  std::vector<scalar_t> values(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++) {
    values[i] = static_cast<scalar_t>(tokens[i]);
  }
  auto tensor = std::make_shared<Tensor>(std::vector<size_t>{1, tokens.size()},
                                         std::move(values));
  tensor->to(device);
  return tensor;
}

double sequence_log_likelihood(Model &model, size_t block_size, Device device,
                               const std::vector<int32_t> &context,
                               const std::vector<int32_t> &continuation) {
  if (context.empty()) {
    throw std::invalid_argument("hellaswag: example has an empty context");
  }
  if (continuation.empty()) {
    throw std::invalid_argument("hellaswag: continuation has no tokens");
  }

  std::vector<int32_t> sequence = context;
  sequence.insert(sequence.end(), continuation.begin(), continuation.end());
  if (sequence.size() > block_size) {
    throw std::runtime_error(
        "hellaswag: example exceeds the model's block size");
  }

  auto logits = model.forward(token_tensor(sequence, device));
  auto log_probs = logits->log_softmax(2);
  log_probs->to(Device::CPU);

  size_t continuation_start = sequence.size() - continuation.size();
  double log_likelihood = 0.0;
  for (size_t pos = continuation_start; pos < sequence.size(); pos++) {
    auto token_id = static_cast<size_t>(sequence[pos]);
    log_likelihood +=
        static_cast<double>(log_probs->at({0, pos - 1, token_id}));
  }
  return log_likelihood;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    Config config = parse_args(argc, argv);
    BPE bpe(config.vocab_path, config.merges_path);
    CoreTask task = build_task(load_records(config.data_path), bpe);

    GPTConfig model_config = build_model_config(config);
    Model model(bpe.vocab_size(), model_config);
    for (auto &parameter : model.parameters()) {
      parameter->to(config.device);
    }
    if (!config.checkpoint_path.empty()) {
      load(config.checkpoint_path, model);
    }

    const NoGradGuard no_grad;
    model.eval();

    ContinuationLogLikelihood scorer =
        [&](const std::vector<int32_t> &context,
            const std::vector<int32_t> &continuation) {
          return sequence_log_likelihood(model, config.block_size,
                                         config.device, context, continuation);
        };

    CoreTaskResult result = evaluate_core_task(task, scorer);
    std::cout << "hellaswag: " << result.example_count << " examples, "
              << "accuracy " << result.accuracy << ", " << "centered "
              << result.centered_accuracy << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hellaswag: " << error.what() << "\n";
    return 1;
  }
}
