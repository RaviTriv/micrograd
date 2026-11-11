#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "gpt/GPTConfig.h"
#include "gpt/Model.h"
#include "gpt/ModelFile.h"
#include "gpt/TrainConfig.h"
#include "micrograd/Autograd.h"
#include "micrograd/Device.h"

namespace {

using micrograd::Device;
using micrograd::NoGradGuard;
using micrograd::gpt::GPTConfig;
using micrograd::gpt::load_model;
using micrograd::gpt::Model;
using micrograd::gpt::parse_device;
using micrograd::gpt::read_model_config;
using micrograd::gpt::score_request;

struct Config {
  std::string checkpoint_path;
  Device device = Device::CPU;
};

Config parse_args(int argc, char **argv) {
  Config config;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (i + 1 >= argc) {
      throw std::invalid_argument("Missing value for " + arg);
    }
    std::string value = argv[++i];
    if (arg == "--checkpoint") {
      config.checkpoint_path = value;
    } else if (arg == "--device") {
      config.device = parse_device(value);
    } else {
      throw std::invalid_argument("Unknown flag: " + arg);
    }
  }
  if (config.checkpoint_path.empty()) {
    throw std::invalid_argument("gpt_score requires --checkpoint");
  }
  return config;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    Config config = parse_args(argc, argv);
    GPTConfig model_config = read_model_config(config.checkpoint_path);
    std::unique_ptr<Model> model =
        load_model(config.checkpoint_path, config.device);

    const NoGradGuard no_grad;
    model->eval();
    std::cerr << "gpt_score: loaded " << config.checkpoint_path << " ("
              << model_config.n_layer << " layers, block "
              << model_config.block_size << ", vocab "
              << model_config.vocab_size << ")\n";

    std::string line;
    while (std::getline(std::cin, line)) {
      std::cout << score_request(*model, model_config, config.device, line)
                << "\n";
      std::cout.flush();
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gpt_score: " << error.what() << "\n";
    return 1;
  }
}
