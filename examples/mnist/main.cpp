#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "MNIST.h"
#include "micrograd/Autograd.h"
#include "micrograd/Init.h"
#include "micrograd/NN.h"

using namespace micrograd;

#ifndef MNIST_DATA_DIR
#define MNIST_DATA_DIR "data"
#endif

namespace {

struct TrainingConfig {
  std::string data_dir = MNIST_DATA_DIR;
  size_t batch_size = 64;
  int epochs = 5;
  scalar_t lr = 1e-3f;
  uint64_t seed = 0;
  Device device = Device::CPU;
};

Device parse_device(const std::string &value) {
  if (value == "cpu") {
    return Device::CPU;
  }
  if (value == "cuda") {
    return Device::CUDA;
  }
  throw std::invalid_argument("Unknown device: " + value);
}

TrainingConfig parse_args(int argc, char **argv) {
  TrainingConfig config;
  std::vector<std::string> positional;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto next_value = [&]() {
      if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + arg);
      }
      return std::string(argv[++i]);
    };

    if (arg == "--batch-size") {
      config.batch_size = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--epochs") {
      config.epochs = std::stoi(next_value());
    } else if (arg == "--lr") {
      config.lr = std::stof(next_value());
    } else if (arg == "--seed") {
      config.seed = std::stoull(next_value());
    } else if (arg == "--device") {
      config.device = parse_device(next_value());
    } else {
      positional.push_back(arg);
    }
  }

  if (!positional.empty()) {
    config.data_dir = positional.front();
  }

  return config;
}

double evaluate(const data::Dataset &set, Sequential &net, size_t batch_size,
                Device device) {
  const NoGradGuard no_grad;
  size_t correct = 0;

  for (const data::Batch &batch : data::DataLoader(set, batch_size, false)) {
    auto pooled = avg_pool2d(batch.inputs, 2);
    pooled->to(device);

    auto out = net.forward(pooled);
    auto predictions = out->argmax(1);
    predictions->to(Device::CPU);

    auto targets = data::class_indices(*batch.targets);
    for (size_t i = 0; i < targets.size(); i++) {
      if (static_cast<size_t>(predictions->data()[i]) == targets[i]) {
        correct++;
      }
    }
  }

  return 100.0 * static_cast<double>(correct) /
         static_cast<double>(set.size());
}

}  // namespace

int main(int argc, char **argv) {
  try {
    TrainingConfig config = parse_args(argc, argv);

    auto train =
        load_mnist(config.data_dir + "/train-images-idx3-ubyte",
                   config.data_dir + "/train-labels-idx1-ubyte", 60000);
    auto test = load_mnist(config.data_dir + "/t10k-images-idx3-ubyte",
                           config.data_dir + "/t10k-labels-idx1-ubyte", 10000);

    auto l1 = std::make_shared<Linear>(196, 100);
    auto l2 = std::make_shared<Linear>(100, 10);
    init::kaiming_uniform_(l1->weights(), 196);
    init::kaiming_uniform_(l2->weights(), 100);
    Sequential net({l1, std::make_shared<ReLU>(), l2});
    for (auto &parameter : net.parameters()) {
      parameter->to(config.device);
    }

    AdamW optimizer(net.parameters(), config.lr);
    data::DataLoader train_loader(train, config.batch_size, true, config.seed);

    for (int epoch = 0; epoch < config.epochs; epoch++) {
      double total_loss = 0.0;
      train_loader.set_epoch(static_cast<size_t>(epoch));

      for (const data::Batch &batch : train_loader) {
        auto pooled = avg_pool2d(batch.inputs, 2);
        pooled->to(config.device);

        auto out = net.forward(pooled);
        auto targets = data::class_indices(*batch.targets);
        auto loss = cross_entropy(out, targets);

        optimizer.zero_grad();
        loss->backward();
        optimizer.step();

        loss->to(Device::CPU);
        total_loss += static_cast<double>(loss->at({})) *
                      static_cast<double>(targets.size());
      }

      std::cout << "Epoch " << epoch + 1 << ": loss = "
                << total_loss / static_cast<double>(train.size())
                << ", test accuracy = "
                << evaluate(test, net, config.batch_size, config.device)
                << "%\n";
    }

    save("mnist.bin", net);
    std::cout << "Saved trained model to mnist.bin\n";
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n\n"
              << "Pass the dataset directory as the first argument, or "
                 "configure with -DFETCH_MNIST=ON to download it.\n";
    return 1;
  }

  return 0;
}
