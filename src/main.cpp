#include <iostream>
#include <memory>
#include <string>

#include "micrograd/Autograd.h"
#include "micrograd/MNIST.h"
#include "micrograd/NN.h"

using namespace micrograd;

#ifndef MNIST_DATA_DIR
#define MNIST_DATA_DIR "data"
#endif

namespace {

double evaluate(const MNISTData &set, Sequential &net) {
  const NoGradGuard no_grad;
  size_t correct = 0;
  for (size_t i = 0; i < set.images.size(); i++) {
    auto pooled = avg_pool_2x2(set.images[i]);
    auto out = net.forward(pooled);
    if (out->argmax(1)->data()[0] == set.labels[i]->argmax(1)->data()[0]) {
      correct++;
    }
  }
  return 100.0 * static_cast<double>(correct) /
         static_cast<double>(set.images.size());
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const std::string data_dir = argc > 1 ? argv[1] : MNIST_DATA_DIR;

    auto train = load_mnist(data_dir + "/train-images-idx3-ubyte",
                            data_dir + "/train-labels-idx1-ubyte", 60000);
    auto test = load_mnist(data_dir + "/t10k-images-idx3-ubyte",
                           data_dir + "/t10k-labels-idx1-ubyte", 10000);

    auto l1 = std::make_shared<Linear>(196, 100);
    auto l2 = std::make_shared<Linear>(100, 10);
    Sequential net({l1, std::make_shared<ReLU>(), l2});

    SGD optimizer(net.parameters(), 0.01f);

    for (int epoch = 0; epoch < 30; epoch++) {
      double total_loss = 0.0;

      for (size_t i = 0; i < train.images.size(); i++) {
        auto pooled = avg_pool_2x2(train.images[i]);
        auto out = net.forward(pooled);
        auto loss = mse_loss(out, train.labels[i]);

        optimizer.zero_grad();
        loss->backward();
        optimizer.step();

        total_loss += static_cast<double>(loss->at({0}));
      }

      std::cout << "Epoch " << epoch + 1 << ": loss = "
                << total_loss / static_cast<double>(train.images.size())
                << ", test accuracy = " << evaluate(test, net) << "%\n";
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
