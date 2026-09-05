#include <iostream>
#include <string>

#include "micrograd/MNIST.h"
#include "micrograd/NN.h"

#ifndef MNIST_DATA_DIR
#define MNIST_DATA_DIR "data"
#endif

namespace {

size_t argmax(const std::shared_ptr<Tensor> &row) {
  size_t best = 0;
  for (size_t j = 1; j < 10; j++) {
    if (row->at({0, j}) > row->at({0, best})) {
      best = j;
    }
  }
  return best;
}

double evaluate(const MNISTData &set, Linear &l1, Linear &l2) {
  size_t correct = 0;
  for (size_t i = 0; i < set.images.size(); i++) {
    auto pooled = avg_pool_2x2(set.images[i]);
    auto out = l2.forward(l1.forward(pooled)->relu());
    if (argmax(out) == argmax(set.labels[i])) {
      correct++;
    }
  }
  return 100.0 * static_cast<double>(correct) /
         static_cast<double>(set.images.size());
}

}  // namespace

int main(int argc, char **argv) {
  const std::string data_dir = argc > 1 ? argv[1] : MNIST_DATA_DIR;

  try {
    auto train = load_mnist(data_dir + "/train-images-idx3-ubyte",
                            data_dir + "/train-labels-idx1-ubyte", 60000);
    auto test = load_mnist(data_dir + "/t10k-images-idx3-ubyte",
                           data_dir + "/t10k-labels-idx1-ubyte", 10000);

    Linear l1(196, 100);
    Linear l2(100, 10);

    SGD optimizer({l1.weights(), l1.bias(), l2.weights(), l2.bias()}, 0.01);

    for (int epoch = 0; epoch < 30; epoch++) {
      double total_loss = 0.0;

      for (size_t i = 0; i < train.images.size(); i++) {
        auto pooled = avg_pool_2x2(train.images[i]);
        auto out = l2.forward(l1.forward(pooled)->relu());
        auto loss = mse_loss(out, train.labels[i]);

        optimizer.zero_grad();
        loss->backward();
        optimizer.step();

        total_loss += loss->at({0});
      }

      std::cout << "Epoch " << epoch + 1 << ": loss = "
                << total_loss / static_cast<double>(train.images.size())
                << ", test accuracy = " << evaluate(test, l1, l2) << "%"
                << std::endl;
    }

    save_model("mnist.bin", l1, l2);
    std::cout << "Saved trained model to mnist.bin" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n\n"
              << "Pass the dataset directory as the first argument, or "
                 "configure with -DFETCH_MNIST=ON to download it.\n";
    return 1;
  }

  return 0;
}
