#include <iostream>

#include "micrograd/mnist.h"
#include "micrograd/nn.h"

int main() {

  auto train = load_mnist("data/train-images-idx3-ubyte",
                          "data/train-labels-idx1-ubyte", 60000);

  Linear l1(196, 100);
  Linear l2(100, 10);

  SGD optimizer({l1.weights(), l1.bias(), l2.weights(), l2.bias()}, 0.01);

  for (int epoch = 0; epoch < 30; epoch++) {
    double total_loss = 0.0;
    int correct = 0;

    for (size_t i = 0; i < train.images.size(); i++) {
      auto pooled = avg_pool_2x2(train.images[i]);
      auto x = l1.forward(pooled)->relu();
      auto out = l2.forward(x);
      auto loss = mse_loss(out, train.labels[i]);

      size_t pred = 0;
      size_t actual = 0;

      for (size_t j = 0; j < 10; j++) {
        if (out->at({0, j}) > out->at({0, pred})) {
          pred = j;
        }
        if (train.labels[i]->at({0, j}) > 0.5) {
          actual = j;
        }
      }
      if (pred == actual) {
        correct++;
      }

      optimizer.zero_grad();
      loss->backward();
      optimizer.step();

      total_loss += loss->at({0});
    }
    double avrg_loss = total_loss / train.images.size();
    double accuracy = 100.0 * correct / train.images.size();
    std::cout << "Epoch " << epoch + 1 << ": Loss = " << avrg_loss
              << ", Accuracy = " << accuracy << "%" << std::endl;
  }

  // save_model("models/mnist.bin", l1, l2);

  return 0;
}