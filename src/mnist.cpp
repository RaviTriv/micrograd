#include "micrograd/mnist.h"
#include "micrograd/Tensor.h"
#include <cstddef>
#include <fstream>
#include <memory>

MNISTData load_mnist(const std::string &images_path,
                     const std::string &labels_path, int sample_count) {
  MNISTData data;

  std::ifstream images_file(images_path, std::ios::binary);
  std::ifstream labels_file(labels_path, std::ios::binary);

  images_file.seekg(16);
  labels_file.seekg(8);

  for (int i = 0; i < sample_count; i++) {
    std::vector<double> pixels(784);
    for (int j = 0; j < 784; j++) {
      unsigned char pixel;
      images_file.read(reinterpret_cast<char *>(&pixel), 1);
      pixels[j] = pixel / 255.0;
    }
    data.images.push_back(
        std::make_shared<Tensor>(std::vector<size_t>{1, 784}, pixels));
    unsigned char label;
    labels_file.read(reinterpret_cast<char *>(&label), 1);
    std::vector<double> one_hot(10, 0.0);
    one_hot[label] = 1.0;
    data.labels.push_back(
        std::make_shared<Tensor>(std::vector<size_t>{1, 10}, one_hot));
  }

  return data;
}