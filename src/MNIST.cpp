#include "micrograd/MNIST.h"

#include <cstddef>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "micrograd/Tensor.h"

MNISTData load_mnist(const std::string &images_path,
                     const std::string &labels_path, int sample_count) {
  MNISTData data;

  std::ifstream images_file(images_path, std::ios::binary);
  if (!images_file) {
    throw std::runtime_error("Could not open MNIST images: " + images_path);
  }

  std::ifstream labels_file(labels_path, std::ios::binary);
  if (!labels_file) {
    throw std::runtime_error("Could not open MNIST labels: " + labels_path);
  }

  images_file.seekg(16);
  labels_file.seekg(8);

  for (int i = 0; i < sample_count; i++) {
    std::vector<double> pixels(784);
    for (size_t j = 0; j < 784; j++) {
      unsigned char pixel;
      images_file.read(reinterpret_cast<char *>(&pixel), 1);
      pixels[j] = pixel / 255.0;
    }
    if (!images_file) {
      throw std::runtime_error("MNIST images truncated at sample " +
                               std::to_string(i) + ": " + images_path);
    }
    data.images.push_back(
        std::make_shared<Tensor>(std::vector<size_t>{1, 784}, pixels));
    unsigned char label;
    labels_file.read(reinterpret_cast<char *>(&label), 1);
    if (!labels_file) {
      throw std::runtime_error("MNIST labels truncated at sample " +
                               std::to_string(i) + ": " + labels_path);
    }
    if (label > 9) {
      throw std::runtime_error("MNIST label out of range at sample " +
                               std::to_string(i));
    }
    std::vector<double> one_hot(10, 0.0);
    one_hot[label] = 1.0;
    data.labels.push_back(
        std::make_shared<Tensor>(std::vector<size_t>{1, 10}, one_hot));
  }

  return data;
}