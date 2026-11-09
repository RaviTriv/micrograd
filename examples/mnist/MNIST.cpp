#include "MNIST.h"

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "micrograd/Tensor.h"

namespace micrograd {

namespace {
constexpr size_t kPixels = 784;
}

data::TensorDataset load_mnist(const std::string &images_path,
                               const std::string &labels_path,
                               size_t sample_count) {
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

  std::vector<unsigned char> raw_pixels(sample_count * kPixels);
  images_file.read(reinterpret_cast<char *>(raw_pixels.data()),
                   static_cast<std::streamsize>(raw_pixels.size()));
  if (!images_file) {
    throw std::runtime_error("MNIST images truncated, expected " +
                             std::to_string(sample_count) +
                             " samples: " + images_path);
  }

  std::vector<unsigned char> raw_labels(sample_count);
  labels_file.read(reinterpret_cast<char *>(raw_labels.data()),
                   static_cast<std::streamsize>(raw_labels.size()));
  if (!labels_file) {
    throw std::runtime_error("MNIST labels truncated, expected " +
                             std::to_string(sample_count) +
                             " samples: " + labels_path);
  }

  std::vector<scalar_t> pixels(raw_pixels.size());
  for (size_t i = 0; i < raw_pixels.size(); i++) {
    pixels[i] = static_cast<scalar_t>(raw_pixels[i]) / 255.0f;
  }

  std::vector<scalar_t> labels(sample_count);
  for (size_t i = 0; i < sample_count; i++) {
    if (raw_labels[i] > 9) {
      throw std::runtime_error("MNIST label out of range at sample " +
                               std::to_string(i));
    }
    labels[i] = static_cast<scalar_t>(raw_labels[i]);
  }

  auto images = Tensor::from_values({sample_count, kPixels}, std::move(pixels));
  auto targets = Tensor::from_values({sample_count}, std::move(labels));
  return {*images, *targets};
}

}  // namespace micrograd
