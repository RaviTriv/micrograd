#include "gpt/TrainState.h"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "micrograd/Random.h"

namespace micrograd::gpt {

std::string optimizer_state_path(const std::string &checkpoint_path) {
  return checkpoint_path + ".opt";
}

void save_optimizer_state(const std::string &path,
                          const TrainOptimizer &optimizer) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for saving: " + path);
  }
  optimizer.write_state(file);
  if (!file) {
    throw std::runtime_error("Failed while writing optimizer state: " + path);
  }
}

std::string data_state_path(const std::string &checkpoint_path) {
  return checkpoint_path + ".data";
}

void save_data_state(const std::string &path, const SampleOrder &order) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for saving: " + path);
  }
  order.write_state(file);
  if (!file) {
    throw std::runtime_error("Failed while writing data state: " + path);
  }
}

void load_data_state(const std::string &path, size_t samples_from_steps,
                     SampleOrder &order) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cout << "no data state at " << path << ", assuming "
              << samples_from_steps
              << " samples consumed from the optimizer step count\n";
    order.seek(samples_from_steps);
    return;
  }
  order.read_state(file);
}

std::string rng_state_path(const std::string &checkpoint_path) {
  return checkpoint_path + ".rng";
}

void save_rng_state(const std::string &path) {
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for saving: " + path);
  }
  file << global_rng();
  if (!file) {
    throw std::runtime_error("Failed while writing RNG state: " + path);
  }
}

void load_rng_state(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cout << "no RNG state at " << path
              << ", dropout will not match the original run\n";
    return;
  }
  std::mt19937_64 rng;
  file >> rng;
  if (!file) {
    throw std::runtime_error("Could not read RNG state: " + path);
  }
  global_rng() = rng;
}

void load_optimizer_state(const std::string &path, TrainOptimizer &optimizer) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for loading: " + path);
  }
  optimizer.read_state(file);
}

}  // namespace micrograd::gpt
