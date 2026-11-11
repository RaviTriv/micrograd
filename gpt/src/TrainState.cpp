#include "gpt/TrainState.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <istream>
#include <optional>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "micrograd/Random.h"

namespace micrograd::gpt {

namespace {

constexpr uint32_t kOptimizerMagic = 0x5453504f;
constexpr uint32_t kDataMagic = 0x54534144;
constexpr uint32_t kRngMagic = 0x5453474e;
constexpr uint32_t kStateVersion = 1;
constexpr uint64_t kMaxNameLength = 4096;
constexpr uint64_t kMaxRank = 8;
constexpr uint64_t kMaxRngTextLength = 65536;

template <typename T>
void write_value(std::ostream &out, T value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

template <typename T>
T read_value(std::istream &in, const std::string &path) {
  T value{};
  in.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error("Train state file is truncated: " + path);
  }
  return value;
}

void write_atomically(const std::string &path,
                      const std::function<void(std::ostream &)> &write) {
  const std::filesystem::path target(path);
  std::filesystem::path temporary = target;
  temporary += ".tmp";
  try {
    std::ofstream file(temporary, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open file for saving: " +
                               temporary.string());
    }
    write(file);
    file.close();
    if (!file) {
      throw std::runtime_error("Failed while writing train state: " + path);
    }
    std::filesystem::rename(temporary, target);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

std::ifstream open_for_loading(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for loading: " + path);
  }
  return file;
}

void write_header(std::ostream &out, uint32_t magic, size_t step) {
  write_value(out, magic);
  write_value(out, kStateVersion);
  write_value(out, static_cast<uint64_t>(step));
}

size_t read_header(std::istream &in, const std::string &path, uint32_t magic,
                   const std::string &what) {
  if (read_value<uint32_t>(in, path) != magic) {
    throw std::runtime_error(what +
                             " state file has the wrong magic number, or was "
                             "written before state files had headers: " +
                             path);
  }
  const auto version = read_value<uint32_t>(in, path);
  if (version != kStateVersion) {
    throw std::runtime_error("Unsupported " + what + " state version " +
                             std::to_string(version) + ": " + path);
  }
  return static_cast<size_t>(read_value<uint64_t>(in, path));
}

void expect_step(const std::string &path, size_t step,
                 std::optional<size_t> expected_step) {
  if (expected_step.has_value() && step != *expected_step) {
    throw std::runtime_error(path + " is from step " + std::to_string(step) +
                             ", expected step " +
                             std::to_string(*expected_step));
  }
}

void expect_end(std::istream &in, const std::string &path) {
  if (in.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("Train state file has trailing bytes: " + path);
  }
}

[[noreturn]] void throw_mismatch(const std::string &path,
                                 const std::string &field,
                                 const std::string &saved,
                                 const std::string &live) {
  throw std::runtime_error("Optimizer state " + field + " mismatch in " + path +
                           ": file has " + saved + ", optimizer has " + live);
}

std::string shape_text(const std::vector<size_t> &shape) {
  std::string text = "[";
  for (size_t i = 0; i < shape.size(); i++) {
    text += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
  }
  return text + "]";
}

void write_string(std::ostream &out, const std::string &value) {
  write_value(out, static_cast<uint64_t>(value.size()));
  out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

std::string read_string(std::istream &in, const std::string &path,
                        uint64_t max_length) {
  const auto length = read_value<uint64_t>(in, path);
  if (length > max_length) {
    throw std::runtime_error("String too long in train state file: " + path);
  }
  std::string name(static_cast<size_t>(length), '\0');
  in.read(name.data(), static_cast<std::streamsize>(name.size()));
  if (!in) {
    throw std::runtime_error("Train state file is truncated: " + path);
  }
  return name;
}

std::vector<size_t> read_shape(std::istream &in, const std::string &path) {
  const auto rank = read_value<uint64_t>(in, path);
  if (rank > kMaxRank) {
    throw std::runtime_error("Parameter rank too large in " + path);
  }
  std::vector<size_t> shape(static_cast<size_t>(rank));
  for (auto &dim : shape) {
    dim = static_cast<size_t>(read_value<uint64_t>(in, path));
  }
  return shape;
}

const char *muon_text(bool muon) { return muon ? "Muon on" : "Muon off"; }

}  // namespace

std::string optimizer_state_path(const std::string &checkpoint_path) {
  return checkpoint_path + ".opt";
}

void save_optimizer_state(const std::string &path,
                          const TrainOptimizer &optimizer) {
  write_atomically(path, [&](std::ostream &out) {
    write_header(out, kOptimizerMagic, optimizer.step_count());
    write_value(out, static_cast<uint64_t>(optimizer.parameter_count()));
    for (size_t i = 0; i < optimizer.parameter_count(); i++) {
      write_string(out, optimizer.parameter_name(i));
      const auto &shape = optimizer.parameter_shape(i);
      write_value(out, static_cast<uint64_t>(shape.size()));
      for (size_t dim : shape) {
        write_value(out, static_cast<uint64_t>(dim));
      }
      write_value(out, static_cast<uint8_t>(optimizer.is_muon(i) ? 1 : 0));
    }
    optimizer.write_moments(out);
  });
}

void load_optimizer_state(const std::string &path, TrainOptimizer &optimizer,
                          std::optional<size_t> expected_step) {
  std::ifstream file = open_for_loading(path);
  const size_t step = read_header(file, path, kOptimizerMagic, "optimizer");
  expect_step(path, step, expected_step);
  const auto count = static_cast<size_t>(read_value<uint64_t>(file, path));
  if (count != optimizer.parameter_count()) {
    throw_mismatch(path, "tensor count", std::to_string(count),
                   std::to_string(optimizer.parameter_count()));
  }
  for (size_t i = 0; i < count; i++) {
    const std::string &live_name = optimizer.parameter_name(i);
    const std::string name = read_string(file, path, kMaxNameLength);
    if (name != live_name) {
      throw_mismatch(path, "name at tensor " + std::to_string(i), name,
                     live_name);
    }
    const std::vector<size_t> shape = read_shape(file, path);
    if (shape != optimizer.parameter_shape(i)) {
      throw_mismatch(path, "shape of " + name, shape_text(shape),
                     shape_text(optimizer.parameter_shape(i)));
    }
    const bool muon = read_value<uint8_t>(file, path) != 0;
    if (muon != optimizer.is_muon(i)) {
      throw_mismatch(path, "Muon flag of " + name, muon_text(muon),
                     muon_text(optimizer.is_muon(i)));
    }
  }
  optimizer.read_moments(file, step);
  expect_end(file, path);
}

std::string data_state_path(const std::string &checkpoint_path) {
  return checkpoint_path + ".data";
}

void save_data_state(const std::string &path, const SampleOrder &order,
                     size_t step) {
  write_atomically(path, [&](std::ostream &out) {
    write_header(out, kDataMagic, step);
    order.write_state(out);
  });
}

void load_data_state(const std::string &path, size_t samples_from_steps,
                     SampleOrder &order, std::optional<size_t> expected_step) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cout << "no data state at " << path << ", assuming "
              << samples_from_steps
              << " samples consumed from the optimizer step count\n";
    order.seek(samples_from_steps);
    return;
  }
  expect_step(path, read_header(file, path, kDataMagic, "data"), expected_step);
  order.read_state(file);
  expect_end(file, path);
}

std::string rng_state_path(const std::string &checkpoint_path) {
  return checkpoint_path + ".rng";
}

void save_rng_state(const std::string &path, size_t step) {
  write_atomically(path, [&](std::ostream &out) {
    write_header(out, kRngMagic, step);
    std::ostringstream text;
    text << global_rng();
    write_string(out, text.str());
  });
}

void load_rng_state(const std::string &path,
                    std::optional<size_t> expected_step) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cout << "no RNG state at " << path
              << ", dropout will not match the original run\n";
    return;
  }
  expect_step(path, read_header(file, path, kRngMagic, "RNG"), expected_step);
  std::istringstream text(read_string(file, path, kMaxRngTextLength));
  expect_end(file, path);
  std::mt19937_64 rng;
  text >> rng;
  if (!text) {
    throw std::runtime_error("Could not read RNG state: " + path);
  }
  global_rng() = rng;
}

std::string checkpoint_set_path(const std::string &checkpoint_path,
                                size_t step) {
  return checkpoint_path + ".step" + std::to_string(step);
}

void write_latest_checkpoint(const std::string &checkpoint_path, size_t step) {
  write_atomically(checkpoint_path + ".latest",
                   [&](std::ostream &out) { out << step << "\n"; });
}

std::optional<size_t> read_latest_checkpoint(
    const std::string &checkpoint_path) {
  const std::string path = checkpoint_path + ".latest";
  std::ifstream file(path);
  if (!file.is_open()) {
    return std::nullopt;
  }
  size_t step = 0;
  file >> step;
  if (!file) {
    throw std::runtime_error("Could not read the latest checkpoint step: " +
                             path);
  }
  return step;
}

std::string resolve_resume_path(const std::string &checkpoint_path) {
  const std::optional<size_t> step = read_latest_checkpoint(checkpoint_path);
  return step ? checkpoint_set_path(checkpoint_path, *step) : checkpoint_path;
}

void remove_checkpoint_set(const std::string &set_path, bool keep_model) {
  const std::string paths[] = {optimizer_state_path(set_path),
                               data_state_path(set_path),
                               rng_state_path(set_path), set_path};
  const size_t count = keep_model ? 3 : 4;
  for (size_t i = 0; i < count; i++) {
    std::error_code ignored;
    std::filesystem::remove(paths[i], ignored);
  }
}

size_t check_train_state(const std::string &checkpoint_path) {
  const std::string paths[] = {optimizer_state_path(checkpoint_path),
                               data_state_path(checkpoint_path),
                               rng_state_path(checkpoint_path)};
  const uint32_t magics[] = {kOptimizerMagic, kDataMagic, kRngMagic};
  const std::string names[] = {"optimizer", "data", "RNG"};
  size_t steps[3] = {};
  for (size_t i = 0; i < 3; i++) {
    std::ifstream file = open_for_loading(paths[i]);
    steps[i] = read_header(file, paths[i], magics[i], names[i]);
  }
  if (steps[0] != steps[1] || steps[0] != steps[2]) {
    throw std::runtime_error(
        "Torn checkpoint " + checkpoint_path + ": optimizer state from step " +
        std::to_string(steps[0]) + ", data state from step " +
        std::to_string(steps[1]) + ", RNG state from step " +
        std::to_string(steps[2]));
  }
  return steps[0];
}

}  // namespace micrograd::gpt
